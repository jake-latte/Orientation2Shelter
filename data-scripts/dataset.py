from __future__ import annotations

import math
import numpy as np
import polars as pl
import torch
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader, get_worker_info
from typing import Dict, Iterator, List, Optional, Tuple

BEHAV_COLS = ["hdir", "hsa", "angular_speed", "translational_speed", "x", "y", "dist"]
NORMS = [math.pi, math.pi, 3 * math.pi, 1024.0, 1024.0, 1024.0, 1024.0]
ID_COLS = ["animal_idx", "exp_idx"]
FRAME_COL = "frame"


class JLDataset(IterableDataset):
    """
    Streams behavioural + neural data from one or more sessions.

    Each sample is a dict with:
      "inputs"  : float32 tensor of shape (n_behav_ctx + n_files,)
                  = normalised behaviour (± temporal context) + one-hot session ID
      "targets" : float32 tensor of shape (total_n_neurons,)
                  = firing rates for this session's neurons; NaN elsewhere

    neuron_labels[i] = (start, end) slice of the global neuron index for files[i].
    total_n_neurons   = sum of all session neuron counts.
    """

    def __init__(
        self,
        files: List[str],
        neuron_labels: List[Tuple[int, int]],
        total_n_neurons: int,
        past_embed: int,
        future_embed: int,
        n_folds: int,
        fold_idx: int,
        train: bool,
        seed: int = 0,
        shuffle_buffer_size: int = 100_000,
        equalize_hdir: bool = True,
        n_hdir_bins: int = 30,
        hdir_range: Optional[Tuple[float, float]] = None,
    ):
        super().__init__()
        self.files = list(files)
        self.neuron_labels = list(neuron_labels)
        self.total_n_neurons = total_n_neurons
        self.past_embed = past_embed
        self.future_embed = future_embed
        self.n_folds = n_folds
        self.fold_idx = fold_idx
        self.train = train
        self.seed = seed
        self.shuffle_buffer_size = shuffle_buffer_size
        self.equalize_hdir = equalize_hdir
        self.n_hdir_bins = n_hdir_bins
        self.hdir_range = hdir_range

    # ------------------------------------------------------------------
    # Worker sharding
    # ------------------------------------------------------------------

    def _worker_files(self) -> Tuple[List[int], List[str], List[Tuple[int, int]]]:
        worker = get_worker_info()
        if worker is None:
            return list(range(len(self.files))), self.files, self.neuron_labels
        ids = list(range(worker.id, len(self.files), worker.num_workers))
        return ids, [self.files[i] for i in ids], [self.neuron_labels[i] for i in ids]

    # ------------------------------------------------------------------
    # Head-direction balancing
    # ------------------------------------------------------------------

    def _balanced_hdir_order(
        self, hdir: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
        n = len(hdir)
        if n == 0:
            return np.empty(0, dtype=np.int64)
        x = hdir.astype(np.float64)
        if self.hdir_range is not None:
            lo, hi = self.hdir_range
        else:
            lo, hi = float(np.nanmin(x)), float(np.nanmax(x))
        if not (np.isfinite(lo) and np.isfinite(hi) and lo != hi):
            return rng.permutation(n)
        edges = np.linspace(lo, hi, self.n_hdir_bins + 1)
        bin_ids = np.clip(np.digitize(x, edges[1:-1], right=False), 0, self.n_hdir_bins - 1)
        bins = [rng.permutation(np.flatnonzero(bin_ids == b)) for b in range(self.n_hdir_bins)]
        cursors = np.zeros(self.n_hdir_bins, dtype=np.int64)
        order: List[int] = []
        while True:
            any_left = False
            for b in rng.permutation(self.n_hdir_bins):
                if cursors[b] < len(bins[b]):
                    order.append(int(bins[b][cursors[b]]))
                    cursors[b] += 1
                    any_left = True
            if not any_left:
                break
        return np.asarray(order, dtype=np.int64)

    # ------------------------------------------------------------------
    # Per-file stream
    # ------------------------------------------------------------------

    def _context_cols(self) -> List[str]:
        cols = list(BEHAV_COLS)
        cols += [f"{c}-{k}" for k in range(1, self.past_embed + 1) for c in BEHAV_COLS]
        cols += [f"{c}+{k}" for k in range(1, self.future_embed + 1) for c in BEHAV_COLS]
        return cols

    def _stream(
        self,
        ids: List[int],
        files: List[str],
        neuron_labels: List[Tuple[int, int]],
    ) -> Iterator[Dict]:
        worker = get_worker_info()
        worker_id = 0 if worker is None else worker.id
        rng = np.random.default_rng(self.seed + worker_id)

        ctx_cols = self._context_cols()
        n_sessions = len(self.files)

        order = list(range(len(files)))
        rng.shuffle(order)

        for idx in order:
            file_id = ids[idx]
            file = files[idx]
            n_start, n_end = neuron_labels[idx]

            schema = pl.read_csv(file, n_rows=0).schema
            target_cols = sorted(c for c in schema if c.isdigit())

            lf = (
                pl.scan_csv(file)
                .sort(ID_COLS + [FRAME_COL])
                .with_columns(
                    [(pl.col(c) / norm).alias(c) for c, norm in zip(BEHAV_COLS, NORMS)]
                )
            )

            shift_exprs, frame_checks = [], []
            for k in range(1, self.past_embed + 1):
                shift_exprs.extend(
                    pl.col(c).shift(k).over(ID_COLS).alias(f"{c}-{k}")
                    for c in BEHAV_COLS
                )
                frame_checks.append(
                    pl.col(FRAME_COL).shift(k).over(ID_COLS).eq(pl.col(FRAME_COL) - k)
                )
            for k in range(1, self.future_embed + 1):
                shift_exprs.extend(
                    pl.col(c).shift(-k).over(ID_COLS).alias(f"{c}+{k}")
                    for c in BEHAV_COLS
                )
                frame_checks.append(
                    pl.col(FRAME_COL).shift(-k).over(ID_COLS).eq(pl.col(FRAME_COL) + k)
                )

            valid = pl.all_horizontal(frame_checks) if frame_checks else pl.lit(True)

            df = (
                lf.with_columns(shift_exprs + [valid.alias("_valid")])
                .filter(pl.col("_valid"))
                .with_row_index("_row")
                .collect()
            )

            if df.height == 0:
                continue

            n_rows = df.height
            t_start = (self.fold_idx * n_rows) // self.n_folds
            t_end = ((self.fold_idx + 1) * n_rows) // self.n_folds

            if self.train:
                df = df.filter((pl.col("_row") < t_start) | (pl.col("_row") >= t_end))
            else:
                df = df.filter((pl.col("_row") >= t_start) & (pl.col("_row") < t_end))

            if df.height == 0:
                continue

            inputs_np = df.select(ctx_cols).to_numpy().astype(np.float32, copy=False)
            targets_np = df.select(target_cols).to_numpy().astype(np.float32, copy=False)

            if self.equalize_hdir:
                hdir_idx = ctx_cols.index("hdir")
                sample_order = self._balanced_hdir_order(inputs_np[:, hdir_idx], rng)
            else:
                sample_order = rng.permutation(len(inputs_np))

            # Build session one-hot once per file (constant for all samples)
            session_oh = F.one_hot(
                torch.tensor(file_id, dtype=torch.long), num_classes=n_sessions
            ).float()

            for i in sample_order:
                i = int(i)
                behav = torch.from_numpy(inputs_np[i])
                inp = torch.cat([behav, session_oh])

                targets_full = torch.full((self.total_n_neurons,), float("nan"))
                targets_full[n_start:n_end] = torch.from_numpy(targets_np[i])

                yield {"inputs": inp, "targets": targets_full}

    # ------------------------------------------------------------------
    # IterableDataset interface — reservoir shuffle buffer
    # ------------------------------------------------------------------

    def __iter__(self) -> Iterator[Dict]:
        ids, files, neuron_labels = self._worker_files()
        stream = self._stream(ids, files, neuron_labels)

        worker = get_worker_info()
        worker_id = 0 if worker is None else worker.id
        rng = np.random.default_rng(self.seed + 10_000 + worker_id)

        buffer: List[Dict] = []
        try:
            while len(buffer) < self.shuffle_buffer_size:
                buffer.append(next(stream))
        except StopIteration:
            pass

        while buffer:
            idx = int(rng.integers(len(buffer)))
            sample = buffer[idx]
            try:
                buffer[idx] = next(stream)
            except StopIteration:
                buffer.pop(idx)
            yield sample


def collate_fn(batch: List[Dict]) -> Dict:
    # All targets are pre-expanded to (total_n_neurons,) so plain stack works.
    return {
        "inputs": torch.stack([b["inputs"] for b in batch]),
        "targets": torch.stack([b["targets"] for b in batch]),
    }


def build_loaders(
    files: List[str],
    neuron_labels: List[Tuple[int, int]],
    total_n_neurons: int,
    config,
    num_workers: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader]:
    if num_workers is None:
        num_workers = config.num_workers

    ds_kwargs = dict(
        neuron_labels=neuron_labels,
        total_n_neurons=total_n_neurons,
        past_embed=config.past_embed,
        future_embed=config.future_embed,
        n_folds=config.cv_folds,
        fold_idx=config.fold_idx,
        equalize_hdir=config.equalize_hdir,
        n_hdir_bins=config.n_hdir_bins,
    )
    train_ds = JLDataset(files, train=True, **ds_kwargs)
    test_ds = JLDataset(files, train=False, **ds_kwargs)

    loader_kwargs = dict(batch_size=config.batch_size, collate_fn=collate_fn)
    train_loader = DataLoader(train_ds, num_workers=num_workers, **loader_kwargs)
    test_loader = DataLoader(test_ds, num_workers=num_workers, **loader_kwargs)
    return train_loader, test_loader
