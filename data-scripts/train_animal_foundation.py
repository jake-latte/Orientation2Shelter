import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader, get_worker_info
device = 'cuda' if torch.cuda.is_available() else 'cpu'

import numpy as np
import os
import matplotlib.pyplot as plt
import polars as pl
from pathlib import Path
from typing import Iterator, List, Dict, Any, Sequence
import wandb
import sys
from datetime import datetime

root = "/ceph/branco/Jake/training_data"

    
def get_variance_explained(targets, predictions):
    # targets, predictions : n_frames, n_neurons

    total_variance = (( targets - targets.nanmean(axis=0)[None,:] )**2 ).nansum().item()
    remaining_variance = (( targets - predictions )**2 ).nansum().item()

    return (total_variance - remaining_variance) / total_variance
    

class FoundationModel(nn.Module):
    def __init__(
            self, 
            n_inputs: int,
            hidden_layers: List[int],
            n_outputs: int,
            l2_lambda: float = 1e-4
        ):
        self.n_layers = len(hidden_layers)+1
        assert self.n_layers >= 2

        self.n_inputs = n_inputs
        self.hidden_layers = hidden_layers
        self.n_outputs = n_outputs
        self.l2_lambda = l2_lambda
        super().__init__()

        modules = []
        for i in range(self.n_layers):
            in_size = n_inputs if i==0 else hidden_layers[i-1]
            out_size = n_outputs if i==self.n_layers-1 else hidden_layers[i]

            ff = nn.Linear(in_size, out_size, bias = i==self.n_layers-1)
            ff.weight.data = torch.randn(size=(out_size, in_size)) * (1 / np.sqrt(in_size)) 
    
            nl = nn.Identity() if i==self.n_layers-1 else nn.Tanh()
            modules.append(nn.Sequential(
                ff, nl
            ))

        self.embedding = nn.Sequential(*modules)
                

    def forward(
            self, 
            X: torch.Tensor
        ) -> torch.Tensor:

        return self.embedding(X)


class JLDataset(IterableDataset):
    def __init__(
        self,
        files: List[str],
        past_embed: int,
        future_embed: int,
        n_folds: int,
        fold_idx: int,
        train: bool,
        seed: int = 0,
        shuffle_buffer_size: int = 100000,
        equalize_hdir: bool = True,
        n_hdir_bins: int = 12,
        hdir_range: tuple[float, float] | None = None,
        session_num_classes: int = 7,
    ):
        super().__init__()
        self.files = files
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
        self.session_num_classes = session_num_classes

    def _files_for_this_worker(self) -> List[str]:
        worker = get_worker_info()
        if worker is None:
            return self.files
        return list(self.files[worker.id :: worker.num_workers])

    def _balanced_order_from_hdir(
        self,
        hdir_values: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        n = len(hdir_values)
        if n == 0:
            return np.empty(0, dtype=np.int64)

        x = np.asarray(hdir_values, dtype=np.float64)

        if self.hdir_range is not None:
            lo, hi = self.hdir_range
        else:
            lo = float(np.nanmin(x))
            hi = float(np.nanmax(x))
            if (not np.isfinite(lo)) or (not np.isfinite(hi)) or (lo == hi):
                return rng.permutation(n)

        edges = np.linspace(lo, hi, self.n_hdir_bins + 1)
        bin_ids = np.clip(
            np.digitize(x, edges[1:-1], right=False),
            0,
            self.n_hdir_bins - 1,
        )

        bins = []
        for b in range(self.n_hdir_bins):
            idx = np.flatnonzero(bin_ids == b)
            if len(idx) > 0:
                idx = rng.permutation(idx)
            bins.append(idx)

        cursors = np.zeros(self.n_hdir_bins, dtype=np.int64)
        order = []

        while True:
            any_left = False
            for b in rng.permutation(self.n_hdir_bins):
                c = cursors[b]
                if c < len(bins[b]):
                    order.append(bins[b][c])
                    cursors[b] += 1
                    any_left = True
            if not any_left:
                break

        return np.asarray(order, dtype=np.int64)

    def _stream_arrays(self, files):
        BEHAV_COLS = [
            "hdir",
            "hsa",
            "angular_speed",
            "translational_speed",
            "x",
            "y",
            "dist",
        ]
        ID_COLS = ["animal_idx", "exp_idx"]
        FRAME_COL = "frame"

        NORMS = [
            np.pi,
            np.pi,
            3*np.pi,
            1024,
            1024,
            1024,
            1024,
        ]

        worker = get_worker_info()
        worker_id = 0 if worker is None else worker.id
        rng = np.random.default_rng(self.seed + worker_id)

        files = list(files)
        rng.shuffle(files)

        for file in files:
            schema = pl.read_csv(file, n_rows=0).schema
            target_cols = [c for c in schema if c.isdigit()]
            target_cols.sort()

            lf = (
                pl.scan_csv(file)
                .sort(ID_COLS + [FRAME_COL])
                .with_columns([
                    (pl.col(c) / norm).alias(c)
                    for c, norm in zip(BEHAV_COLS, NORMS)
                ])
            )

            shifted_feature_exprs = []
            shifted_frame_checks = []

            for k in range(1, self.past_embed + 1):
                shifted_feature_exprs.extend(
                    pl.col(c).shift(k).over(ID_COLS).alias(f"{c}-{k}")
                    for c in BEHAV_COLS
                )
                shifted_frame_checks.append(
                    pl.col(FRAME_COL).shift(k).over(ID_COLS).eq(pl.col(FRAME_COL) - k)
                )

            for k in range(1, self.future_embed + 1):
                shifted_feature_exprs.extend(
                    pl.col(c).shift(-k).over(ID_COLS).alias(f"{c}+{k}")
                    for c in BEHAV_COLS
                )
                shifted_frame_checks.append(
                    pl.col(FRAME_COL).shift(-k).over(ID_COLS).eq(pl.col(FRAME_COL) + k)
                )

            valid_window = (
                pl.all_horizontal(shifted_frame_checks)
                if shifted_frame_checks
                else pl.lit(True)
            )

            contextual_behave_cols = list(BEHAV_COLS)
            contextual_behave_cols += [
                f"{c}-{k}" for k in range(1, self.past_embed + 1) for c in BEHAV_COLS
            ]
            contextual_behave_cols += [
                f"{c}+{k}" for k in range(1, self.future_embed + 1) for c in BEHAV_COLS
            ]

            df = (
                lf.with_columns(shifted_feature_exprs + [valid_window.alias("_valid_window")])
                  .filter(pl.col("_valid_window"))
                  .with_row_index("row_idx")
                  .collect()
            )

            n_rows = df.height
            test_start = (self.fold_idx * n_rows) // self.n_folds
            test_end = ((self.fold_idx + 1) * n_rows) // self.n_folds

            if self.train:
                df = df.filter(
                    (pl.col("row_idx") < test_start) | (pl.col("row_idx") >= test_end)
                )
            else:
                df = df.filter(
                    (pl.col("row_idx") >= test_start) & (pl.col("row_idx") < test_end)
                )

            if df.height == 0:
                continue

            df = df.select(contextual_behave_cols + target_cols + ["exp_idx"])

            input_array = df.select(contextual_behave_cols).to_numpy().astype(np.float32, copy=False)
            target_array = df.select(target_cols).to_numpy().astype(np.float32, copy=False)
            exp_idx_array = df["exp_idx"].to_numpy()

            if self.equalize_hdir:
                hdir_idx = contextual_behave_cols.index("hdir")
                order = self._balanced_order_from_hdir(input_array[:, hdir_idx], rng)
            else:
                order = rng.permutation(len(input_array))

            for i in order:
                i = int(i)

                inputs = torch.from_numpy(input_array[i])

                session_label = torch.tensor(exp_idx_array[i], dtype=torch.long)
                session_oh = F.one_hot(
                    session_label,
                    num_classes=self.session_num_classes,
                ).to(inputs.dtype)

                inputs = torch.concatenate((inputs, session_oh), dim=0)

                targets = torch.from_numpy(target_array[i])

                yield {
                    "inputs": inputs,
                    "targets": targets,
                    "source_file": str(file),
                }

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        files = self._files_for_this_worker()
        stream = self._stream_arrays(files)

        worker = get_worker_info()
        worker_id = 0 if worker is None else worker.id
        rng = np.random.default_rng(self.seed + 10_000 + worker_id)

        buffer = []

        try:
            for _ in range(self.shuffle_buffer_size):
                buffer.append(next(stream))
        except StopIteration:
            pass

        while buffer:
            idx = rng.integers(len(buffer))
            sample = buffer[idx]

            try:
                buffer[idx] = next(stream)
            except StopIteration:
                buffer.pop(idx)

            yield sample


def collate_fn(batch):
    def _pad_stack_nan(xs: Sequence[torch.Tensor]) -> torch.Tensor:
        device = xs[0].device
        dtype = xs[0].dtype

        if not (dtype.is_floating_point or dtype.is_complex):
            raise TypeError("NaN padding requires a floating-point or complex dtype")

        lengths = [x.numel() for x in xs]
        max_len = max(lengths)

        out = torch.full((len(xs), max_len), torch.nan, dtype=dtype, device=device)

        for i, x in enumerate(xs):
            n = x.numel()
            out[i, :n] = x

        return out

    inputs = torch.stack([b["inputs"] for b in batch], dim=0)
    targets = _pad_stack_nan([b["targets"] for b in batch])

    return {
        "inputs": inputs,
        "targets": targets,
        "source_file": [b["source_file"] for b in batch],
    }


def preds_from_embeds(embeds, targets, n_outputs, l2_lambda):
    # embeds: (T, H)
    # targets:(T, N)

    _embeds = embeds.clone()
    _targets = targets.clone()

    mask = (~torch.isnan(_targets)).to(targets.dtype)                           # (T, N)
    Y0 = torch.nan_to_num(_targets, nan=0.0).to(targets.dtype)                  # (T, N)

    XX = torch.einsum("th,tH,tn->nhH", _embeds, _embeds, mask)                   # (N, H, H)
    I = torch.eye(n_outputs, device=device, dtype=targets.dtype)               # (H, H)

    A = XX + l2_lambda * I.unsqueeze(0)                                        # (N, H, H)

    XY = torch.einsum("th,tn,tn->nh", _embeds, mask, Y0)                        # (N, H)

    weights = torch.linalg.solve(
        A, XY.unsqueeze(-1)                                                    # (N, H)
    ).squeeze(-1)

    preds = embeds @ weights.T                                                 # (T, N)

    return preds


def iter_epochs(model, optim, train_loader, test_loader, print_interval:int=100, epochs:int=10000):

    for e in range(epochs):
        optim.zero_grad()
        train_losses, test_losses = [], []

        def _iter_all_batches(loader, train:bool):

            avg_mse:float = 0
            avg_l2:float = 0
            avg_loss:float = 0
            
            for b, batch in enumerate(loader):
                inputs, targets = batch['inputs'].to(device), batch['targets'].to(device)

                embeds = model(inputs)                                                          # (T, H)
                preds = preds_from_embeds(embeds, targets, model.n_outputs, model.l2_lambda)    # (T, N)
                mask = (~torch.isnan(targets))                                                  # (T, N)

                mse = torch.mean( torch.square( preds[mask] - targets[mask] ) ) 
                l2 = torch.mean( torch.square( torch.concatenate([p.ravel() for p in model.parameters()]) ))

                loss = mse + model.l2_lambda*l2

                avg_mse += mse
                avg_l2 += l2
                avg_loss += loss

                if (b+1)%print_interval == 0:
                    print(f'[{e:05d}] ({"T" if train else "V"}) {b+1:05d}: loss={avg_loss/print_interval:.4f} | mse={avg_mse/print_interval:.4f} | l2={avg_l2/print_interval:.4f}')

                    avg_mse, avg_l2, avg_loss = 0, 0, 0
                
                yield mse, l2, loss

        model.train()
        for _, _, loss in _iter_all_batches(train_loader, train=True):
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optim.step()
            optim.zero_grad()
            train_losses.append(loss.detach().cpu().item())

        print()
        model.eval()
        with torch.no_grad():
            for _, _, loss in _iter_all_batches(test_loader, train=False):
                test_losses.append(loss.detach().cpu().item())
        print()

        yield e, np.array(train_losses), np.array(test_losses)



def get_total_var_explained(test_loader, model):
    total_variance, remaining_variance = 0.0, 0.0

    with torch.no_grad():

        for b, batch in enumerate(test_loader):
            inputs, targets = batch['inputs'].to(device), batch['targets'].to(device)
            embeds = model(inputs)
            preds = preds_from_embeds(embeds, targets, n_outputs=model.n_outputs, l2_lambda=model.l2_lambda)

            total_variance += (( targets - targets.nanmean(axis=0)[None,:] )**2 ).nanmean().item()
            remaining_variance += (( targets - preds )**2 ).nanmean().item()

    return (total_variance - remaining_variance) / total_variance

def main(
    animal_dir: str,
    num_workers:int=4,
    past_embed:int=0,
    future_embed:int=0,
    embed_dim:int=256,
    cv_folds:int=5,
    batch_size:int=512, 
    l2_lambda:float=1e-4, 
    hidden_layers:List[int]=[256,256], 
    lr:float=1e-4, 
    epochs:int=1000, 
    print_interval:int=10,
    equalize_hdir:bool=True,
    n_hdir_bins:int=30):

    files = []
    for session in os.listdir(animal_dir):
        if 'data.csv' in list(os.listdir(os.path.join(animal_dir, session))):
            files.append(
                os.path.join(animal_dir, session, 'data.csv')
            )

    ckpt_dir = datetime.now().strftime("%d-%m-%y_%H:%M:%S.%f")
    os.mkdir( os.path.join(animal_dir, ckpt_dir) )

    for k in range(cv_folds):

        train_ds = JLDataset( files, past_embed = past_embed, future_embed = future_embed, n_folds = cv_folds, fold_idx = k, train = True ,
                             equalize_hdir=equalize_hdir, n_hdir_bins=n_hdir_bins)
        test_ds  = JLDataset( files, past_embed = past_embed, future_embed = future_embed, n_folds = cv_folds, fold_idx = k, train = False ,
                             equalize_hdir=equalize_hdir, n_hdir_bins=n_hdir_bins)

        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=collate_fn,
        )

        test_loader = DataLoader(
            test_ds,
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=collate_fn,
        )

        model = FoundationModel(
            n_inputs = 7 * (2 + past_embed + future_embed),
            hidden_layers = hidden_layers,
            n_outputs = embed_dim,
            l2_lambda = l2_lambda
        ).to(device)

        layerwise_lr = []
        for l, layer in enumerate(model.embedding):
            output_dim = layer[0].out_features
            this_lr = {
                "params": model.embedding[l].parameters(), "lr": lr*output_dim
            } 
            layerwise_lr.append(this_lr)


        optim = torch.optim.Adam(layerwise_lr)

        print(f'[{k}/{cv_folds}] Beginning training for {animal_dir}')
        print(model)

        all_train_losses, all_test_losses = np.array([]), np.array([])
        for epoch, train_losses, test_losses in iter_epochs(model, optim, train_loader, test_loader,
                                                            print_interval=print_interval, epochs=epochs):
            all_train_losses, all_test_losses = np.concatenate((all_train_losses, train_losses)), np.concatenate((all_test_losses, test_losses))

            print(f'\n[{k}/{cv_folds}] Epoch complete: total variance explained = {get_total_var_explained(test_loader, model):.6f}\n')


            ckpt_path = os.path.join(animal_dir, ckpt_dir, f"{k}.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optim_state_dict": optim.state_dict(),
                    "config": dict(
                        num_workers=num_workers,
                        past_embed=past_embed,
                        future_embed=future_embed,
                        embed_dim=embed_dim,
                        cv_folds=cv_folds,
                        batch_size=batch_size, 
                        l2_lambda=l2_lambda, 
                        hidden_layers=hidden_layers, 
                        lr=lr, 
                        epochs=epochs, 
                        print_interval=print_interval,
                        equalize_hdir=equalize_hdir,
                        n_hdir_bins=n_hdir_bins
                    ),
                    "train_losses": all_train_losses,
                    "test_losses": all_test_losses,
                },
                ckpt_path,
            )
        
        print(f'\n[{k}/{cv_folds}] Training complete: total variance explained = {get_total_var_explained(test_loader, model):.6f}\n')



if __name__ == '__main__':
    animal = sys.argv[1]
    animal_dir = os.path.join(root, animal)

    config = dict(
        num_workers=4,
        past_embed=0,
        future_embed=0,
        embed_dim=64,
        cv_folds=5,
        batch_size=512, 
        l2_lambda=1e-4, 
        hidden_layers=[64,64], 
        lr=1e-4, 
        epochs=1000, 
        print_interval=10,
        equalize_hdir=True,
        n_hdir_bins=30
    )

    for i, arg in enumerate(sys.argv):
        if arg.startswith('-'):
            param = arg[1:]

            try:
                if isinstance(config[param], list):
                    inner_dtype = type( config[param][0] )
                    config[param] = [ inner_dtype( e ) for e in sys.argv[i+1].strip('[]').split(',')]
                else:
                    dtype = type(config[param])
                    value = dtype( sys.argv[i+1] )
                    config[param] = value
            except Exception as e:
                print(f'Setting of {param} config value failed:')
                print(e)


    main(
        animal_dir, 
        **config
    )