import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader, get_worker_info
device = 'cuda' if torch.cuda.is_available() else 'cpu'

import numpy as np
import math
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

    total_variance = (( targets - targets.mean(axis=0)[None,:] )**2 ).sum().item()
    remaining_variance = (( targets - predictions )**2 ).sum().item()

    return (total_variance - remaining_variance) / total_variance
    

class EncoderModel(nn.Module):
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

        self.encoder = nn.Sequential(*modules)
                

    def forward(
            self, 
            X: torch.Tensor
        ) -> torch.Tensor:

        return self.encoder(X)


class JLDataset(IterableDataset):
    def __init__(
        self,
        file: str,
        past_embed: int,
        future_embed: int,
        n_folds: int,
        fold_idx: int,
        train: bool,
        seed: int = 0,
        shuffle_buffer_size: int = 1000,
        equalize_hdir: bool = False,
        n_hdir_bins: int = 12,
        hdir_range: tuple[float, float] | None = None,   # e.g. (-pi, pi) or (0, 2*pi)
    ):
        super().__init__()
        self.file = file
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

    def _load_arrays(self):
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

        schema = pl.read_csv(self.file, n_rows=0).schema
        target_cols = [c for c in schema if c.isdigit()]

        lf = (
            pl.scan_csv(self.file)
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
            pl.all_horizontal(shifted_frame_checks) if shifted_frame_checks else pl.lit(True)
        )

        contextual_behave_cols = list(BEHAV_COLS)
        contextual_behave_cols += [
            f"{c}-{k}" for k in range(1, self.past_embed + 1) for c in BEHAV_COLS
        ]
        contextual_behave_cols += [
            f"{c}+{k}" for k in range(1, self.future_embed + 1) for c in BEHAV_COLS
        ]

        # First collect only the valid rows, in order
        df = (
            lf.with_columns(shifted_feature_exprs + [valid_window.alias("_valid_window")])
            .filter(pl.col("_valid_window"))
            .select(contextual_behave_cols + target_cols)
            .collect()
        )

        # Contiguous k-fold split
        n_rows = df.height
        row_idx = np.arange(n_rows)

        # Fold boundaries like numpy.array_split
        block_starts = [(i * n_rows) // self.n_folds for i in range(self.n_folds)]
        block_ends   = [((i + 1) * n_rows) // self.n_folds for i in range(self.n_folds)]

        test_start = block_starts[self.fold_idx]
        test_end = block_ends[self.fold_idx]

        if self.train:
            keep_mask = (row_idx < test_start) | (row_idx >= test_end)
        else:
            keep_mask = (row_idx >= test_start) & (row_idx < test_end)

        df = df.filter(pl.Series(keep_mask))

        inputs = df.select(contextual_behave_cols).to_numpy().astype(np.float32, copy=False)
        targets = df.select(target_cols).to_numpy().astype(np.float32, copy=False)

        return inputs, targets, contextual_behave_cols

    def _balanced_order_from_hdir(
        self,
        hdir_values: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """
        Return an index order such that consecutive samples are approximately
        balanced across hdir bins.
        """
        n = len(hdir_values)
        if n == 0:
            return np.empty(0, dtype=np.int64)

        x = np.asarray(hdir_values, dtype=np.float64)

        # Choose bin edges
        if self.hdir_range is not None:
            lo, hi = self.hdir_range
        else:
            lo = float(np.nanmin(x))
            hi = float(np.nanmax(x))
            if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
                return rng.permutation(n)

        edges = np.linspace(lo, hi, self.n_hdir_bins + 1)

        # Bin indices into [0, n_hdir_bins - 1]
        bin_ids = np.clip(np.digitize(x, edges[1:-1], right=False), 0, self.n_hdir_bins - 1)

        bins: List[np.ndarray] = []
        for b in range(self.n_hdir_bins):
            idx = np.flatnonzero(bin_ids == b)
            if len(idx) > 0:
                idx = rng.permutation(idx)
                bins.append(idx)
            else:
                bins.append(np.empty(0, dtype=np.int64))

        # Round-robin interleaving across non-empty bins
        cursors = np.zeros(self.n_hdir_bins, dtype=np.int64)
        order = []

        while True:
            any_left = False
            # randomize bin traversal each cycle to avoid deterministic striping
            for b in rng.permutation(self.n_hdir_bins):
                c = cursors[b]
                if c < len(bins[b]):
                    order.append(bins[b][c])
                    cursors[b] += 1
                    any_left = True
            if not any_left:
                break

        return np.asarray(order, dtype=np.int64)

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        inputs, targets, feature_names = self._load_arrays()

        n = len(inputs)
        rng = np.random.default_rng(self.seed)

        if not self.equalize_hdir:
            order = rng.permutation(n)
        else:
            hdir_idx = feature_names.index("hdir")
            hdir_values = inputs[:, hdir_idx]
            order = self._balanced_order_from_hdir(hdir_values, rng)

        for i in order:
            yield {
                "inputs": torch.from_numpy(inputs[i]),
                "targets": torch.from_numpy(targets[i]),
                "source_file": self.file,
            }


def collate_fn(batch):
    inputs = torch.stack([b["inputs"] for b in batch], dim=0)
    targets = torch.stack([b["targets"] for b in batch], dim=0)
    return {
        "inputs": inputs,
        "targets": targets,
        "source_file": [b["source_file"] for b in batch],
    }

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

                preds = model(inputs)

                mse = torch.mean( torch.square( preds - targets ) ) 
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
            optim.step()
            optim.zero_grad()
            train_losses.append(loss.detach().cpu().item())

        model.eval()
        with torch.no_grad():
            for _, _, loss in _iter_all_batches(test_loader, train=False):
                test_losses.append(loss.detach().cpu().item())

        yield e, np.array(train_losses), np.array(test_losses)

def get_total_var_explained(test_loader, model):
    with torch.no_grad():
        inputs, targets = None, None
        for b, batch in enumerate(test_loader):
            _inputs, _targets = batch['inputs'].to(device), batch['targets'].to(device)
            if inputs is None and targets is None:
                inputs, targets = _inputs, _targets
            else:
                inputs, targets = torch.concatenate((inputs, _inputs), dim=0), torch.concatenate((targets, _targets), dim=0)

        preds = model(inputs)
        return get_variance_explained(targets, preds)

def main(
    data_dir: str,
    past_embed:int=0,
    future_embed:int=0,
    cv_folds:int=5,
    batch_size:int=512, 
    l2_lambda:float=1e-4, 
    hidden_layers:List[int]=[256,256], 
    lr:float=1e-4, 
    epochs:int=1000, 
    print_interval:int=10,
    equalize_hdir:bool=True,
    n_hdir_bins:int=30):

    assert 'data.csv' in os.listdir(data_dir)
    data_path = os.path.join(data_dir, 'data.csv')
    ckpt_dir = datetime.now().strftime("%d-%m-%y_%H:%M:%S.%f")
    os.mkdir( os.path.join(data_dir, ckpt_dir) )

    for k in range(cv_folds):

        train_ds = JLDataset( data_path, past_embed = past_embed, future_embed = future_embed, n_folds = cv_folds, fold_idx = k, train = True ,
                             equalize_hdir=equalize_hdir, n_hdir_bins=n_hdir_bins)
        test_ds  = JLDataset( data_path, past_embed = past_embed, future_embed = future_embed, n_folds = cv_folds, fold_idx = k, train = False ,
                             equalize_hdir=equalize_hdir, n_hdir_bins=n_hdir_bins)

        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            num_workers=0,
            collate_fn=collate_fn,
        )

        test_loader = DataLoader(
            test_ds,
            batch_size=batch_size,
            num_workers=0,
            collate_fn=collate_fn,
        )

        df = pl.read_csv( data_path , n_rows=0)
        n_outputs = sum(1 for col in df.schema if col.isdigit())

        model = EncoderModel(
            n_inputs = 7 * (1 + past_embed + future_embed),
            hidden_layers = hidden_layers,
            n_outputs = n_outputs,
            l2_lambda = l2_lambda
        ).to(device)

        layerwise_lr = []
        for l, layer in enumerate(model.encoder):
            output_dim = layer[0].out_features
            this_lr = {
                "params": model.encoder[l].parameters(), "lr": lr if l == len(model.encoder)-1 else lr*output_dim
            } 
            layerwise_lr.append(this_lr)


        optim = torch.optim.Adam(layerwise_lr)

        print(f'[{k}/{cv_folds}] Beginning training for {data_dir}')
        print(model)

        all_train_losses, all_test_losses = np.array([]), np.array([])
        for epoch, train_losses, test_losses in iter_epochs(model, optim, train_loader, test_loader,
                                                            print_interval=print_interval, epochs=epochs):
            all_train_losses, all_test_losses = np.concatenate((all_train_losses, train_losses)), np.concatenate((all_test_losses, test_losses))
        
            print(f'\n[{k}/{cv_folds}] Epoch complete: total variance explained = {get_total_var_explained(test_loader, model):.6f}\n')

        print(f'\n[{k}/{cv_folds}] Training complete: total variance explained = {get_total_var_explained(test_loader, model):.6f}\n')
        ckpt_path = os.path.join(data_dir, ckpt_dir, f"{k}.pt")
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optim_state_dict": optim.state_dict(),
                "config": dict(
                    past_embed=past_embed,
                    future_embed=future_embed,
                    cv_folds=cv_folds,
                    batch_size=batch_size, 
                    l2_lambda=l2_lambda, 
                    hidden_layers=hidden_layers, 
                    lr=lr, 
                    epochs=epochs, 
                    print_interval=print_interval
                ),
                "train_losses": all_train_losses,
                "test_losses": all_test_losses,
            },
            ckpt_path,
        )


if __name__ == '__main__':
    animal, session = sys.argv[1], sys.argv[2]
    data_dir = os.path.join(root, animal, session)

    config = dict(
        past_embed=0,
        future_embed=0,
        cv_folds=5,
        batch_size=512, 
        l2_lambda=1e-4, 
        hidden_layers=[64,64], 
        lr=1e-4, 
        epochs=10000, 
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


    if 'data.csv' not in os.listdir(data_dir):
        print('data.csv not found')
    else:
        main(
            data_dir, 
            **config
        )