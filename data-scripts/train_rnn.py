import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader, get_worker_info
from torch.func import functional_call, jvp
from collections import OrderedDict
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

    
    

class RNNModel(nn.Module):
    def __init__(
            self, 
            n_inputs: int,
            n_hidden: int,
            # n_outputs: int,
            dt: float = 0.1,
            tau: float = 1.0,
            l2_lambda: float = 1e-4,
            activation_func: nn.Module = nn.ReLU,
            dtype: torch.dtype = torch.float32,
            device: torch.device = torch.device('cpu')
        ):
        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        # self.n_outputs = n_outputs
        self.dt = dt
        self.tau = tau
        self.l2_lambda = l2_lambda
        self.activation_func = activation_func
        self.dtype = dtype
        self.device = device
        super().__init__()

        self.x_0 = nn.Parameter(torch.zeros((n_hidden,), dtype=dtype, device=device), requires_grad=True)
        self.input = nn.Linear(n_inputs, n_hidden, bias=True, dtype=dtype, device=device)
        self.hidden = nn.Sequential(
            self.activation_func(), nn.Linear(n_hidden, n_hidden, bias=False, dtype=dtype, device=device)
        )
        # self.output = nn.Sequential(
        #     self.activation_func(), nn.Linear(n_hidden, n_outputs, bias=False, dtype=dtype, device=device)
        # )
                

    def forward(
            self, 
            u: torch.Tensor
        ) -> torch.Tensor:
    
        # u : B, T, D_in
        # returns 
        # z : B, T, D_out

        B, T, D_in = u.shape
        alpha = self.dt / self.tau

        x_list = [self.x_0.reshape((1,-1)).repeat((B,1))]
        z_list = []
        for t in range(T):
            u_prev = u[:,t]                                # B, D_in
            x_prev = x_list[-1]
            x = (1 - alpha)*x_prev + alpha*( self.hidden( x_prev ) + self.input( u_prev ) )
            x_list.append(x)
            # z_list.append( self.output(x_t) )

        # return torch.stack(z_list, dim=1)               # B, T, D_out
        return torch.stack(x_list[1:], dim=1)



class JLDataset(IterableDataset):
    def __init__(
        self,
        files: List[str],
        n_timesteps: int,
        n_folds: int,
        fold_idx: int,
        train: bool,
        seed: int = 0,
        shuffle_buffer_size: int = 100000,
        equalize_hdir: bool = True,
        n_hdir_bins: int = 12,
        hdir_range: tuple[float, float] | None = None,
        animal_num_classes: int = 5,
        session_num_classes: int = 7,
    ):
        super().__init__()
        self.files = files
        self.n_timesteps = n_timesteps
        self.n_folds = n_folds
        self.fold_idx = fold_idx
        self.train = train
        self.seed = seed
        self.shuffle_buffer_size = shuffle_buffer_size

        self.equalize_hdir = equalize_hdir
        self.n_hdir_bins = n_hdir_bins
        self.hdir_range = hdir_range
        self.animal_num_classes = animal_num_classes
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

        worker = get_worker_info()
        worker_id = 0 if worker is None else worker.id
        rng = np.random.default_rng(self.seed + worker_id)

        files = list(files)
        rng.shuffle(files)

        n_session_classes = self.animal_num_classes * self.session_num_classes

        for file in files:
            schema = pl.read_csv(file, n_rows=0).schema
            target_cols = sorted([c for c in schema if c.isdigit()])

            lf = pl.scan_csv(file).sort(ID_COLS + [FRAME_COL])

            # Build shifted columns for timesteps 1..n_timesteps-1.
            shifted_exprs = []
            shifted_frame_checks = []

            for k in range(1, self.n_timesteps):
                shifted_exprs.extend(
                    pl.col(c).shift(-k).over(ID_COLS).alias(f"{c}+{k}")
                    for c in BEHAV_COLS + target_cols
                )
                shifted_frame_checks.append(
                    pl.col(FRAME_COL).shift(-k).over(ID_COLS) == (pl.col(FRAME_COL) + k)
                )

            valid_window = (
                pl.all_horizontal(shifted_frame_checks)
                if shifted_frame_checks
                else pl.lit(True)
            )

            # Time-major column layout:
            # [t0 cols..., t1 cols..., ..., t_{T-1} cols...]
            input_window_cols = []
            target_window_cols = []

            for k in range(self.n_timesteps):
                suffix = "" if k == 0 else f"+{k}"
                input_window_cols.extend([f"{c}{suffix}" for c in BEHAV_COLS])
                target_window_cols.extend([f"{c}{suffix}" for c in target_cols])

            df = (
                lf.with_columns(shifted_exprs + [valid_window.alias("_valid_window")])
                .filter(pl.col("_valid_window"))
                .with_row_index("row_idx")
                .collect()
            )

            n_rows = df.height
            if n_rows == 0:
                continue

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

            # Keep only what we need.
            df = df.select(input_window_cols + target_window_cols + ID_COLS)

            input_array = df.select(input_window_cols).to_numpy().astype(np.float32, copy=False)
            target_array = df.select(target_window_cols).to_numpy().astype(np.float32, copy=False)
            animal_idx_array = df["animal_idx"].to_numpy()
            exp_idx_array = df["exp_idx"].to_numpy()

            # Equalize by hdir at the first timestep.
            if self.equalize_hdir:
                hdir_idx = input_window_cols.index("hdir")
                order = self._balanced_order_from_hdir(input_array[:, hdir_idx], rng)
            else:
                order = rng.permutation(df.height)

            n_behav = len(BEHAV_COLS)
            n_neurons = len(target_cols)

            for i in order:
                i = int(i)

                inputs = torch.from_numpy(
                    input_array[i].reshape(self.n_timesteps, n_behav)
                )

                targets = torch.from_numpy(
                    target_array[i].reshape(self.n_timesteps, n_neurons)
                )

                session_label = torch.tensor(
                    animal_idx_array[i] * self.session_num_classes + exp_idx_array[i],
                    dtype=torch.long,
                )

                session_oh = F.one_hot(
                    session_label,
                    num_classes=n_session_classes,
                ).to(inputs.dtype)

                # Repeat metadata across time so inputs remain 2D: [T, features]
                session_oh = session_oh.unsqueeze(0).expand(self.n_timesteps, -1)
                inputs = torch.cat((inputs, session_oh), dim=1)

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
    def _pad_stack_nan_lastdim(xs: Sequence[torch.Tensor]) -> torch.Tensor:
        if len(xs) == 0:
            raise ValueError("Empty batch")

        first = xs[0]
        device = first.device
        dtype = first.dtype

        if not (dtype.is_floating_point or dtype.is_complex):
            raise TypeError("NaN padding requires a floating-point or complex dtype")

        # All samples should have the same time dimension
        n_timesteps = first.shape[0]
        for x in xs:
            if x.ndim != 2:
                raise ValueError(f"Expected 2D tensors, got shape {tuple(x.shape)}")
            if x.shape[0] != n_timesteps:
                raise ValueError(
                    f"All tensors must have the same first dimension; "
                    f"got {x.shape[0]} and {n_timesteps}"
                )

        max_last_dim = max(x.shape[1] for x in xs)

        out = torch.full(
            (len(xs), n_timesteps, max_last_dim),
            torch.nan,
            dtype=dtype,
            device=device,
        )

        for i, x in enumerate(xs):
            out[i, :, : x.shape[1]] = x

        return out

    inputs = torch.stack([b["inputs"] for b in batch], dim=0)
    targets = _pad_stack_nan_lastdim([b["targets"] for b in batch])

    return {
        "inputs": inputs,      # [B, n_timesteps, n_input_features]
        "targets": targets,    # [B, n_timesteps, max_n_neurons]
        "source_file": [b["source_file"] for b in batch],
    }


def preds_from_embeds(embeds, targets, l2_lambda):
    # embeds: (B, T, H)
    # targets:(B, T, N)

    _, _, n_outputs = embeds.shape

    mask = (~torch.isnan(targets)).to(targets.dtype)                           # (B, T, N)
    Y0 = torch.nan_to_num(targets, nan=0.0).to(targets.dtype)                  # (B, T, N)

    XX = torch.einsum("bth,btH,btn->bnhH", embeds, embeds, mask)               # (B, N, H, H)
    I = torch.eye(n_outputs, device=device, dtype=targets.dtype)               # (H, H)

    A = XX + l2_lambda * I.unsqueeze(0).unsqueeze(1)                           # (B, N, H, H)

    XY = torch.einsum("bth,btn,btn->bnh", embeds, mask, Y0)                    # (B, N, H)

    weights = torch.linalg.solve(
        A, XY.unsqueeze(-1)                                                    # (B, N, H)
    ).squeeze(-1)

    preds = torch.einsum('bth, bnh -> btn', embeds, weights)                   # (B, T, N)

    return preds


def iter_epochs(model, optim, train_loader, test_loader, print_interval:int=100, epochs:int=10000, clip_grad_norm:float=1.0):

    for e in range(epochs):
        optim.zero_grad()
        train_losses, test_losses = [], []

        def _iter_all_batches(loader, train:bool):

            avg_mse:float = 0
            avg_l2:float = 0
            avg_loss:float = 0
            
            for b, batch in enumerate(loader):
                inputs, targets = batch['inputs'].to(device), batch['targets'].to(device)

                embeds = model(inputs)                                                          # (B, T, H)
                preds = preds_from_embeds(embeds, targets, model.n_outputs, model.l2_lambda)    # (B, T, N)
                mask = (~torch.isnan(targets))                                                  # (B, T, N)

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

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad_norm)

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

            total_variance += (( targets - targets.nanmean(axis=0)[None,:] )**2 ).nansum().item()
            remaining_variance += (( targets - preds )**2 ).nansum().item()

    return (total_variance - remaining_variance) / total_variance






def main(**kwargs):
    num_workers:int= kwargs.get('num_workers', 4)
    n_timesteps:int= kwargs.get('n_timesteps', 0)
    n_hidden:int= kwargs.get('n_hidden', 256)
    cv_folds:int= kwargs.get('cv_folds', 5)
    batch_size:int= kwargs.get('batch_size', 512)
    l2_lambda:float= kwargs.get('l2_lambda', 1e-4)
    hidden_layers:List[int]= kwargs.get('hidden_layers', [64,64])
    lr:float= kwargs.get('lr', 1e-4)
    epochs:int= kwargs.get('epochs', 1000)
    print_interval:int= kwargs.get('print_interval', 10)
    equalize_hdir:bool= kwargs.get('equalize_hdir', True)
    n_hdir_bins:int= kwargs.get('n_hdir_bins', 30)
    use_wandb:bool = kwargs.get('use_wandb', False)
    clip_grad_norm:float = kwargs.get('clip_grad_norm', 1.0)

    files = []
    for animal in os.listdir(root):
        animal_dir = os.path.join(root, animal)
        if not os.path.isdir(animal_dir):
            continue

        for session in os.listdir(animal_dir):
            session_dir = os.path.join(animal_dir, session)
            if not os.path.isdir(session_dir):
                continue

            if 'data.csv' in list(os.listdir(session_dir)):
                files.append(
                    os.path.join(animal_dir, session, 'data.csv')
                )

    ckpt_dir = datetime.now().strftime("%d-%m-%y_%H:%M:%S.%f")
    os.mkdir( os.path.join(root, ckpt_dir) )

    run = None
    if use_wandb:
        run = wandb.init(
            project="rnn",
            name=ckpt_dir,
            config=kwargs
        )

    for k in range(cv_folds):

        train_ds = JLDataset( files, n_timesteps=n_timesteps, n_folds = cv_folds, fold_idx = k, train = True ,
                             equalize_hdir=equalize_hdir, n_hdir_bins=n_hdir_bins)
        test_ds  = JLDataset( files, n_timesteps=n_timesteps, n_folds = cv_folds, fold_idx = k, train = False ,
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

        model = RNNModel(
            n_inputs = 7 + 35,
            n_hidden = n_hidden,
            l2_lambda = l2_lambda
        ).to(device)


        optim = torch.optim.Adam(model.parameters(), lr=lr)

        print(model)
        print(f'[{k}/{cv_folds}] Beginning training: total variance explained = {get_total_var_explained(test_loader, model):.6f}\n')

        if use_wandb:
            run.watch(model, log="all", log_freq=200)

        all_train_losses, all_test_losses = np.array([]), np.array([])
        for epoch, train_losses, test_losses in iter_epochs(model, optim, train_loader, test_loader,
                                                            print_interval=print_interval, epochs=epochs, clip_grad_norm=clip_grad_norm):
            all_train_losses, all_test_losses = np.concatenate((all_train_losses, train_losses)), np.concatenate((all_test_losses, test_losses))

            var_explained = get_total_var_explained(test_loader, model)
            print(f'\n[{k+1}/{cv_folds}] Epoch complete: total variance explained = {var_explained:.6f}\n')


            ckpt_path = os.path.join(root, ckpt_dir, f"{k}.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optim_state_dict": optim.state_dict(),
                    "config": kwargs,
                    "train_losses": all_train_losses,
                    "test_losses": all_test_losses,
                },
                ckpt_path,
            )

            if use_wandb:
                run.log({
                    "train_avg_loss": train_losses.mean(),
                    "test_avg_loss": test_losses.mean(),
                    "var_explained": var_explained,
                    "epoch": epoch,
                    "fold": k
                })
        
        print(f'\n[{k+1}/{cv_folds}] Training complete: total variance explained = {get_total_var_explained(test_loader, model):.6f}\n')

    if use_wandb:
        run.finish()


if __name__ == '__main__':

    config = dict(
        num_workers=0,
        n_timesteps=512,
        n_hidden=64,
        cv_folds=5,
        batch_size=32, 
        l2_lambda=1e-4,
        lr=1e-4, 
        epochs=1000, 
        print_interval=10,
        equalize_hdir=True,
        n_hdir_bins=30,
        use_wandb=False,
        clip_grad_norm=1.0
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
        **config
    )