"""
Unified training script for neural-prediction MLPs.

Typical use
-----------
# Train fold 2 of a cross-animal narrow model:
python data-scripts/train.py --scope all --output_mode narrow --fold_idx 2

# Train a per-animal wide model, saving checkpoints to a specific directory:
python data-scripts/train.py \\
    --scope animal --animal 0 --output_mode wide \\
    --fold_idx 0 --run_dir /ceph/branco/Jake/training_data/runs/my_run

From a notebook
---------------
from data-scripts.config import ModelConfig
from data-scripts.train import main

cfg = ModelConfig(scope='session', animal='0', session='1', output_mode='narrow',
                  epochs=500, fold_idx=0)
result = main(cfg, run_dir='/tmp/my_run')
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import OrderedDict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Allow running as a script from repo root or from within data-scripts/
_here = os.path.dirname(os.path.abspath(__file__))
if _here not in sys.path:
    sys.path.insert(0, _here)

from config import ModelConfig, add_config_args, config_from_args  # noqa: E402
from dataset import JLDataset, collate_fn, build_loaders            # noqa: E402
from model import FoundationModel, preds_from_embeds, get_lazy_metric  # noqa: E402

device = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Data discovery
# ---------------------------------------------------------------------------

def build_files(
    config: ModelConfig,
) -> Tuple[List[str], List[Tuple[int, int]], int]:
    """
    Discover data.csv files according to scope and return
    (files, neuron_labels, total_n_neurons).
    """
    root = config.data_root
    files: List[str] = []

    if config.scope == "session":
        assert config.animal and config.session, \
            "--animal and --session required for scope=session"
        p = os.path.join(root, config.animal, config.session, "data.csv")
        assert os.path.isfile(p), f"Not found: {p}"
        files = [p]

    elif config.scope == "animal":
        assert config.animal, "--animal required for scope=animal"
        animal_dir = os.path.join(root, config.animal)
        for s in sorted(os.listdir(animal_dir)):
            p = os.path.join(animal_dir, s, "data.csv")
            if os.path.isfile(p):
                files.append(p)

    else:  # "all"
        for a in sorted(os.listdir(root)):
            adir = os.path.join(root, a)
            if not os.path.isdir(adir):
                continue
            for s in sorted(os.listdir(adir)):
                p = os.path.join(adir, s, "data.csv")
                if os.path.isfile(p):
                    files.append(p)

    assert files, f"No data.csv found for scope={config.scope!r}"

    neuron_labels: List[Tuple[int, int]] = []
    total = 0
    for f in files:
        schema = pl.read_csv(f, n_rows=0).schema
        n = sum(1 for c in schema if c.isdigit())
        neuron_labels.append((total, total + n))
        total += n

    return files, neuron_labels, total


# ---------------------------------------------------------------------------
# Model + optimiser construction
# ---------------------------------------------------------------------------

def build_model(
    config: ModelConfig,
    n_files: int,
    total_n_neurons: int,
) -> FoundationModel:
    n_behav = 7 * (1 + config.past_embed + config.future_embed)
    n_inputs = n_behav + n_files  # behaviour window + session one-hot

    n_outputs = (
        config.embed_dim if config.output_mode == "narrow" else total_n_neurons
    )

    return FoundationModel(
        n_inputs=n_inputs,
        hidden_layers=config.hidden_layers,
        n_outputs=n_outputs,
        output_mode=config.output_mode,
        total_n_neurons=total_n_neurons,
        l2_lambda=config.l2_lambda,
    ).to(device)


def build_optimizer(model: FoundationModel, config: ModelConfig) -> torch.optim.Adam:
    # µP-style: learning rate scaled by each layer's output dimension.
    param_groups = [
        {"params": layer.parameters(), "lr": config.lr * layer[0].out_features}
        for layer in model.embedding
    ]
    return torch.optim.Adam(param_groups)


# ---------------------------------------------------------------------------
# Forward pass (shared between train and eval)
# ---------------------------------------------------------------------------

def _forward(
    model: FoundationModel,
    inputs: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    if model.output_mode == "narrow":
        embeds = model(inputs)
        return preds_from_embeds(embeds, targets, model.n_outputs, model.l2_lambda)
    return model(inputs)


# ---------------------------------------------------------------------------
# Single-epoch loop
# ---------------------------------------------------------------------------

def run_epoch(
    model: FoundationModel,
    loader: DataLoader,
    config: ModelConfig,
    train: bool,
    optim: Optional[torch.optim.Optimizer] = None,
    prefix: str = "",
    fixed_W: Optional[torch.Tensor] = None,
) -> Dict[str, Any]:
    """
    One full pass through loader.  Computes loss and variance-explained
    in a single iteration — no second pass needed, which avoids the
    DataLoader multiprocessing freeze that occurred when iterating
    the test loader twice per epoch.

    For narrow mode training: accumulates per-batch ridge regression weights
    and returns their neuron-presence-weighted average as "W_avg" in the
    result dict.  For narrow mode evaluation: uses fixed_W (if provided)
    as a fixed readout matrix instead of fitting new regression weights.
    """
    model.train() if train else model.eval()

    total_mse = 0.0
    total_l2 = 0.0
    total_variance = 0.0
    remaining_variance = 0.0
    n_batches = 0

    # Narrow-mode training: accumulate readout weights across batches.
    # W_sum[i]   = sum of W_batch[i] over batches where neuron i was present
    #              (W_batch[i] == 0 for absent neurons, so sum is uncontaminated)
    # W_count[i] = number of batches where neuron i was present
    W_sum: Optional[torch.Tensor] = None
    W_count: Optional[torch.Tensor] = None

    for b, batch in enumerate(loader):
        inputs = batch["inputs"].to(device)
        targets = batch["targets"].to(device)

        if train:
            if model.output_mode == "narrow":
                embeds = model(inputs)
                preds, W_batch = preds_from_embeds(
                    embeds, targets, model.n_outputs, model.l2_lambda,
                    return_weights=True,
                )
                with torch.no_grad():
                    if W_sum is None:
                        W_sum = torch.zeros_like(W_batch)
                        W_count = torch.zeros(
                            W_batch.shape[0], device=W_batch.device, dtype=W_batch.dtype
                        )
                    W_sum += W_batch.detach()
                    W_count += (~torch.isnan(targets)).any(dim=0).to(W_batch.dtype)
            else:
                preds = model(inputs)
            preds_det = preds.detach()
            mask = ~torch.isnan(targets)
            mse = F.mse_loss(preds[mask], targets[mask])
            l2 = torch.cat([p.ravel() for p in model.parameters()]).pow(2).mean()
            loss = mse + model.l2_lambda * l2
            loss.backward()
            if config.clip_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_grad_norm)
            optim.step()
            optim.zero_grad()
        else:
            with torch.no_grad():
                if model.output_mode == "narrow":
                    embeds = model(inputs)
                    if fixed_W is not None:
                        preds_det = embeds @ fixed_W.T
                    else:
                        preds_det = preds_from_embeds(
                            embeds, targets, model.n_outputs, model.l2_lambda
                        )
                else:
                    preds_det = model(inputs)
                mask = ~torch.isnan(targets)
                mse = F.mse_loss(preds_det[mask], targets[mask])
                l2 = torch.cat([p.ravel() for p in model.parameters()]).pow(2).mean()

        total_mse += mse.item()
        total_l2 += l2.item()
        n_batches += 1

        # Accumulate variance explained in-loop (avoids re-iterating the loader)
        with torch.no_grad():
            means = targets.nanmean(dim=0, keepdim=True)
            total_variance += ((targets - means) ** 2).nansum().item()
            remaining_variance += ((targets - preds_det) ** 2).nansum().item()

        if (b + 1) % config.print_interval == 0:
            avg_mse = total_mse / n_batches
            avg_l2 = total_l2 / n_batches
            print(
                f"{prefix}({'T' if train else 'V'}) {b + 1:05d}: "
                f"mse={avg_mse:.4f} | l2={avg_l2:.4f} | "
                f"loss={avg_mse + model.l2_lambda * avg_l2:.4f}"
            )

    avg_mse = total_mse / max(n_batches, 1)
    avg_l2 = total_l2 / max(n_batches, 1)
    var_exp = max(0.0, (total_variance - remaining_variance) / (total_variance + 1e-12))

    result: Dict[str, Any] = {
        "mse": avg_mse,
        "l2": avg_l2,
        "loss": avg_mse + model.l2_lambda * avg_l2,
        "var_explained": var_exp,
    }

    # Compute the epoch-average readout matrix: W[i,h] = mean regression weight
    # of neuron i from embedding dimension h, averaged only over batches where
    # neuron i was observed.
    if train and model.output_mode == "narrow" and W_sum is not None:
        safe_count = W_count.clamp(min=1.0).unsqueeze(-1)  # (N, 1)
        result["W_avg"] = W_sum / safe_count               # (N, H)

    return result


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _save_checkpoint(
    path: str,
    epoch: int,
    model: FoundationModel,
    optim: torch.optim.Optimizer,
    config: ModelConfig,
    history: Dict,
    extra: Optional[Dict] = None,
) -> None:
    ckpt = {
        "epoch": epoch,
        "config": config.to_dict(),
        "model_state_dict": model.state_dict(),
        "optim_state_dict": optim.state_dict(),
        **history,
    }
    if extra:
        ckpt.update(extra)
    torch.save(ckpt, path)


# ---------------------------------------------------------------------------
# Fold training
# ---------------------------------------------------------------------------

def train_fold(
    config: ModelConfig,
    files: List[str],
    neuron_labels: List[Tuple[int, int]],
    total_n_neurons: int,
    run_dir: str,
    wandb_run=None,
) -> Dict:
    k = config.fold_idx
    fold_dir = os.path.join(run_dir, f"fold{k}")
    os.makedirs(fold_dir, exist_ok=True)

    train_loader, test_loader = build_loaders(
        files, neuron_labels, total_n_neurons, config
    )

    # Separate single-process loader for lazy metric — avoids multiprocessing
    # state conflicts when called outside of the main training iteration.
    lazy_loader = (
        DataLoader(
            test_loader.dataset,
            batch_size=config.batch_size,
            num_workers=0,
            collate_fn=collate_fn,
        )
        if config.compute_lazy_metric
        else None
    )

    model = build_model(config, len(files), total_n_neurons)
    optim = build_optimizer(model, config)

    init_params = (
        OrderedDict((n, p.detach().clone()) for n, p in model.named_parameters())
        if config.compute_lazy_metric
        else None
    )

    print(model)
    print(
        f"\n[fold {k}] scope={config.scope}, output_mode={config.output_mode}, "
        f"device={device}, sessions={len(files)}, total_neurons={total_n_neurons}"
    )

    history: Dict = {
        "train_losses": [],
        "test_losses": [],
        "var_exp_train": [],
        "var_exp_test": [],
        "lazy_metrics": [],
    }

    for epoch in range(config.epochs):
        prefix = f"[{epoch:05d}] "

        train_stats = run_epoch(
            model, train_loader, config, train=True, optim=optim, prefix=prefix
        )
        # Extract the epoch-average readout matrix built during training.
        # For narrow mode this is a (N, H) tensor; None for wide mode.
        W_avg: Optional[torch.Tensor] = train_stats.pop("W_avg", None)
        print()
        test_stats = run_epoch(
            model, test_loader, config, train=False, prefix=prefix,
            fixed_W=W_avg,
        )
        print()

        history["train_losses"].append(train_stats["loss"])
        history["test_losses"].append(test_stats["loss"])
        history["var_exp_train"].append(train_stats["var_explained"])
        history["var_exp_test"].append(test_stats["var_explained"])

        # Lazy metric: computed at checkpoint intervals to amortise jvp cost.
        # Uses num_workers=0 loader to avoid multiprocessing conflicts.
        lazy_metric: Optional[float] = None
        if config.compute_lazy_metric:
            at_interval = (epoch + 1) % config.checkpoint_interval == 0
            at_end = epoch == config.epochs - 1
            if at_interval or at_end:
                lazy_metric = get_lazy_metric(model, init_params, lazy_loader, device)
        history["lazy_metrics"].append(lazy_metric)

        line = (
            f"[{epoch:05d}] fold={k} | "
            f"train_ve={train_stats['var_explained']:.4f} | "
            f"test_ve={test_stats['var_explained']:.4f}"
        )
        if lazy_metric is not None:
            line += f" | lazy={lazy_metric:.4f}"
        print(line)

        if wandb_run is not None:
            log = {
                "epoch": epoch,
                "fold": k,
                "train_loss": train_stats["loss"],
                "test_loss": test_stats["loss"],
                "train_var_explained": train_stats["var_explained"],
                "test_var_explained": test_stats["var_explained"],
            }
            if lazy_metric is not None:
                log["lazy_metric"] = lazy_metric
            wandb_run.log(log)

        # Save checkpoint every checkpoint_interval epochs and at the final epoch.
        at_interval = (epoch + 1) % config.checkpoint_interval == 0
        at_end = epoch == config.epochs - 1
        if at_interval or at_end:
            ckpt_path = os.path.join(fold_dir, f"epoch_{epoch:06d}.pt")
            extra: Dict[str, Any] = {}
            if lazy_metric is not None:
                extra["lazy_metric"] = lazy_metric
            if W_avg is not None:
                extra["W_avg"] = W_avg.cpu()
            _save_checkpoint(
                ckpt_path, epoch, model, optim, config, history,
                extra=extra or None,
            )

    return history


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(config: ModelConfig, run_dir: Optional[str] = None) -> Dict:
    files, neuron_labels, total_n_neurons = build_files(config)

    if run_dir is None:
        ts = datetime.now().strftime("%d-%m-%y_%H-%M-%S.%f")
        parts = [config.scope, config.output_mode]
        if config.animal:
            parts.append(config.animal)
        if config.session:
            parts.append(config.session)
        parts.append(ts)
        run_dir = os.path.join(config.data_root, "runs", "_".join(parts))

    os.makedirs(run_dir, exist_ok=True)
    config_path = os.path.join(run_dir, "config.json")
    if not os.path.exists(config_path):  # don't overwrite if parallel folds share run_dir
        with open(config_path, "w") as f:
            f.write(config.to_json())

    wandb_run = None
    if config.use_wandb:
        import wandb  # noqa: F401
        project = f"o2s-{config.scope}-{config.output_mode}"
        wandb_run = wandb.init(
            project=project,
            name=os.path.basename(run_dir),
            config=config.to_dict(),
        )

    result = train_fold(
        config, files, neuron_labels, total_n_neurons, run_dir, wandb_run
    )

    if wandb_run is not None:
        wandb_run.finish()

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train MLP to predict neural activity from behaviour"
    )
    add_config_args(parser)
    parser.add_argument(
        "--run_dir", default=None,
        help="Checkpoint root directory (auto-generated from config + timestamp if omitted). "
             "Parallel fold jobs should share the same --run_dir.",
    )
    args = parser.parse_args()
    config = config_from_args(args)
    main(config, run_dir=args.run_dir)
