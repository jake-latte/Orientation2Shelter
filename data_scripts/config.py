from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field, asdict
from typing import List, Optional

BEHAV_COLS = ["hdir", "hsa", "angular_speed", "translational_speed", "x", "y", "dist"]
NORMS = [math.pi, math.pi, 3 * math.pi, 1024.0, 1024.0, 1024.0, 1024.0]


@dataclass
class ModelConfig:
    # --- scope ---
    scope: str = "all"           # "session" | "animal" | "all"
    output_mode: str = "narrow"  # "narrow" | "wide"

    # --- data ---
    data_root: str = "/ceph/branco/Jake/training_data"
    animal: Optional[str] = None
    session: Optional[str] = None

    # --- architecture ---
    hidden_layers: List[int] = field(default_factory=lambda: [64, 64])
    embed_dim: int = 64

    # --- temporal context ---
    past_embed: int = 0
    future_embed: int = 0

    # --- training ---
    l2_lambda: float = 1e-4
    lr: float = 1e-4
    batch_size: int = 512
    epochs: int = 1000
    num_workers: int = 4

    # --- cross-validation ---
    cv_folds: int = 5
    fold_idx: int = 0

    # --- regularisation ---
    clip_grad_norm: Optional[float] = 1.0  # None = disabled

    # --- data sampling ---
    equalize_hdir: bool = True
    n_hdir_bins: int = 30

    # --- logging / checkpointing ---
    print_interval: int = 10
    checkpoint_interval: int = 50
    use_wandb: bool = False
    compute_lazy_metric: bool = True

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, d: dict) -> ModelConfig:
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**known)

    @classmethod
    def from_json(cls, s: str) -> ModelConfig:
        return cls.from_dict(json.loads(s))

    @classmethod
    def from_checkpoint(cls, path: str) -> ModelConfig:
        import torch
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        return cls.from_dict(ckpt["config"])


def add_config_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--scope", default="all", choices=["session", "animal", "all"])
    parser.add_argument("--output_mode", default="narrow", choices=["narrow", "wide"])
    parser.add_argument("--data_root", default="/ceph/branco/Jake/training_data")
    parser.add_argument("--animal", default=None)
    parser.add_argument("--session", default=None)
    parser.add_argument(
        "--hidden_layers", default="64,64",
        help="Comma-separated hidden layer widths, e.g. 128,128,128",
    )
    parser.add_argument("--embed_dim", type=int, default=64)
    parser.add_argument("--past_embed", type=int, default=0)
    parser.add_argument("--future_embed", type=int, default=0)
    parser.add_argument("--l2_lambda", type=float, default=1e-4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--cv_folds", type=int, default=5)
    parser.add_argument("--fold_idx", type=int, default=0)
    parser.add_argument("--clip_grad_norm", type=float, default=1.0)
    parser.add_argument(
        "--no_clip_grad", action="store_true",
        help="Disable gradient clipping",
    )
    parser.add_argument(
        "--equalize_hdir", type=lambda x: x.lower() != "false", default=True,
        metavar="BOOL",
    )
    parser.add_argument("--n_hdir_bins", type=int, default=30)
    parser.add_argument("--print_interval", type=int, default=10)
    parser.add_argument("--checkpoint_interval", type=int, default=50)
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument(
        "--no_lazy_metric", action="store_true",
        help="Skip lazy training metric (useful for wide models when output is large)",
    )
    return parser


def config_from_args(args: argparse.Namespace) -> ModelConfig:
    return ModelConfig(
        scope=args.scope,
        output_mode=args.output_mode,
        data_root=args.data_root,
        animal=args.animal,
        session=args.session,
        hidden_layers=[int(x) for x in args.hidden_layers.split(",")],
        embed_dim=args.embed_dim,
        past_embed=args.past_embed,
        future_embed=args.future_embed,
        l2_lambda=args.l2_lambda,
        lr=args.lr,
        batch_size=args.batch_size,
        epochs=args.epochs,
        num_workers=args.num_workers,
        cv_folds=args.cv_folds,
        fold_idx=args.fold_idx,
        clip_grad_norm=None if args.no_clip_grad else args.clip_grad_norm,
        equalize_hdir=args.equalize_hdir,
        n_hdir_bins=args.n_hdir_bins,
        print_interval=args.print_interval,
        checkpoint_interval=args.checkpoint_interval,
        use_wandb=args.use_wandb,
        compute_lazy_metric=not args.no_lazy_metric,
    )
