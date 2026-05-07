from __future__ import annotations

import math
from collections import OrderedDict
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.func import functional_call, jvp


class FoundationModel(nn.Module):
    """
    MLP: behaviour (+ session one-hot) → embedding or direct neuron predictions.

    output_mode='narrow'  : n_outputs = embed_dim
                            training loop applies preds_from_embeds to get neuron predictions
    output_mode='wide'    : n_outputs = total_n_neurons
                            training loop uses model output directly
    """

    def __init__(
        self,
        n_inputs: int,
        hidden_layers: List[int],
        n_outputs: int,
        output_mode: str,
        total_n_neurons: int,
        l2_lambda: float = 1e-4,
    ):
        super().__init__()
        assert len(hidden_layers) >= 1, "Require at least one hidden layer"
        assert output_mode in ("narrow", "wide")

        self.n_inputs = n_inputs
        self.hidden_layers = list(hidden_layers)
        self.n_outputs = n_outputs
        self.output_mode = output_mode
        self.total_n_neurons = total_n_neurons
        self.l2_lambda = l2_lambda

        sizes = [n_inputs] + list(hidden_layers) + [n_outputs]
        modules = []
        for i in range(len(sizes) - 1):
            in_sz, out_sz = sizes[i], sizes[i + 1]
            is_last = i == len(sizes) - 2
            linear = nn.Linear(in_sz, out_sz, bias=is_last)
            linear.weight.data = torch.randn(out_sz, in_sz) * (1.0 / math.sqrt(in_sz))
            modules.append(nn.Sequential(linear, nn.Identity() if is_last else nn.Tanh()))

        self.embedding = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedding(x)


# ---------------------------------------------------------------------------
# Narrow-mode analytic readout
# ---------------------------------------------------------------------------

def preds_from_embeds(
    embeds: torch.Tensor,
    targets: torch.Tensor,
    embed_dim: int,
    l2_lambda: float,
    return_weights: bool = False,
):
    """
    Per-batch ridge regression from embedding space to neuron space.
    Handles NaN targets (neurons absent in a given session).

    embeds  : (T, H)  — model output
    targets : (T, N)  — ground truth, NaN where neuron not present
    returns : (T, N) predictions [, (N, H) weight matrix if return_weights=True]
    """
    mask = (~torch.isnan(targets)).to(targets.dtype)          # (T, N)
    Y0 = torch.nan_to_num(targets, nan=0.0)                   # (T, N)

    XX = torch.einsum("th,tH,tn->nhH", embeds, embeds, mask)  # (N, H, H)
    reg = l2_lambda * torch.eye(embed_dim, device=embeds.device, dtype=embeds.dtype)
    A = XX + reg.unsqueeze(0)                                  # (N, H, H)
    XY = torch.einsum("th,tn,tn->nh", embeds, mask, Y0)       # (N, H)

    W = torch.linalg.solve(A, XY.unsqueeze(-1)).squeeze(-1)   # (N, H)
    preds = embeds @ W.T                                       # (T, N)
    if return_weights:
        return preds, W
    return preds


# ---------------------------------------------------------------------------
# Lazy training metric
# ---------------------------------------------------------------------------

@torch.no_grad()
def _sq_norm(x: torch.Tensor) -> torch.Tensor:
    return x.pow(2).sum()


def get_lazy_metric(
    model: FoundationModel,
    init_params: OrderedDict,
    loader,
    device: Optional[str] = None,
    max_batches: Optional[int] = None,
) -> float:
    """
    Laziness = || f_theta(X) - f_lin(X) || / || f_theta(X) ||

    where f_lin(x) = f_{theta0}(x) + J_{theta0}(x)(theta - theta0).

    Measures how far the learned function has moved from the linearisation
    around initialisation.  Uses jvp to avoid materialising the full Jacobian.

    loader should use num_workers=0 to avoid DataLoader multiprocessing
    conflicts when called mid-epoch.
    """
    if device is None:
        device = str(next(model.parameters()).device)

    model.eval()

    theta = OrderedDict((n, p) for n, p in model.named_parameters())
    theta0 = OrderedDict(
        (n, init_params[n].detach().to(device=p.device, dtype=p.dtype))
        for n, p in theta.items()
    )
    delta = OrderedDict((n, theta[n].detach() - theta0[n]) for n in theta)

    num_sq = torch.zeros((), device=device)
    den_sq = torch.zeros((), device=device)

    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        x = batch["inputs"].to(device)

        def f(params):
            return functional_call(model, params, (x,))

        f0, jvp_out = jvp(f, (theta0,), (delta,))
        f_full = f(theta)
        f_lin = f0 + jvp_out

        num_sq += _sq_norm(f_full - f_lin)
        den_sq += _sq_norm(f_full)

    return torch.sqrt(num_sq / (den_sq + 1e-12)).item()
