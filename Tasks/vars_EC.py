import torch

from typing import Dict, Tuple

from task import *
from build import *
from config import *
from test_funcs import *


import Tasks.vars_2D as template_2D



default_params = {
    'n_place_cells': 50,
    'n_head_direction_cells': 25,
    'place_cell_scale': 0.2,
    'head_direction_cell_concentration': 0.5
}

def PC_activity(X: torch.Tensor, 
                Y: torch.Tensor, 
                mu_c: torch.Tensor, 
                sigma_c: float) -> torch.Tensor:
    """
    Vectorized place-cell activity function.

    Args:
        X:       (B, T) x-coordinates for BxT samples
        Y:       (B, T) y-coordinates for BxT samples
        mu_c:    (N, 2) center positions for the N place cells
        sigma_c: scalar float (or a broadcastable shape) for the place-cell scale

    Returns:
        (B, T, N) tensor:
            The activity for each sample (b,t) and each cell i, i.e. the ratio
            exp(-((X - mu_i,0)^2 + (Y - mu_i,1)^2)/(2*sigma^2))  
            -------------------------------------------------
            Sum over j of same expression for mu_j
    """

    # X, Y have shape (B, T). We'll add a dimension so we can broadcast against (N,).
    # mu_c has shape (N, 2): mu_c[:, 0] is x-centers, mu_c[:, 1] is y-centers.
    # After broadcasting:
    #    (X.unsqueeze(-1) - mu_c[:,0]) will have shape (B, T, N).
    
    # Compute squared distance from each (X, Y) to each cell center mu_c[i].
    dist_sq = (
        (X.unsqueeze(-1) - mu_c[:, 0]) ** 2 +
        (Y.unsqueeze(-1) - mu_c[:, 1]) ** 2
    )
    
    # Compute exponent for each cell: shape (B, T, N).
    # If sigma_c is a scalar, it will broadcast automatically.
    exps = torch.exp(-dist_sq / (2.0 * sigma_c**2))
    
    # Sum across the N dimension to get the denominator: shape (B, T, 1).
    denom = exps.sum(dim=-1, keepdim=True)
    
    # Divide to get each cell's fraction: shape (B, T, N).
    pc_activity = exps / denom  # broadcasts along the last dimension
    
    return pc_activity

def HD_activity(Theta: torch.Tensor,
                mu_h: torch.Tensor,
                k_h: float,
                scale: float) -> torch.Tensor:
    """
    Vectorized head-direction activity function.

    Args:
        Theta: (B, T) or (T,) head-direction angles
        mu_h:  (N,) center angles for N head-direction cells
        k_h:   concentration parameter (scalar)
        scale: scaling factor (scalar)

    Returns:
        A tensor of shape (B, T, N) (or (T, N) if Theta is 1D) giving
        the normalized head-direction activity for each cell i across samples.
    """

    # If Theta is 2D (B, T), unsqueeze along the last dimension => (B, T, 1)
    # If Theta is 1D (T,), unsqueeze => (T, 1)
    # mu_h is (N,), so the broadcasted difference has shape (B, T, N) or (T, N).
    diff = Theta.unsqueeze(-1) - mu_h  # shape: (B, T, N) or (T, N)

    # Compute the exponent for each cell
    exps = torch.exp(k_h * torch.cos(diff))  # shape: (B, T, N)

    # Sum across cells (the last dimension), keep that dimension for broadcasting
    denom = exps.sum(dim=-1, keepdim=True)    # shape: (B, T, 1)

    # Compute normalized activity
    hd_activity = (exps / denom) * scale      # shape: (B, T, N)

    return hd_activity

def init_func(task):
    config = task.config
    if config.n_inputs == 0:
        config.n_inputs = config.n_place_cells + config.n_head_direction_cells

        config.dict['place_cell_centers'] = 2*torch.rand(config.n_place_cells, 2) - 1

        config.dict['head_direction_cell_centers'] = torch.rand(config.n_head_direction_cells) * 2*np.pi
        scale = torch.min(torch.stack([
            torch.sum(
                torch.exp(
                    config.head_direction_cell_concentration * torch.cos(config.head_direction_cell_centers[i] - config.head_direction_cell_centers)))
            for i in range(config.n_head_direction_cells)], dim=0)) / np.exp(config.head_direction_cell_concentration)
        config.dict['head_direction_cell_scale'] = scale.item()

        input_map = {}
        for i in range(config.n_place_cells):
            input_map[f'PC_{i+1}'] = i
        for i in range(config.n_head_direction_cells):
            input_map[f'HD_{i+1}'] = config.n_place_cells + i

        task.input_map = input_map


def fill_head_direction_cell_inputs(config: Config, inputs: torch.Tensor, vars: Dict[str, torch.Tensor]) -> torch.Tensor:
    init_duration, batch_size = config.init_duration, inputs.shape[0]

    head_direction = vars['hd']

    inputs[:,:,config.n_place_cells:] = HD_activity(
        Theta=head_direction, 
        mu_h=config.head_direction_cell_centers, k_h=config.head_direction_cell_concentration, scale=config.head_direction_cell_scale) 
    
    return inputs

def fill_place_cell_inputs(config: Config, inputs: torch.Tensor, vars: Dict[str, torch.Tensor]) -> torch.Tensor:
    init_duration, batch_size, n_timesteps = config.init_duration, inputs.shape[0], inputs.shape[1]


    sx = vars['sx'].reshape((batch_size,1)).repeat((1,n_timesteps))
    sy = vars['sy'].reshape((batch_size,1)).repeat((1,n_timesteps))

    if '0D' in config.task:
        x = torch.zeros((batch_size, n_timesteps))
        y = torch.zeros((batch_size, n_timesteps))
    elif '1D' in config.task:
        x = torch.cos(vars['x'])
        y = torch.sin(vars['x'])
    else:
        x = vars['x']
        y = vars['y']

    n_self_pos = config.n_place_cells//2
    inputs[:,:,:n_self_pos] = PC_activity(X=x, Y=y, mu_c=config.place_cell_centers[:n_self_pos], sigma_c=config.place_cell_scale)
    inputs[:,:,n_self_pos:config.n_place_cells] = PC_activity(X=sx, Y=sy, mu_c=config.place_cell_centers[n_self_pos:], sigma_c=config.place_cell_scale)
    
    return inputs

def fill_inputs(config: Config, inputs: torch.Tensor, mask: torch.Tensor, vars: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    init_duration, batch_size = config.init_duration, inputs.shape[0]

    inputs = fill_head_direction_cell_inputs(config, inputs, vars)
    inputs = fill_place_cell_inputs(config, inputs, vars)

    mask[:,:init_duration] = False

    return inputs, mask





