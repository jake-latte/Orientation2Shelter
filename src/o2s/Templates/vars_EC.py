import torch
import numpy as np
from typing import Dict, Tuple
import math

import o2s
import o2s.Templates.vars_2D as template_2D



default_params = {
    'cells_initialised': False,
    'n_position_place_cells': 0,
    'n_shelter_place_cells': 50,
    'n_head_direction_cells': 25,
    'place_cell_scale': 0.1,
    'head_direction_cell_scale': 0.1
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
    mu_c = mu_c.to(X.device)
    dist_sq = (
        (X.unsqueeze(-1) - mu_c[:, 0]) ** 2 +
        (Y.unsqueeze(-1) - mu_c[:, 1]) ** 2
    )
    
    # Compute exponent for each cell: shape (B, T, N).
    # If sigma_c is a scalar, it will broadcast automatically.
    pc_activity = torch.exp(-dist_sq / (2.0 * sigma_c**2))
    
    return pc_activity

def HD_activity(Theta: torch.Tensor,
                mu_h: torch.Tensor,
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
    diff = torch.remainder(diff + np.pi, 2*np.pi) - np.pi

    hd_activity = torch.exp(-diff**2 / (2 * scale**2) )

    return hd_activity

def closest_factors(n):
    if n==0:
        return 0, 0
    root = int(math.sqrt(n))
    for i in range(root, 0, -1):
        if n % i == 0:
            return i, n // i

def init_func(task):
    config = task.config

    n_x, n_y = closest_factors(config.n_position_place_cells)
    x_centers, y_centers = torch.meshgrid(torch.linspace(-1, 1, n_x), torch.linspace(-1, 1, n_y), indexing='ij')
    config.dict['position_place_cell_centers'] = torch.stack([x_centers.flatten(), y_centers.flatten()]).T

    n_sx, n_sy = closest_factors(config.n_shelter_place_cells)
    sx_centers, sy_centers = torch.meshgrid(torch.linspace(-1, 1, n_sx), torch.linspace(-1, 1, n_sy), indexing='ij')
    config.dict['shelter_place_cell_centers'] = torch.stack([sx_centers.flatten(), sy_centers.flatten()]).T

    config.dict['head_direction_cell_centers'] = torch.linspace(0, 2*np.pi, config.n_head_direction_cells+1)[:-1]

    input_map = {k: v for k, v in task.input_map.items() if 'position_PC' not in k and 'shelter_PC' not in k and 'HD' not in k}
    n_other_inputs = len(input_map)
    for i in range(config.n_position_place_cells):
        input_map[f'position_PC_{i+1}'] = n_other_inputs + i
    for i in range(config.n_shelter_place_cells):
        input_map[f'shelter_PC_{i+1}'] = n_other_inputs + config.n_position_place_cells + i
    for i in range(config.n_head_direction_cells):
        input_map[f'HD_{i+1}'] = n_other_inputs + config.n_position_place_cells + config.n_shelter_place_cells + i
    task.input_map = input_map
    config.n_inputs = len(input_map)
    config.n_other_inputs = n_other_inputs

    # config.position_place_cell_centers = config.position_place_cell_centers.to(config.device)
    # config.shelter_place_cell_centers = config.shelter_place_cell_centers.to(config.device)
    # config.head_direction_cell_centers = config.head_direction_cell_centers.to(config.device)


def fill_head_direction_cell_inputs(config: o2s.config.Config, vars: Dict[str, torch.Tensor], inputs: torch.Tensor) -> torch.Tensor:
    init_duration, batch_size = config.init_duration, inputs.shape[0]

    head_direction = vars['hd'][:,:init_duration]

    inputs[:,:init_duration,-config.n_head_direction_cells:] = HD_activity(
        Theta=head_direction, 
        mu_h=config.head_direction_cell_centers.to(head_direction.device), scale=config.head_direction_cell_scale) 
    
    return inputs

def fill_place_cell_inputs(config: o2s.config.Config, vars: Dict[str, torch.Tensor], inputs: torch.Tensor) -> torch.Tensor:
    init_duration, batch_size, n_timesteps = config.init_duration, inputs.shape[0], inputs.shape[1]

    sx = vars['sx'].reshape((batch_size,1)).repeat((1,init_duration))
    sy = vars['sy'].reshape((batch_size,1)).repeat((1,init_duration))

    if 'x' in vars and 'y' not in vars:
        x = torch.cos(vars['x'][:,:init_duration])
        y = torch.sin(vars['x'][:,:init_duration])
    elif 'x' in vars and 'y' in vars:
        x = vars['x'][:,:init_duration]
        y = vars['y'][:,:init_duration]
    else:
        x, y = None, None
        assert config.n_position_place_cells == 0

    if config.n_position_place_cells > 0:
        start_i = config.n_other_inputs
        end_i = config.n_other_inputs + config.n_position_place_cells
        inputs[:,:init_duration,start_i:end_i] = PC_activity(X=x, Y=y, mu_c=config.position_place_cell_centers.to(x.device), sigma_c=config.place_cell_scale)
    if config.n_shelter_place_cells > 0:
        start_i = config.n_other_inputs + config.n_position_place_cells
        end_i = config.n_other_inputs + config.n_position_place_cells + config.n_shelter_place_cells
        inputs[:,:init_duration,start_i:end_i] = PC_activity(X=sx, Y=sy, mu_c=config.shelter_place_cell_centers.to(sx.device), sigma_c=config.place_cell_scale)
    
    return inputs

def fill_inputs(config: o2s.config.Config, vars: Dict[str, torch.Tensor], inputs: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    init_duration, batch_size = config.init_duration, inputs.shape[0]

    inputs = fill_head_direction_cell_inputs(config, vars, inputs)
    inputs = fill_place_cell_inputs(config, vars, inputs)

    mask[:,:init_duration] = False

    return inputs, mask

class VarsEC(o2s.task.Task):
    default_params = default_params
    fill_inputs = staticmethod(fill_inputs)
    init_func = staticmethod(init_func)
