import torch

import time

from typing import Dict, Tuple

import o2s


default_params = {
    'n_transformed_outputs': 10,
    'transformer_seed': 0
}


def init_func(task):
    config = task.config

    config.dict['n_outputs'] = config.n_transformed_outputs

    torch.manual_seed(config.transformer_seed)
    config.dict['transformer'] = torch.rand((config.n_transformed_outputs, 2), dtype=torch.float64 if config.precise else torch.float32)
    torch.manual_seed(int(time.time()))

    for i in range(config.n_transformed_outputs):
        task.target_map[f'transformed_{i+1}'] = i

class VarsHighdim(o2s.task.Task):
    default_params = default_params
    init_func = staticmethod(init_func)
