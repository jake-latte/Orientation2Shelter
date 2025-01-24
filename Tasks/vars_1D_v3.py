import torch
import numpy as np

from typing import Dict, Tuple

from task import *
from build import *
from config import *
from test_funcs import *

input_map = {
    'av': 0,
    'sin_hd_0': 1,
    'cos_hd_0': 2,
    'xv': 3,
    'yv': 4,
    'sin_x_0': 5,
    'cos_x_0': 6,
    'sx': 7,
    'sy': 8
}


def fill_inputs(config: Config, inputs: torch.Tensor, mask: torch.Tensor, vars: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    batch_size = inputs.shape[0]
    init_duration = config.init_duration
    
    inputs[:,:,input_map['av']] = vars['av']
    inputs[:,:init_duration,input_map['sin_hd_0']] = torch.sin(vars['hd'][:,0]).reshape((batch_size,1)).repeat((1,init_duration))
    inputs[:,:init_duration,input_map['cos_hd_0']] = torch.cos(vars['hd'][:,0]).reshape((batch_size,1)).repeat((1,init_duration))
    inputs[:,:,input_map['xv']] = -vars['v'] * torch.sin(vars['x'])
    inputs[:,:,input_map['yv']] = vars['v'] * torch.cos(vars['x'])
    inputs[:,:init_duration,input_map['sin_x_0']] = torch.sin(vars['x'][:,0].reshape((batch_size, 1)).repeat((1, init_duration)))
    inputs[:,:init_duration,input_map['cos_x_0']] = torch.cos(vars['x'][:,0].reshape((batch_size, 1)).repeat((1, init_duration)))
    inputs[:,:init_duration,input_map['sx']] = vars['sx'].reshape((batch_size,1)).repeat((1,init_duration))
    inputs[:,:init_duration,input_map['sy']] = vars['sy'].reshape((batch_size,1)).repeat((1,init_duration))

    mask[:,:init_duration] = False

    return inputs, mask





