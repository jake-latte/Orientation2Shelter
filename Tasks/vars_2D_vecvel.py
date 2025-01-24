import torch

from typing import Dict, Tuple

import Tasks.vars_2D_linvel
import Tasks.vars_2D_linvel
from task import *
from build import *
from config import *
from test_funcs import *

default_params = Tasks.vars_2D_linvel.default_params

input_map = {
    'av': 0,
    'sin_hd_0': 1,
    'cos_hd_0': 2,
    'sx': 3,
    'sy': 4,
    'x_0': 5,
    'y_0': 6,
    'xv': 7,
    'yv': 8
}

def create_data(config, for_training=True):

    vars = Tasks.vars_2D_linvel.create_data(config, for_training=for_training)

    hd = vars['hd']
    v = vars['v']
    del vars['v']

    vars['xv'] = v * torch.cos(hd)
    vars['yv'] = v * torch.sin(hd)

    return vars

def fill_inputs(config: Config, inputs: torch.Tensor, mask: torch.Tensor, vars: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    init_duration, batch_size = config.init_duration, inputs.shape[0]

    inputs[:,:,input_map['av']] = vars['av']
    inputs[:,:init_duration,input_map['sin_hd_0']] = torch.sin(vars['hd'][:,:init_duration])
    inputs[:,:init_duration,input_map['cos_hd_0']] = torch.cos(vars['hd'][:,:init_duration])
    inputs[:,:init_duration,input_map['sx']] = vars['sx'].reshape((batch_size,1)).repeat((1,init_duration))
    inputs[:,:init_duration,input_map['sy']] = vars['sy'].reshape((batch_size,1)).repeat((1,init_duration))
    inputs[:,:init_duration,input_map['x_0']] = vars['x'][:,0].reshape((batch_size,1)).repeat((1,init_duration))
    inputs[:,:init_duration,input_map['y_0']] = vars['y'][:,0].reshape((batch_size,1)).repeat((1,init_duration))
    inputs[:,:,input_map['xv']] = vars['xv']
    inputs[:,:,input_map['yv']] = vars['yv']

    mask[:,:init_duration] = False

    return inputs, mask

