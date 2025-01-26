import torch

import sys

from task import *
from build import *
from test_funcs import *


import Tasks.vars_2D as template_2D

target_map = {
    'x': 0,
    'y': 1,
    'sin_hd': 2,
    'cos_hd': 3
}

def create_data(config, inputs, targets, mask):
    
    vars = template_2D.create_data(config, for_training=(inputs.shape[0] == config.batch_size and inputs.shape[1] == config.n_timesteps))
    inputs, mask = template_2D.fill_inputs(config, inputs, mask, vars)

    targets[:,:,target_map['x']] = vars['x']
    targets[:,:,target_map['y']] = vars['y']
    targets[:,:,target_map['sin_hd']] = torch.sin(vars['hd'])
    targets[:,:,target_map['cos_hd']] = torch.cos(vars['hd'])

    return inputs, targets, vars, mask



PI_HD_2D_TASK = Task('PI_HD-2D',
                    task_specific_params=template_2D.default_params, 
                    create_data_func=create_data,
                    input_map=template_2D.input_map,
                    target_map=target_map,
                    test_func=test_tuning,
                    test_func_args=dict(tuning_vars_list=['HD', 'AV', 'x', 'y']))



