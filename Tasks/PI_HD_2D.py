import torch

import sys

from task import *
from build import *
from test_funcs import *

target_map = {
    'x': 0,
    'y': 1,
    'sin_hd': 2,
    'cos_hd': 3
}

def create_data(config, inputs, targets, mask):
    
    vars = Tasks.vars_2D.create_data(config, for_training=(inputs.shape[0] == config.batch_size and inputs.shape[1] == config.n_timesteps))
    inputs, mask = Tasks.vars_2D.fill_inputs(config, inputs, mask, vars)

    targets[:,:,target_map['x']] = vars['x']
    targets[:,:,target_map['y']] = vars['y']
    targets[:,:,target_map['sin_hd']] = torch.sin(vars['hd'])
    targets[:,:,target_map['cos_hd']] = torch.cos(vars['hd'])

    return inputs, targets, vars, mask



PI_HD_2D_TASK = Task('PI_HD-2D', 
                    n_inputs=8, n_outputs=4, 
                    task_specific_params=Tasks.vars_2D.default_params, 
                    create_data_func=create_data,
                    input_map=Tasks.vars_2D.input_map,
                    target_map=target_map,
                    test_func=test_tuning,
                    test_func_args=dict(tuning_vars_list=['HD', 'AV', 'x', 'y']))



