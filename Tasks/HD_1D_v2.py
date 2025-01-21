import torch

import sys

from task import *
from build import *
from test_funcs import *

target_map = {
    'sin_hd': 0,
    'cos_hd': 1
}

def create_data(config, inputs, targets, mask):
    
    vars = Tasks.vars_1D_v2.create_data(config, for_training=(inputs.shape[0] == config.batch_size and inputs.shape[1] == config.n_timesteps))
    inputs, mask = Tasks.vars_1D_v2.fill_inputs(config, inputs, mask, vars)

    targets[:,:,target_map['sin_hd']] = torch.sin(vars['hd'])
    targets[:,:,target_map['cos_hd']] = torch.cos(vars['hd'])

    return inputs, targets, vars, mask



HD_1D_TASK = Task('HD-1D', 
                    n_inputs=7, n_outputs=2, 
                    task_specific_params=Tasks.vars_1D_v2.default_params, 
                    create_data_func=create_data,
                    input_map=Tasks.vars_1D_v2.input_map,
                    target_map=target_map,
                    test_func=test_tuning,
                    test_func_args=dict(tuning_vars_list=['HD', 'AV', 'x']))



