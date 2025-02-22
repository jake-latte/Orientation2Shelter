import torch

import sys

from task import *
from build import *
from test_funcs import *


import Tasks.vars_1D as template_1D

target_map = {
    'x': 0,
    'sin_sd': 1,
    'cos_sd': 2
}

def create_data(config, inputs, targets, mask):
    
    vars = template_1D.create_data(config, for_training=(inputs.shape[0] == config.batch_size and inputs.shape[1] == config.n_timesteps))
    inputs, mask = template_1D.fill_inputs(config, inputs, mask, vars)

    targets[:,:,target_map['x']] = vars['x']
    targets[:,:,target_map['sin_sd']] = torch.sin(vars['sd'])
    targets[:,:,target_map['cos_sd']] = torch.cos(vars['sd'])

    return inputs, targets, vars, mask



PI_SD_1D_TASK = Task('PI_SD-1D',
                    task_specific_params=template_1D.default_params, 
                    create_data_func=create_data,
                    input_map=template_1D.input_map,
                    target_map=target_map,
                    test_func=test_tuning,
                    test_func_args=dict(tuning_vars_list=['ego_SD', 'allo_SD', 'AV', 'x']))



