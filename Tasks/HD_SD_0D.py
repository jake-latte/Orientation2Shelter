import torch

import sys

from task import *
from build import *
from test_funcs import *

# Task Dependencies
import Tasks.vars_0D as template_0D

target_map = {
    'sin_hd': 0,
    'cos_hd': 1,
    'sin_sd': 2,
    'cos_sd': 3
}

def create_data(config, inputs, targets, mask):
    
    vars = template_0D.create_data(config, for_training=(inputs.shape[0] == config.batch_size and inputs.shape[1] == config.n_timesteps))
    inputs, mask = template_0D.fill_inputs(config, inputs, mask, vars)

    targets[:,:,target_map['sin_hd']] = torch.sin(vars['hd'])
    targets[:,:,target_map['cos_hd']] = torch.cos(vars['hd'])
    targets[:,:,target_map['sin_sd']] = torch.sin(vars['sd'])
    targets[:,:,target_map['cos_sd']] = torch.cos(vars['sd'])

    return inputs, targets, vars, mask



HD_SD_0D_TASK = Task('HD_SD-0D', 
                    n_inputs=5, n_outputs=4, 
                    task_specific_params=template_0D.default_params, 
                    create_data_func=create_data,
                    input_map=template_0D.input_map,
                    target_map=target_map,
                    test_func=test_tuning,
                    test_func_args=dict(tuning_vars_list=['HD', 'ego_SD', 'allo_SD', 'AV']))



