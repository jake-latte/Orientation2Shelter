import torch

import sys

from task import *
from build import *
from test_funcs import *


import Tasks.vars_1D_vecvel as template_1D_vecvel

target_map = {
    'sin_x': 0,
    'cos_x': 1,
    'sin_hd': 2,
    'cos_hd': 3,
    'sin_sd': 4,
    'cos_sd': 5
}

def create_data(config, inputs, targets, mask):
    
    vars = template_1D_vecvel.create_data(config, for_training=(inputs.shape[0] == config.batch_size and inputs.shape[1] == config.n_timesteps))
    inputs, mask = template_1D_vecvel.fill_inputs(config, inputs, mask, vars)

    targets[:,:,target_map['sin_x']] = torch.sin(vars['x'])
    targets[:,:,target_map['cos_x']] = torch.cos(vars['x'])
    targets[:,:,target_map['sin_hd']] = torch.sin(vars['hd'])
    targets[:,:,target_map['cos_hd']] = torch.cos(vars['hd'])
    targets[:,:,target_map['sin_sd']] = torch.sin(vars['sd'])
    targets[:,:,target_map['cos_sd']] = torch.cos(vars['sd'])

    return inputs, targets, vars, mask



PI_HD_SD_1D_vecvel_TASK = Task('PI_HD_SD-1D_vecvel',
                    task_specific_params=template_1D_vecvel.default_params, 
                    create_data_func=create_data,
                    input_map=template_1D_vecvel.input_map,
                    target_map=target_map,
                    test_func=test_tuning,
                    test_func_args=dict(tuning_vars_list=['HD', 'ego_SD', 'allo_SD', 'AV', 'x']))



