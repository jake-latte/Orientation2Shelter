import torch

import sys

from task import *
from build import *
from test_funcs import *

target_map = {
    'sin_sd': 0,
    'cos_sd': 1
}

input_map = {
    'sin_hd': 0,
    'cos_hd': 1,
    'x': 2,
    'sx': 3,
    'sy': 4
}

def create_data(config, inputs, targets, mask):
    
    batch_size, n_timesteps = inputs.shape[0], inputs.shape[1]

    vars = Tasks.vars_1D.create_data(config, for_training=(inputs.shape[0] == config.batch_size and inputs.shape[1] == config.n_timesteps))

    inputs[:,:,input_map['sin_hd']] = torch.sin(vars['hd'])
    inputs[:,:,input_map['cos_hd']] = torch.cos(vars['hd'])
    inputs[:,:,input_map['x']] = vars['x']
    inputs[:,:,input_map['sx']] = vars['sx'].reshape((batch_size,1)).repeat((1,n_timesteps))
    inputs[:,:,input_map['sy']] = vars['sy'].reshape((batch_size,1)).repeat((1,n_timesteps))

    targets[:,:,target_map['sin_sd']] = torch.sin(vars['sd'])
    targets[:,:,target_map['cos_sd']] = torch.cos(vars['sd'])

    mask[:,:config.angle_0_duration] = False

    return inputs, targets, vars, mask



SD_1D_TRANS_TASK = Task('SD-1D_trans', 
                    n_inputs=5, n_outputs=2, 
                    task_specific_params=Tasks.vars_1D.default_params, 
                    create_data_func=create_data,
                    input_map=input_map,
                    target_map=target_map,
                    test_func=test_tuning,
                    test_func_args=dict(tuning_vars_list=['ego_SD', 'allo_SD', 'AV', 'x']))



