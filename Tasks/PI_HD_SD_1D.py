import torch

import sys

from task import *
from build import *
from test_funcs import *

target_map = {
    'x': 0,
    'sin_hd': 1,
    'cos_hd': 2,
    'sin_sd': 3,
    'cos_sd': 4
}

def create_data(config, inputs, targets, mask):
    
    batch_size = inputs.shape[0]
    angle_0_duration = config.angle_0_duration
    input_map = Tasks.vars_1D.input_map
    
    vars = Tasks.vars_1D.create_data(config, for_training=(inputs.shape[0] == config.batch_size and inputs.shape[1] == config.n_timesteps))

    inputs[:,:,input_map['av']] = vars['av']
    inputs[:,:angle_0_duration,input_map['sin_hd_0']] = torch.sin(vars['hd'][:,0]).reshape((batch_size,1)).repeat((1,angle_0_duration))
    inputs[:,:angle_0_duration,input_map['cos_hd_0']] = torch.cos(vars['hd'][:,0]).reshape((batch_size,1)).repeat((1,angle_0_duration))
    inputs[:,:angle_0_duration,input_map['sx']] = vars['sx'].reshape((batch_size,1)).repeat((1,angle_0_duration))
    inputs[:,:angle_0_duration,input_map['sy']] = vars['sy'].reshape((batch_size,1)).repeat((1,angle_0_duration))
    inputs[:,:,input_map['v']] = vars['v']
    inputs[:,:angle_0_duration,input_map['x_0']] = vars['x'][:,0].reshape((batch_size, 1)).repeat((1, angle_0_duration))

    targets[:,:,target_map['x']] = vars['x']
    targets[:,:,target_map['sin_hd']] = torch.sin(vars['hd'])
    targets[:,:,target_map['cos_hd']] = torch.cos(vars['hd'])
    targets[:,:,target_map['sin_sd']] = torch.sin(vars['sd'])
    targets[:,:,target_map['cos_sd']] = torch.cos(vars['sd'])

    return inputs, targets, vars, mask



PI_HD_SD_1D_TASK = Task('PI_HD_SD-1D', 
                    n_inputs=7, n_outputs=5, 
                    task_specific_params=Tasks.vars_1D.default_params, 
                    create_data_func=create_data,
                    input_map=Tasks.vars_1D.input_map,
                    target_map=target_map,
                    test_func=test_tuning,
                    test_func_args=dict(tuning_vars_list=['HD', 'ego_SD', 'allo_SD', 'AV', 'x']))



