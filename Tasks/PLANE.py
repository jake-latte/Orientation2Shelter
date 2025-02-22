import torch

import sys

from task import *
from build import *
from test_funcs import *

default_params = {
    'min_xy': -1,
    'max_xy': 1,
    'init_duration': 10
}

input_map = {
    'x': 0,
    'y': 1
}

target_map = {
    'x': 0,
    'y': 1
}

def create_data(config, inputs, targets, mask):
    batch_size, n_timesteps = inputs.shape[0], inputs.shape[1]
    init_duration = config.init_duration

    x_vals = (config.max_xy-config.min_xy)*torch.rand((batch_size,)) + config.min_xy
    y_vals = (config.max_xy-config.min_xy)*torch.rand((batch_size,)) + config.min_xy

    inputs[:,:init_duration,input_map['x']] = x_vals.reshape((batch_size,1)).repeat((1,init_duration))
    inputs[:,:init_duration,input_map['y']] = y_vals.reshape((batch_size,1)).repeat((1,init_duration))

    mask[:,:init_duration] = False

    targets[:,:,target_map['x']] = x_vals.reshape((batch_size,1)).repeat((1,n_timesteps))
    targets[:,:,target_map['y']] = y_vals.reshape((batch_size,1)).repeat((1,n_timesteps))
    
    vars = {'x': x_vals, 'y': y_vals}


    return inputs, targets, vars, mask


PLANE_TASK = Task('plane',
                    task_specific_params=default_params, 
                    create_data_func=create_data,
                    input_map=input_map,
                    target_map=target_map)



