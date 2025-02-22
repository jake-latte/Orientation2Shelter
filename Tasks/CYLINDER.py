import torch

import sys

from task import *
from build import *
from test_funcs import *

default_params = {
    'init_duration': 10,
}

input_map = {
    'sin_theta': 0,
    'cos_theta': 1,
    'r': 2
}

target_map = {
    'theta': 0,
    'r': 1
}

def create_data(config, inputs, targets, mask):
    batch_size, n_timesteps = inputs.shape[0], inputs.shape[1]
    init_duration = config.init_duration

    theta_vals = 2*np.pi*torch.rand((batch_size,))
    r_vals = 2*torch.rand((batch_size,)) - 1

    inputs[:,:init_duration,input_map['sin_theta']] = torch.sin(theta_vals).reshape((batch_size,1)).repeat((1,init_duration))
    inputs[:,:init_duration,input_map['cos_theta']] = torch.cos(theta_vals).reshape((batch_size,1)).repeat((1,init_duration))
    inputs[:,:init_duration,input_map['r']] = r_vals.reshape((batch_size,1)).repeat((1,init_duration))

    mask[:,:init_duration] = False

    targets[:,:,target_map['theta']] = theta_vals.reshape((batch_size,1)).repeat((1,n_timesteps))
    targets[:,:,target_map['r']] = r_vals.reshape((batch_size,1)).repeat((1,n_timesteps))

    vars = {'theta': theta_vals, 'r': r_vals}

    return inputs, targets, vars, mask


CYLINDER_TASK = Task('CYLINDER',
                    task_specific_params=default_params, 
                    create_data_func=create_data,
                    input_map=input_map,
                    target_map=target_map)



