import torch
import numpy as np
import sys

import o2s

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

def get_vars(config):
    batch_size, n_timesteps = config.batch_size, config.n_timesteps
    init_duration = config.init_duration

    theta_vals = 2*np.pi*torch.rand((batch_size,))
    r_vals = 2*torch.rand((batch_size,)) - 1

    return {'theta': theta_vals, 'r': r_vals}

def create_data(config, vars, inputs, targets, mask):
    batch_size, n_timesteps = inputs.shape[0], inputs.shape[1]
    init_duration = config.init_duration

    theta_vals, r_vals = vars['theta'], vars['r']

    inputs[:,:init_duration,input_map['sin_theta']] = torch.sin(theta_vals).reshape((batch_size,1)).repeat((1,init_duration))
    inputs[:,:init_duration,input_map['cos_theta']] = torch.cos(theta_vals).reshape((batch_size,1)).repeat((1,init_duration))
    inputs[:,:init_duration,input_map['r']] = r_vals.reshape((batch_size,1)).repeat((1,init_duration))

    mask[:,:init_duration] = False

    targets[:,:,target_map['theta']] = theta_vals.reshape((batch_size,1)).repeat((1,n_timesteps))
    targets[:,:,target_map['r']] = r_vals.reshape((batch_size,1)).repeat((1,n_timesteps))    

    return inputs, targets, mask


CYLINDER_TASK = o2s.task.Task('CYLINDER',
                    task_specific_params=default_params, 
                    get_vars_func=get_vars,
                    create_data_func=create_data,
                    input_map=input_map,
                    target_map=target_map)



