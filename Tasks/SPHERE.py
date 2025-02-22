import torch

import sys

from task import *
from build import *
from test_funcs import *

default_params = {
    'init_duration': 10
}

input_map = {
    'sin_theta': 0,
    'cos_theta': 1,
    'sin_phi': 2,
    'cos_phi': 3
}

target_map = {
    'theta': 0,
    'phi': 1,
}

def create_data(config, inputs, targets, mask):
    batch_size, n_timesteps = inputs.shape[0], inputs.shape[1]
    init_duration = config.init_duration

    theta_vals = 2*np.pi*torch.rand((batch_size,))
    phi_vals = np.pi*torch.rand((batch_size,))

    inputs[:,:init_duration,input_map['sin_theta']] = torch.sin(theta_vals).reshape((batch_size,1)).repeat((1,init_duration))
    inputs[:,:init_duration,input_map['cos_theta']] = torch.cos(theta_vals).reshape((batch_size,1)).repeat((1,init_duration))
    inputs[:,:init_duration,input_map['sin_phi']] = torch.sin(phi_vals).reshape((batch_size,1)).repeat((1,init_duration))
    inputs[:,:init_duration,input_map['cos_phi']] = torch.cos(phi_vals).reshape((batch_size,1)).repeat((1,init_duration))

    mask[:,:init_duration] = False

    targets[:,:,target_map['theta']] = theta_vals.reshape((batch_size,1)).repeat((1,n_timesteps))
    targets[:,:,target_map['phi']] = phi_vals.reshape((batch_size,1)).repeat((1,n_timesteps))
    
    vars = {'theta': theta_vals, 'phi': phi_vals}

    return inputs, targets, vars, mask


SPHERE_TASK = Task('SPHERE',
                    task_specific_params=default_params, 
                    create_data_func=create_data,
                    input_map=input_map,
                    target_map=target_map)



