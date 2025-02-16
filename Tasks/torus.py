import torch

import sys

from task import *
from build import *
from test_funcs import *

default_params = {
    'max_angles_per_trial': 20,
    'pulse_length': 5
}

input_map = {
    'sin_theta': 0,
    'cos_theta': 1,
    'sin_phi': 2,
    'cos_phi': 3
}

target_map = {
    'x': 0,
    'y': 1,
    'z': 2
}

def create_data(config, inputs, targets, mask):
    
    for trial in range(inputs.shape[0]):
        n_theta = torch.randint(0, config.max_angles_per_trial, (1,))
        n_phi = torch.randint(0, config.max_angles_per_trial, (1,))

        theta_times = torch.randint(0, config.n_timesteps, (n_theta,)).sort()[0]
        theta_vals = 2*np.pi*torch.rand((n_theta,))

        phi_times = torch.randint(0, config.n_timesteps, (n_phi,)).sort()[0]
        phi_vals = np.pi*torch.rand((n_phi,))

        theta_out = torch.zeros((config.n_timesteps,))
        phi_out = torch.zeros((config.n_timesteps,))

        for i in range(n_theta):
            this_time = theta_times[i]
            this_theta = theta_vals[i]
            inputs[trial, this_time:this_time+config.pulse_length, input_map['sin_theta']] = torch.sin(this_theta)
            inputs[trial, this_time:this_time+config.pulse_length, input_map['cos_theta']] = torch.cos(this_theta)

            if i < n_theta-1:
                next_time = theta_times[i+1]
                theta_out[this_time+config.pulse_length:next_time] = this_theta
            else:
                theta_out[this_time+config.pulse_length:] = this_theta

        for i in range(n_phi):
            this_time = phi_times[i]
            this_phi = phi_vals[i]
            inputs[trial, this_time:this_time+config.pulse_length, input_map['sin_phi']] = torch.sin(this_phi)
            inputs[trial, this_time:this_time+config.pulse_length, input_map['cos_phi']] = torch.cos(this_phi)

            if i < n_phi-1:
                next_time = phi_times[i+1]
                phi_out[this_time+config.pulse_length:next_time] = this_phi
            else:
                phi_out[this_time+config.pulse_length:] = this_phi

            targets[trial, :, target_map['x']] = (1 + torch.cos(theta_out))*torch.cos(phi_out)
            targets[trial, :, target_map['y']] = (1 + torch.cos(theta_out))*torch.sin(phi_out)
            targets[trial, :, target_map['z']] = torch.sin(theta_out)

    return inputs, targets, {}, mask


TORUS_TASK = Task('torus',
                    task_specific_params=default_params, 
                    create_data_func=create_data,
                    input_map=input_map,
                    target_map=target_map)



