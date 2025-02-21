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
    'sin_azim': 0,
    'cos_azim': 1,
    'sin_elev': 2,
    'cos_elev': 3
}

target_map = {
    'x': 0,
    'y': 1,
    'z': 2
}

def create_data(config, inputs, targets, mask):
    
    for trial in range(inputs.shape[0]):
        n_azim = torch.randint(0, config.max_angles_per_trial, (1,))
        n_elev = torch.randint(0, config.max_angles_per_trial, (1,))

        azim_times = torch.randint(0, config.n_timesteps, (n_azim,)).sort()[0]
        azim_vals = 2*np.pi*torch.rand((n_azim,))

        elev_times = torch.randint(0, config.n_timesteps, (n_elev,)).sort()[0]
        elev_vals = np.pi*torch.rand((n_elev,))

        azim_out = torch.zeros((config.n_timesteps,))
        elev_out = torch.zeros((config.n_timesteps,))

        for i in range(n_azim):
            this_time = azim_times[i]
            this_azim = azim_vals[i]
            inputs[trial, this_time:this_time+config.pulse_length, input_map['sin_azim']] = torch.sin(this_azim)
            inputs[trial, this_time:this_time+config.pulse_length, input_map['cos_azim']] = torch.cos(this_azim)

            if i < n_azim-1:
                next_time = azim_times[i+1]
                azim_out[this_time+config.pulse_length:next_time] = this_azim
            else:
                azim_out[this_time+config.pulse_length:] = this_azim

        for i in range(n_elev):
            this_time = elev_times[i]
            this_elev = elev_vals[i]
            inputs[trial, this_time:this_time+config.pulse_length, input_map['sin_elev']] = torch.sin(this_elev)
            inputs[trial, this_time:this_time+config.pulse_length, input_map['cos_elev']] = torch.cos(this_elev)

            if i < n_elev-1:
                next_time = elev_times[i+1]
                elev_out[this_time+config.pulse_length:next_time] = this_elev
            else:
                elev_out[this_time+config.pulse_length:] = this_elev

            targets[trial, :, target_map['x']] = torch.sin(elev_out)*torch.cos(azim_out)
            targets[trial, :, target_map['y']] = torch.sin(elev_out)*torch.sin(azim_out)
            targets[trial, :, target_map['z']] = torch.cos(elev_out)

    return inputs, targets, {}, mask


SPHERE_TASK = Task('sphere',
                    task_specific_params=default_params, 
                    create_data_func=create_data,
                    input_map=input_map,
                    target_map=target_map)



