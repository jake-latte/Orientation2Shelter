import torch

import sys

from task import *
from build import *
from test_funcs import *

default_params = {
    'min_xy': -1,
    'max_xy': 1,
    'max_xy_per_trial': 20,
    'pulse_length': 5
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
    
    for trial in range(inputs.shape[0]):
        n_x = torch.randint(0, config.max_xy_per_trial, (1,))
        n_y = torch.randint(0, config.max_xy_per_trial, (1,))

        x_times = torch.randint(0, config.n_timesteps, (n_x,)).sort()[0]
        x_vals = (config.max_xy-config.min_xy)*torch.rand((n_x,)) + config.min_xy

        y_times = torch.randint(0, config.n_timesteps, (n_y,)).sort()[0]
        y_vals = (config.max_xy-config.min_xy)*torch.rand((n_y,)) + config.min_xy

        x_out = torch.zeros((config.n_timesteps,))
        y_out = torch.zeros((config.n_timesteps,))

        for i in range(n_x):
            this_time = x_times[i]
            inputs[trial, this_time:this_time+config.pulse_length, input_map['x']] = x_vals[i]

            if i < n_x-1:
                next_time = x_times[i+1]
                x_out[this_time+config.pulse_length:next_time] = x_vals[i]
            else:
                x_out[this_time+config.pulse_length:] = x_vals[i]

        for i in range(n_y):
            this_time = y_times[i]
            inputs[trial, this_time:this_time+config.pulse_length, input_map['y']] = y_vals[i]

            if i < n_y-1:
                next_time = y_times[i+1]
                y_out[this_time+config.pulse_length:next_time] = y_vals[i]
            else:
                y_out[this_time+config.pulse_length:] = y_vals[i]

            targets[trial, :, target_map['x']] = x_out
            targets[trial, :, target_map['y']] = y_out

    vars = {}


    return inputs, targets, vars, mask


PLANE_TASK = Task('plane',
                    task_specific_params=default_params, 
                    create_data_func=create_data,
                    input_map=input_map,
                    target_map=target_map)



