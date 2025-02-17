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
    'sin_angle': 0,
    'cos_angle': 1,
    'r': 2
}

target_map = {
    'x': 0,
    'y': 1,
    'z': 2
}

def create_data(config, inputs, targets, mask):
    
    for trial in range(inputs.shape[0]):
        n_angles = torch.randint(0, config.max_angles_per_trial, (1,))
        angle_times = torch.randint(0, config.n_timesteps, (n_angles,)).sort()[0]
        angle_vals = 2*np.pi*torch.rand((n_angles,))
        angle_out = torch.zeros((config.n_timesteps,))

        n_r = torch.randint(0, config.max_angles_per_trial, (1,))
        r_times = torch.randint(0, config.n_timesteps, (n_r,)).sort()[0]
        r_vals = 2*torch.rand((n_r,)) - 1
        r_out = torch.zeros((config.n_timesteps,))

        for i in range(n_angles):
            this_time = angle_times[i]
            this_angle = angle_vals[i]
            inputs[trial, this_time:this_time+config.pulse_length, input_map['sin_angle']] = torch.sin(this_angle)
            inputs[trial, this_time:this_time+config.pulse_length, input_map['cos_angle']] = torch.cos(this_angle)

            if i < n_angles-1:
                next_time = angle_times[i+1]
                angle_out[this_time+config.pulse_length:next_time] = this_angle
            else:
                angle_out[this_time+config.pulse_length:] = this_angle

        for i in range(n_r):
            this_time = r_times[i]
            this_r = r_vals[i]
            inputs[trial, this_time:this_time+config.pulse_length, input_map['r']] = this_r

            if i < n_r-1:
                next_time = r_times[i+1]
                r_out[this_time+config.pulse_length:next_time] = this_r
            else:
                r_out[this_time+config.pulse_length:] = this_r


        targets[trial, :, target_map['x']] = torch.sin(angle_out)
        targets[trial, :, target_map['y']] = torch.cos(angle_out)
        targets[trial, :, target_map['z']] = r_out

    vars = {}


    return inputs, targets, vars, mask


CYLINDER_TASK = Task('cylinder',
                    task_specific_params=default_params, 
                    create_data_func=create_data,
                    input_map=input_map,
                    target_map=target_map)



