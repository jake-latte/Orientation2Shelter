import torch

import sys

from ratinabox.Environment import Environment
from ratinabox.Agent import Agent

from task import *
from build import *
from test_funcs import *


default_params = {
    **Tasks.allo0D.default_params,
    # Standard deviation of noise in angular velocity input
    'xv_step_std': 0.03,
    # Momentum of previous step's angular velocity
    'xv_step_momentum': 0.8
}

input_map = {
    **Tasks.allo0D.input_map,
    'v': 3,
    'x_0': 4
}

target_map = {
    **Tasks.allo0D.target_map,
    'x': 2
}


def create_data(config, inputs, targets, mask):
    # Create local copies of parameter properties (for brevity's sake)
    batch_size, n_timesteps = inputs.shape[0], inputs.shape[1]
    init_duration = config.init_duration
    xv_step_std, xv_step_momentum = config.xv_step_std, config.xv_step_momentum

    inputs, targets, mask = Tasks.allo0D.create_data(config, inputs, targets, mask)

    x_0 = torch.rand(batch_size)
    x_velocity = torch.zeros((batch_size, n_timesteps))
    x_position = torch.tile(x_0.reshape(batch_size,1), dims=(1,n_timesteps))

    for trial in range(batch_size):
        trial_velocity = torch.zeros((n_timesteps - init_duration,))
        trial_position = torch.full((n_timesteps - init_duration,), fill_value=x_0[trial])

        for t in range(0, n_timesteps - init_duration):
            xv_step = np.random.normal(loc=targets[trial,t,target_map['cos_hd']], scale=xv_step_std) + xv_step_momentum * trial_velocity[t-1]
            max_xv_step = 1 - trial_position[t] 
            min_xv_step = -trial_position[t]

            if xv_step > max_xv_step:
                xv_step = max_xv_step
            if xv_step < min_xv_step:
                xv_step = min_xv_step

            trial_velocity[t] = xv_step
            trial_position[t:] += xv_step
        
        x_velocity[trial, init_duration:] = trial_velocity
        x_position[trial, init_duration:] = trial_position

    # Save input and target data streams
    inputs[:,init_duration:,input_map['v']] = x_velocity[:,init_duration:]
    inputs[:,:init_duration,input_map['x_0']] = torch.tile(x_0.reshape(batch_size,1), dims=(1,init_duration))

    targets[:,:,target_map['x']] = x_position[:,:]

    return inputs, targets, mask




ALLO1D_TASK = Task('allo1D', 
                    n_inputs=5, n_outputs=3, 
                    task_specific_params=default_params, 
                    create_data_func=create_data,
                    test_func=test_allo,
                    input_map=input_map,
                    target_map=target_map)





