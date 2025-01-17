import torch

import sys

from ratinabox.Environment import Environment
from ratinabox.Agent import Agent

from task import *
from build import *
from test_funcs import *

input_map = {
    'v': 0,
    'x_0': 1
}

target_map = {
    'x': 0
}

def create_data(config, inputs, targets, mask):
    # Create local copies of parameter properties (for brevity's sake)
    batch_size, n_timesteps = inputs.shape[0], inputs.shape[1]
    angle_0_duration = config.angle_0_duration
    xv_step_std, xv_step_momentum = config.xv_step_std, config.xv_step_momentum
    
    x_0 = torch.rand(batch_size)
    x_velocity = torch.zeros((batch_size, n_timesteps))
    x_position = torch.tile(x_0.reshape(batch_size,1), dims=(1,n_timesteps))

    for trial in range(batch_size):
        trial_velocity = torch.zeros((n_timesteps - angle_0_duration,))
        trial_position = torch.full((n_timesteps - angle_0_duration,), fill_value=x_0[trial])

        for t in range(0, n_timesteps - angle_0_duration):
            xv_step = xv_step_std * np.random.randn() + xv_step_momentum * trial_velocity[t-1]
            max_xv_step = 1 - trial_position[t] 
            min_xv_step = -trial_position[t]

            if xv_step > max_xv_step:
                xv_step = max_xv_step
            if xv_step < min_xv_step:
                xv_step = min_xv_step

            trial_velocity[t] = xv_step
            trial_position[t:] += xv_step
        
        x_velocity[trial, angle_0_duration:] = trial_velocity
        x_position[trial, angle_0_duration:] = trial_position

    # Save input and target data streams
    inputs[:,angle_0_duration:,input_map['v']] = x_velocity[:,angle_0_duration:]
    inputs[:,:angle_0_duration,input_map['x_0']] = torch.tile(x_0.reshape(batch_size,1), dims=(1,angle_0_duration))

    targets[:,:,target_map['x']] = x_position

    return inputs, targets, mask



POS1D_TASK = Task('pos1D', 
                    n_inputs=2, n_outputs=1, 
                    task_specific_params=Tasks.ego1D.default_params, 
                    create_data_func=create_data,
                    input_map=input_map, target_map=target_map,
                    test_func=test_tuning,
                    test_func_args=dict(tuning_vars_list=['HD', 'V', 'x']))





