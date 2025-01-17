import torch

import sys

from ratinabox.Environment import Environment
from ratinabox.Agent import Agent

from task import *
from build import *
from test_funcs import *


target_map = {
    'x': 0,
    'y': 1,
}

def create_data(config, inputs, targets, mask):
     # Create local copies of parameter properties (for brevity's sake)
    batch_size, n_timesteps = inputs.shape[0], inputs.shape[1]
    angle_0_duration = config.angle_0_duration
    
    # Create data as per egocentric equivalent (creates inputs[:,:,0-2] and targets[:,:,0-1])
    inputs, targets, mask = Tasks.allo2D.create_data(config, inputs, targets, mask)

    head_direction = torch.atan2(targets[:,:,0], targets[:,:,1])

    x_0 = torch.tile(inputs[:,0,3].reshape(batch_size,1), dims=(1,n_timesteps))
    x_position = x_0 + torch.cumsum(inputs[:,:,5] * targets[:,:,1], dim=1)

    y_0 = torch.tile(inputs[:,0,4].reshape(batch_size,1), dims=(1,n_timesteps))
    y_position = y_0 + torch.cumsum(inputs[:,:,5] * targets[:,:,0], dim=1)

    targets[:,:,target_map['x']] = x_position
    targets[:,:,target_map['y']] = y_position

    return inputs, targets, mask


POS2D_TASK = Task('pos2D', 
                    n_inputs=6, n_outputs=2, 
                    task_specific_params=Tasks.ego2D.default_params, 
                    create_data_func=create_data,
                    input_map=Tasks.allo2D.input_map,
                    target_map=target_map,
                    test_func=test_tuning,
                    test_func_args=dict(tuning_vars_list=['HD', 'AV', 'v', 'x', 'y']))





