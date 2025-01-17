import torch

import sys

from task import *
from build import *
from test_funcs import *

input_map = Tasks.ego0D.input_map
target_map = {
    'sin_sd': 0,
    'cos_sd': 1
}

def create_data(config, inputs, targets, mask):
    # Create local copies of parameter properties (for brevity's sake)
    batch_size, n_timesteps = inputs.shape[0], inputs.shape[1]
    angle_0_duration = config.angle_0_duration
    
    inputs, targets, mask = Tasks.ego1D.create_data(config, inputs, targets, mask)

    # Extract target head direction from data created by egocentric superclass
    allo_head_angle = torch.atan2(targets[:,:,0], targets[:,:,1])

    # Initialise allocentric target angle (relative to zero head-direction) for each sequence
    allo_shelter_angle_0 = (torch.rand(batch_size) - 1) * 2 * np.pi
    # Create time-varying allocentric angle as difference between constant allocentric target and
    # current head direction
    ego_sheler_angle = allo_shelter_angle_0.reshape((batch_size,1)).repeat((1,n_timesteps)) - allo_head_angle
    ego_sheler_angle = torch.remainder(ego_sheler_angle, 2 * np.pi)


    # Save input and target data streams
    inputs[:,:angle_0_duration,input_map['sx']] = torch.sin(allo_shelter_angle_0).reshape((batch_size,1)).repeat((1,angle_0_duration))
    inputs[:,:angle_0_duration,input_map['sy']] = torch.cos(allo_shelter_angle_0).reshape((batch_size,1)).repeat((1,angle_0_duration))

    targets[:,:,target_map['sin_sd']] = torch.sin(ego_sheler_angle)
    targets[:,:,target_map['cos_sd']] = torch.cos(ego_sheler_angle)

    return inputs, targets, mask



EGO0D_TASK = Task('ego0D_lHD', 
                    n_inputs=5, n_outputs=2, 
                    task_specific_params=Tasks.ego0D.default_params, 
                    create_data_func=create_data)



