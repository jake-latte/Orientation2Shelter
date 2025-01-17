import torch

import sys

from task import *
from build import *
from test_funcs import *

input_map = {
    'sin_hd_0': 0,
    'cos_hd_0': 1,
    'sin_sd_0': 2,
    'cos_sd_0': 3,
    'sin_E': 4,
    'cos_E': 5,
    'sin_N': 6,
    'cos_N': 7,
    'sin_W': 8,
    'cos_W': 9,
    'sin_S': 10,
    'cos_S': 11
}

def create_data(config, inputs, targets, mask):
    # Create local copies of parameter properties (for brevity's sake)
    batch_size, n_timesteps = inputs.shape[0], inputs.shape[1]
    angle_0_duration = config.angle_0_duration
    
    inputs, targets, mask = Tasks.allo0D.create_data(config, inputs, targets, mask)

    # Extract target head direction from data created by egocentric superclass
    allo_head_angle = torch.atan2(targets[:,:,0], targets[:,:,1])
    allo_head_angle_0 = allo_head_angle[:,0]

    # Initialise allocentric target angle (relative to zero head-direction) for each sequence
    allo_shelter_angle_0 = (torch.rand(batch_size) - 1) * 2 * np.pi
    # Create time-varying allocentric angle as difference between constant allocentric target and
    # current head direction
    ego_sheler_angle = allo_shelter_angle_0.reshape((batch_size,1)).repeat((1,n_timesteps)) - allo_head_angle
    ego_sheler_angle = torch.remainder(ego_sheler_angle, 2 * np.pi)

    landmarks = torch.tensor([
        0, np.pi/2, np.pi, 3 * np.pi/2
    ])
    landmark_allo_head_angle = torch.stack([allo_head_angle for _ in range(4)], dim=2)
    landmark_angles = landmarks.tile(dims=(batch_size, n_timesteps, 1)) - landmark_allo_head_angle
    landmark_angles = torch.remainder(landmark_angles, 2 * np.pi)

    inputs = np.zeros_like(inputs)

    inputs[:,:angle_0_duration,0] = torch.sin(allo_head_angle_0).reshape((batch_size,1)).repeat((1,angle_0_duration))
    inputs[:,:angle_0_duration,1] = torch.cos(allo_head_angle_0).reshape((batch_size,1)).repeat((1,angle_0_duration))
    inputs[:,:angle_0_duration,2] = torch.sin(allo_shelter_angle_0).reshape((batch_size,1)).repeat((1,angle_0_duration))
    inputs[:,:angle_0_duration,3] = torch.cos(allo_shelter_angle_0).reshape((batch_size,1)).repeat((1,angle_0_duration))
    inputs[:,:,4:] = torch.sin(landmark_angles)
    inputs[:,:,8:] = torch.cos(landmark_angles)

    targets[:,:,Task.ego0D.target_map['sin_sd']] = torch.sin(ego_sheler_angle)
    targets[:,:,Task.ego0D.target_map['cos_sd']] = torch.cos(ego_sheler_angle)

    return inputs, targets, mask



LANDMARKEGO0D_TASK = Task('landmarkego0D', 
                            n_inputs=12, n_outputs=4, 
                            task_specific_params=Tasks.ego0D.default_params, 
                            create_data_func=create_data,
                            input_map=input_map,
                            target_map=Tasks.ego0D.target_map,
                            test_func=test_tuning,
                            test_func_args=dict(tuning_vars_list=['HD', 'ego_SD', 'allo_SD', 'AV']))



