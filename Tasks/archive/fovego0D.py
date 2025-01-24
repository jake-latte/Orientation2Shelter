import torch

import sys

from task import *
from build import *
from test_funcs import *

default_params = {
    **Tasks.ego0D.default_params,

    'fov': 30.0,
    'n_timesteps': 1000
}

def create_data(config, inputs, targets, mask):
    # Create local copies of parameter properties (for brevity's sake)
    batch_size, n_timesteps = inputs.shape[0], inputs.shape[1]
    init_duration = config.init_duration
    fov = config.fov * np.pi/180
    
    inputs, targets, mask = Tasks.ego0D.create_data(config, inputs, targets, mask)

    # Extract target head direction from data created by egocentric superclass
    allo_head_angle = torch.atan2(targets[:,:,0], targets[:,:,1])
    allo_shelter_angle = torch.atan2(inputs[:,0,3], inputs[:,0,4]).reshape((batch_size, 1)).repeat((1, n_timesteps))

    sin_allo_shelter_angle_fov = torch.where(np.abs(allo_head_angle - allo_shelter_angle) <= fov/2, torch.sin(allo_shelter_angle), torch.zeros_like(allo_shelter_angle))
    cos_allo_shelter_angle_fov = torch.where(np.abs(allo_head_angle - allo_shelter_angle) <= fov/2, torch.cos(allo_shelter_angle), torch.zeros_like(allo_shelter_angle))


    # Save input and target data streams
    inputs[:,:,Tasks.ego0D.input_map['sx']] = sin_allo_shelter_angle_fov
    inputs[:,:,Tasks.ego0D.input_map['sy']] = cos_allo_shelter_angle_fov

    return inputs, targets, mask



FOVEGO0D_TASK = Task('fovego0D', 
                    n_inputs=5, n_outputs=4, 
                    task_specific_params=default_params, 
                    create_data_func=create_data,
                    input_map=Tasks.ego0D.input_map,
                    target_map=Tasks.ego0D.target_map,
                    test_func=test_tuning,
                    test_func_args=dict(tuning_vars_list=['HD', 'ego_SD', 'allo_SD', 'AV']))



