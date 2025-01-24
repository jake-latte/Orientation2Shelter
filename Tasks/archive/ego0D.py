import torch

import sys

from task import *
from build import *
from test_funcs import *

default_params = {
    **Tasks.allo0D.default_params,

    'training_threshold': 0,
    'max_learning_rate': 0.0001,
    'training_convergence_std_threshold': 0,
    'learning_rate_schedule': 1.0
}

input_map = {
    **Tasks.allo0D.input_map,
    'sx': 3,
    'sy': 4
}

target_map = {
    **Tasks.allo0D.target_map,
    'sin_sd': 2,
    'cos_sd': 3
}

def create_data(config, inputs, targets, mask):
    # Create local copies of parameter properties (for brevity's sake)
    batch_size, n_timesteps = inputs.shape[0], inputs.shape[1]
    init_duration = config.init_duration
    
    inputs, targets, mask = Tasks.allo0D.create_data(config, inputs, targets, mask)

    # Extract target head direction from data created by egocentric superclass
    allo_head_angle = torch.atan2(targets[:,:,0], targets[:,:,1])
    allo_head_angle[allo_head_angle<0] += 2*np.pi

    # Initialise allocentric target angle (relative to zero head-direction) for each sequence
    allo_shelter_angle_0 = (torch.rand(batch_size) - 1) * 2 * np.pi
    # Create time-varying allocentric angle as difference between constant allocentric target and
    # current head direction
    ego_sheler_angle = allo_shelter_angle_0.reshape((batch_size,1)).repeat((1,n_timesteps)) - allo_head_angle
    ego_sheler_angle = torch.remainder(ego_sheler_angle, 2 * np.pi)


    # Save input and target data streams
    inputs[:,:init_duration,input_map['sx']] = torch.sin(allo_shelter_angle_0).reshape((batch_size,1)).repeat((1,init_duration))
    inputs[:,:init_duration,input_map['sy']] = torch.cos(allo_shelter_angle_0).reshape((batch_size,1)).repeat((1,init_duration))

    targets[:,:,target_map['sin_sd']] = torch.sin(ego_sheler_angle)
    targets[:,:,target_map['cos_sd']] = torch.cos(ego_sheler_angle)

    return inputs, targets, {'av': inputs[:,:,0], 'hd': allo_head_angle, 'sd': ego_sheler_angle}, mask



EGO0D_TASK = Task('ego0D', 
                    n_inputs=5, n_outputs=4, 
                    task_specific_params=default_params, 
                    create_data_func=create_data,
                    input_map=input_map,
                    target_map=target_map,
                    test_func=test_tuning,
                    test_func_args=dict(tuning_vars_list=['HD', 'ego_SD', 'allo_SD', 'AV']))



