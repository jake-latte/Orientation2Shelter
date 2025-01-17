import torch

import sys

from ratinabox.Environment import Environment
from ratinabox.Agent import Agent

from task import *
from build import *
from test_funcs import *


default_params = {
    **Tasks.allo1D.default_params,

    'training_threshold': 0,
    'max_learning_rate': 0.0001,
    'training_convergence_std_threshold': 0.0,
    'learning_rate_schedule': 1.0
}

input_map = {
    **Tasks.allo1D.input_map,
    'sx': 5
}

target_map = {
    **Tasks.allo1D.target_map,
    'sin_sd': 2,
    'cos_sd': 3,
    'x': 4
}

def create_data(config, inputs, targets, mask):
    # Create local copies of parameter properties (for brevity's sake)
    batch_size, n_timesteps = inputs.shape[0], inputs.shape[1]
    
    inputs, targets, mask = Tasks.allo1D.create_data(config, inputs, targets, mask)
                            
    shelter_x_0 = torch.rand(batch_size)
    shelter_x = torch.tile(shelter_x_0.reshape(batch_size,1), dims=(1,n_timesteps))

    head_direction = torch.atan2(targets[:,:,0], targets[:,:,1])

    x_0 = torch.tile(inputs[:,0,4].reshape(batch_size,1), dims=(1,n_timesteps))
    x_position = x_0 + torch.cumsum(inputs[:,:,3], dim=1)

    d_x = shelter_x - x_position
    d_y = 0.1 * torch.ones_like(d_x)
    dist = torch.sqrt(d_x**2 + d_y**2)
    pert = 10e-6 * torch.ones((batch_size,n_timesteps))
    dist[torch.where(dist==0)[0]] += (pert * np.random.choice([1, -1]))[torch.where(dist==0)[0]]
    allo_shelter_angle = torch.atan2(d_y, d_x)

    ego_angle = allo_shelter_angle - head_direction

    # Save input and target data streams
    inputs[:,:config.angle_0_duration,input_map['sx']] = shelter_x[:,:config.angle_0_duration]

    targets[:,:,target_map['sin_sd']] = torch.sin(ego_angle)
    targets[:,:,target_map['cos_sd']] = torch.cos(ego_angle)
    targets[:,:,target_map['x']] = x_position

    return inputs, targets, mask



EGO1D_TASK = Task('ego1D', 
                    n_inputs=6, n_outputs=5, 
                    task_specific_params=default_params, 
                    create_data_func=create_data,
                    input_map=input_map,
                    target_map=target_map,
                    test_func=test_tuning,
                    test_func_args=dict(tuning_vars_list=['HD', 'ego_SD', 'allo_SD', 'AV', 'x']))





