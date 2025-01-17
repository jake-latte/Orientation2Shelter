import torch

import sys

from task import *
from build import *
from test_funcs import *

target_map = {
    'sin_sd': 0,
    'cos_sd': 1
}

default_params = {
    **Tasks.vars_2D.default_params,

    'n_place_cells': 50,
    'n_head_direction_cells': 50,
    'place_cell_scale': 0.2,
    'head_direction_cell_concentration': 0.5
}

input_map = {
    'av': 0,
    'v': 1,
    'sx': 2,
    'sy': 3
}

def init_func(task):
    config = task.config
    if config.n_inputs == -1:
        config.n_inputs = config.n_place_cells + config.n_head_direction_cells + len(input_map)

        config.__dict__['place_cell_centers'] = torch.rand(config.n_place_cells, 2)
        config.__dict__['head_direction_cell_centers'] = torch.rand(config.n_head_direction_cells) * 2*np.pi

        for i in range(config.n_place_cells):
            input_map[f'PC_{i+1}'] = i+2
        for i in range(config.n_head_direction_cells):
            input_map[f'HD_{i+1}'] = i+config.n_place_cells+2

        task.input_map = input_map


def create_data(config, inputs, targets, mask):
        # Create local copies of parameter properties (for brevity's sake)
        batch_size, n_timesteps = inputs.shape[0], inputs.shape[1]
        angle_0_duration = config.angle_0_duration
        
        # Create data as per egocentric equivalent (creates inputs[:,:,0-2] and targets[:,:,0-1])
        vars = Tasks.vars_2D.create_data(config, for_training=(inputs.shape[0] == config.batch_size and inputs.shape[1] == config.n_timesteps))

        head_direction = vars['hd']
        x_position = vars['x']
        y_position = vars['y']

        def _PC_activity(X, Y, i):
            mu_c, sigma_c = config.place_cell_centers, config.place_cell_scale
            numerator = torch.exp(
                - ((X - mu_c[i, 0])**2 + (Y - mu_c[i, 1])**2) / (2 * sigma_c**2)
            )

            denomenator = torch.zeros(X.shape)
            for j in range(config.n_place_cells):
                denomenator += torch.exp(
                    - ((X - mu_c[j, 0])**2 + (Y - mu_c[j, 1])**2) / (2 * sigma_c**2)
                )

            return numerator / denomenator

        def _HD_activity(Theta, i):
            mu_h, k_h = config.head_direction_cell_centers, config.head_direction_cell_concentration

            numerator = torch.exp(
                k_h * torch.cos(Theta - mu_h[i])
            )

            denomenator = torch.zeros(Theta.shape)
            for j in range(config.n_head_direction_cells):
                denomenator += torch.exp(
                    k_h * torch.cos(Theta - mu_h[j])
                )

            return numerator / denomenator
        
        inputs[:,:,input_map['av']] = vars['av']
        inputs[:,:,input_map['v']] = vars['v']
        inputs[:,:angle_0_duration,input_map['sx']] = vars['sx'].reshape((batch_size,1)).repeat((1,angle_0_duration))
        inputs[:,:angle_0_duration,input_map['sy']] = vars['sy'].reshape((batch_size,1)).repeat((1,angle_0_duration))

        for k in range(batch_size):
            for i in range(config.n_place_cells):
                X, Y = x_position[k], y_position[k]
                inputs[k,:,input_map[f'PC_{i+1}']] = _PC_activity(X, Y, i)
            for i in range(config.n_head_direction_cells):
                Theta = head_direction[k]
                inputs[k,:,input_map[f'HD_{i+1}']] = _HD_activity(Theta, i) 

        targets[:,:,target_map['sin_sd']] = torch.sin(vars['sd'])
        targets[:,:,target_map['cos_sd']] = torch.cos(vars['sd'])

        return inputs, targets, vars, mask



SD_2D_EC_TASK = Task('SD-2D_EC', 
                    n_inputs=-1, n_outputs=2, 
                    task_specific_params=default_params, 
                    create_data_func=create_data,
                    input_map=input_map,
                    target_map=target_map,
                    test_func=test_tuning,
                    test_func_args=dict(tuning_vars_list=['x', 'y', 'AV', 'allo_SD', 'ego_SD']),
                    init_func=init_func)



