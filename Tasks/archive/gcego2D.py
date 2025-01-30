import torch

import sys

from ratinabox.Environment import Environment
from ratinabox.Agent import Agent

from task import *
from build import *
from test_funcs import *

default_params = {
    **Tasks.ego2D.default_params,

    'n_place_cells': 50,
    'n_head_direction_cells': 50,
    'place_cell_scale': 0.2,
    'head_direction_cell_concentration': 0.5
}


def init_func(task):
    config = task.config
    if config.n_inputs == -1:
        config.n_inputs = config.n_place_cells + config.n_head_direction_cells + 2

        config.dict['place_cell_centers'] = torch.rand(config.n_place_cells, 2)
        config.dict['head_direction_cell_centers'] = torch.rand(config.n_head_direction_cells) * 2*np.pi

        input_map = {
            'av': 0,
            'v': 1
        }
        for i in range(config.n_place_cells):
            input_map[f'PC_{i+1}'] = i+2
        for i in range(config.n_head_direction_cells):
            input_map[f'HD_{i+1}'] = i+config.n_place_cells+2

        task.input_map = input_map

def create_data(config, inputs, targets, mask):
        # Create local copies of parameter properties (for brevity's sake)
        batch_size, n_timesteps = inputs.shape[0], inputs.shape[1]
        
        # Create data as per egocentric equivalent (creates inputs[:,:,0-2] and targets[:,:,0-1])
        inputs, targets, mask = Tasks.ego2D.create_data(config, inputs, targets, mask)

        head_direction = torch.atan2(targets[:,:,0], targets[:,:,1])
        head_direction[head_direction<0] += 2*np.pi
        x_position = targets[:,:,4]
        y_position = targets[:,:,5]

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
        
        
        
        inputs[:,:,0] = inputs[:,:,0] # angular velocity
        inputs[:,:,1] = inputs[:,:,5] # linear velocity

        for k in range(batch_size):
            for i in range(config.n_place_cells):
                X, Y = x_position[k], y_position[k]
                inputs[k,:,i+2] = _PC_activity(X, Y, i)
            for i in range(config.n_head_direction_cells):
                Theta = head_direction[k]
                inputs[k,:,i+2+config.n_place_cells] = _HD_activity(Theta, i) 

        return inputs, targets, mask

EGO2D_TASK = Task('gcego2D', 
                    n_inputs=-1, n_outputs=6, 
                    task_specific_params=default_params, 
                    create_data_func=create_data,
                    init_func=init_func,
                    target_map=Tasks.ego2D.target_map,
                    test_func=test_tuning,
                    test_func_args=dict(tuning_vars_list=['HD', 'ego_SD', 'allo_SD', 'AV', 'x', 'y']))




