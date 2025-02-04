import torch

import sys

from task import *
from build import *
from test_funcs import *


import Tasks.vars_2D_vecvel as template_2D_vecvel

default_parms = {
    **template_2D_vecvel.default_params,
    'PI_penalty_lambda': 10
}

def loss_func(task: Task, net: 'RNN', batch: dict) -> Tuple[torch.tensor, torch.tensor]:
    for_training = (batch['inputs'].shape[0] == task.config.batch_size and batch['inputs'].shape[1] == task.config.n_timesteps)

    _, activity, outputs = net(batch['inputs'], noise=batch['noise'])

    loss_all = torch.sum(torch.square(outputs - batch['targets'])[batch['mask']]) / torch.sum(batch['mask']==1)
    loss_PI = torch.sum(torch.square(
        outputs[:,:,[task.target_map['x'], task.target_map['y']]] - batch['targets'][:,:,[task.target_map['x'], task.target_map['y']]])[batch['mask'][:,:,[task.target_map['x'], task.target_map['y']]]]) / torch.sum(batch['mask'][:,:,[task.target_map['x'], task.target_map['y']]]==1)
    
    loss_prediction = loss_all + (task.config.PI_penalty_lambda - 1) * loss_PI

    # Rate L2
    loss_activity = task.config.rate_lambda * torch.mean(activity**2)

    # I think more appropriate:
    loss_weight = task.config.weight_lambda * (torch.mean(net.W_rec.weight**2) + torch.mean(net.W_in.weight**2) + torch.mean(net.W_out.weight**2))

    loss = loss_prediction + loss_activity + loss_weight

    return loss, outputs

    


    return loss, outputs

target_map = {
    'x': 0,
    'y': 1,
    'sin_hd': 2,
    'cos_hd': 3,
    'sin_sd': 4,
    'cos_sd': 5
}

def create_data(config, inputs, targets, mask):

    vars = template_2D_vecvel.create_data(config, for_training=(inputs.shape[0] == config.batch_size and inputs.shape[1] == config.n_timesteps))
    inputs, mask = template_2D_vecvel.fill_inputs(config, inputs, mask, vars)

    targets[:,:,target_map['x']] = vars['x']
    targets[:,:,target_map['y']] = vars['y']
    targets[:,:,target_map['sin_hd']] = torch.sin(vars['hd'])
    targets[:,:,target_map['cos_hd']] = torch.cos(vars['hd'])
    targets[:,:,target_map['sin_sd']] = torch.sin(vars['sd'])
    targets[:,:,target_map['cos_sd']] = torch.cos(vars['sd'])

    return inputs, targets, vars, mask



PI_HD_SD_2D_vecvel_TASK = Task('PI_HD_SD-2D_vecvel',
                            task_specific_params=default_params, 
                            create_data_func=create_data,
                            input_map=template_2D_vecvel.input_map,
                            target_map=target_map,
                            test_func=test_tuning,
                            test_func_args=dict(tuning_vars_list=['HD', 'ego_SD', 'allo_SD', 'AV', 'x', 'y']))



