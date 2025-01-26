import torch

import sys

from task import *
from build import *
from test_funcs import *

import Tasks.vars_2D as template_2D

target_map = {
    'x': 0,
    'y': 1
}

cueva_params = {
    **template_2D.default_params,
    'W_rec_penalty': 0.0,
    'W_in_penalty': 0.1,
    'W_out_penalty': 0.1,
    'r_penalty': 0.1
}

def cueva_loss_func(task, net, batch):
    _, activity, outputs = net(batch['inputs'], noise=batch['noise'])

    loss_prediction = torch.sum(torch.square(outputs - batch['targets'])[batch['mask']]) / torch.sum(batch['mask']==1)
    loss_activity = task.config.r_penalty * torch.mean(activity**2)

    loss_W_rec = task.config.W_rec_penalty * torch.mean(net.W_rec.weight**2)
    loss_W_in = task.config.W_in_penalty * torch.mean(net.W_in.weight**2)
    loss_W_out = task.config.W_out_penalty * torch.mean(net.W_out.weight**2)

    loss = loss_prediction + loss_activity + loss_W_rec + loss_W_in + loss_W_out

    return loss, outputs

def create_data(config, inputs, targets, mask):

    vars = template_2D.create_data(config, for_training=(inputs.shape[0] == config.batch_size and inputs.shape[1] == config.n_timesteps))
    inputs, mask = template_2D.fill_inputs(config, inputs, mask, vars)

    targets[:,:,target_map['x']] = vars['x']
    targets[:,:,target_map['y']] = vars['y']

    return inputs, targets, vars, mask



PI_2D_TASK = Task('PI-2D',
                    task_specific_params=cueva_params, 
                    create_data_func=create_data,
                    input_map=template_2D.input_map,
                    target_map=target_map,
                    test_func=test_tuning,
                    test_func_args=dict(tuning_vars_list=['HD', 'AV', 'x', 'y']),
                    loss_func=cueva_loss_func)



