import torch

import sys
from typing import Tuple

import o2s
import o2s.Templates.vars_0D as template_0D
import o2s.Templates.vars_EC as template_EC

input_map = {
    'av': 0
}

target_map = {
    'sin_sd': 0,
    'cos_sd': 1,
}

default_params = {
    **template_0D.default_params,
    'n_position_place_cells': 0,
    'n_shelter_place_cells': 25,
    'n_head_direction_cells': 25,
    'head_direction_cell_scale': template_EC.default_params['head_direction_cell_scale'],
    'place_cell_scale': template_EC.default_params['place_cell_scale'],
    'rank_lambda': 0.1,
    'target_rank': -1,
    'target_sval': 1
}


def create_data(config, vars, inputs, targets, mask):

    inputs, mask = template_EC.fill_inputs(config, vars, inputs, mask)

    inputs[:,:,input_map['av']] = vars['av']

    targets[:,:,target_map['sin_sd']] = torch.sin(vars['sd'])
    targets[:,:,target_map['cos_sd']] = torch.cos(vars['sd'])

    return inputs, targets, mask



def loss_func(task: o2s.task.Task, net: 'RNN', batch: dict) -> Tuple[torch.tensor, torch.tensor]:
    states, activity, outputs = net(batch['inputs'], noise=batch['noise'], offload=task.config.conserve_vram, collapse=task.config.conserve_vram)

    # MSE of (masked) outputs
    loss_prediction = torch.sum(torch.square(outputs - batch['targets'])[batch['mask']]) / torch.sum(batch['mask']==1)

    # Rate Loss
    if task.config.conserve_vram:
        loss_activity = task.config.rate_lambda * activity
    else:
        if task.config.rate_loss_type == 1:
            loss_activity = task.config.rate_lambda * torch.mean(torch.abs(activity))
        else:
            loss_activity = task.config.rate_lambda * torch.mean(torch.square(activity))

    # Weight Loss
    if task.config.weight_loss_type == 1:
        loss_weight = 0
        for weight, include in zip([net.W_rec.weight, net.W_in.weight, net.W_out.weight], [task.config.learn_W_rec, task.config.learn_W_in, task.config.learn_W_out]):
            if include:
                loss_weight += task.config.weight_lambda * torch.mean(torch.abs(weight))
    else:
        loss_weight = 0
        for weight, include in zip([net.W_rec.weight, net.W_in.weight, net.W_out.weight], [task.config.learn_W_rec, task.config.learn_W_in, task.config.learn_W_out]):
            if include:
                loss_weight += task.config.weight_lambda * torch.mean(torch.square(weight))

    # Rank Loss
    _, S, _ = torch.linalg.svd(states, full_matrices=True, compute_uv=True)
    S = S[:,:task.config.target_rank]
    # print(S)
    # print(torch.maximum(S, torch.ones_like(S)))
    # print(S - torch.maximum(S, torch.ones_like(S)))
    loss_rank = torch.mean((torch.maximum(S, task.config.target_sval*torch.ones_like(S) - S)))
    loss_rank = task.config.rank_lambda * loss_rank

    loss = loss_prediction + loss_activity + loss_weight + loss_rank
    print(f'Loss: {loss.item():.4f} | Prediction: {loss_prediction.item():.4f} | Activity: {loss_activity.item():.4f} | Weight: {loss_weight.item():.4f} | Rank: {loss_rank.item():.4f}')

    return loss, outputs



SD_0D_EC_highrank_TASK = o2s.task.Task('SD-0D_EC_highrank',
                    task_specific_params=default_params,
                    get_vars_func=template_0D.get_vars, 
                    create_data_func=create_data,
                    init_func=template_EC.init_func,
                    input_map=input_map,
                    target_map=target_map,
                    test_func=o2s.test.test_tuning,
                    loss_func=loss_func,
                    test_func_args=dict(tuning_vars_list=['ego_SD', 'allo_SD', 'AV', 'HD']),
                    get_subtask_vars_funcs={'joint': template_0D.get_joint_vars,
                                            'hd_iso': template_0D.get_hd_iso_vars,
                                            'sd_iso': template_0D.get_sd_iso_vars,
                                            'av': template_0D.get_av_vars,
                                            'metric': template_0D.get_metric_vars})