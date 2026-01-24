import torch

import sys

import o2s

import o2s.Templates.vars_2D as template_2D

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

def create_data(config, vars, inputs, targets, mask):

    inputs, mask = template_2D.fill_inputs(config, vars, inputs, mask)

    targets[:,:,target_map['x']] = vars['x']
    targets[:,:,target_map['y']] = vars['y']

    return inputs, targets, mask



PI_2D_TASK = o2s.task.Task('PI-2D',
                    task_specific_params=cueva_params, 
                    get_vars_func=template_2D.get_vars,
                    create_data_func=create_data,
                    input_map=template_2D.input_map,
                    target_map=target_map,
                    test_func=o2s.test.test_tuning,
                    test_func_args=dict(tuning_vars_list=['HD', 'AV', 'x', 'y']),
                    loss_func=cueva_loss_func,
                    get_subtask_vars_funcs={'joint': template_2D.get_joint_vars,
                                            'hd_iso': template_2D.get_hd_iso_vars,
                                            'sd_iso': template_2D.get_sd_iso_vars,
                                            'av': template_2D.get_av_vars,
                                            'metric': template_2D.get_metric_vars})



