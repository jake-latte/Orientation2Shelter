import torch

import sys

import o2s

# Task Dependencies
import o2s.Templates.vars_0D as template_0D

default_params = {
    **template_0D.default_params,
    'first_loss_time': 50,
    'n_loss_times': 1,
}

target_map = {
    'sin_hd': 0,
    'cos_hd': 1,
    'sin_sd': 2,
    'cos_sd': 3
}

def create_data(config, vars, inputs, targets, mask):
    
    inputs, mask = template_0D.fill_inputs(config, vars,inputs, mask)

    loss_times = torch.linspace(config.n_timesteps-1, config.first_loss_time, config.n_loss_times).long()

    mask[:,:,:] = False
    mask[:,loss_times,:] = True

    targets[:,:,target_map['sin_hd']] = torch.sin(vars['hd'])
    targets[:,:,target_map['cos_hd']] = torch.cos(vars['hd'])
    targets[:,:,target_map['sin_sd']] = torch.sin(vars['sd'])
    targets[:,:,target_map['cos_sd']] = torch.cos(vars['sd'])

    return inputs, targets, mask



HD_SD_0_SPARSELOSS_TASK = o2s.task.Task('HD_SD-0D-sparseloss',
                    task_specific_params=default_params, 
                    get_vars_func=template_0D.get_vars,
                    create_data_func=create_data,
                    input_map=template_0D.input_map,
                    target_map=target_map,
                    test_func=o2s.test.test_tuning,
                    test_func_args=dict(tuning_vars_list=['HD', 'ego_SD', 'allo_SD', 'AV']),
                    get_subtask_vars_funcs={'joint': template_0D.get_joint_vars,
                                            'hd_iso': template_0D.get_hd_iso_vars,
                                            'sd_iso': template_0D.get_sd_iso_vars,
                                            'av': template_0D.get_av_vars,
                                            'metric': template_0D.get_metric_vars})



