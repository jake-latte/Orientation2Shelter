import torch
import numpy as np
import sys

import o2s

# Task Dependencies
import o2s.Templates.vars_0D as template_0D

default_params = {
    **template_0D.default_params,
    'min_swap_time': 100,
    'max_swap_time': 400,
    'loss_period_len': 50
}

input_map = {
    'av': 0,
    'sin_hd_0': 1,
    'cos_hd_0': 2,
    'sx': 3,
    'sy': 4,
    'output_angle_is_hd': 5
}

target_map = {
    'sin_angle': 0,
    'cos_angle': 1
}

def get_vars(config):
    vars = template_0D.get_vars(config)

    vars['swap_time'] = torch.randint(config.min_swap_time, config.max_swap_time, (config.batch_size,1))

    return vars

def create_data(config, vars, inputs, targets, mask):
    
    inputs, mask = template_0D.fill_inputs(config, vars,inputs, mask)

    swap_time = vars['swap_time']
    output_is_hd = torch.ones((config.batch_size, config.n_timesteps), dtype=torch.bool)
    for i in range(config.batch_size):
        output_is_hd[i,swap_time[i]:] = False

    inputs[:,:,input_map['output_angle_is_hd']] = output_is_hd.int()


    targets[:,:,target_map['sin_angle']][output_is_hd] = torch.sin(vars['hd'])[output_is_hd]
    targets[:,:,target_map['cos_angle']][output_is_hd] = torch.cos(vars['hd'])[output_is_hd]

    ad = torch.remainder(vars['hd'] + vars['sd'], 2*np.pi)
    targets[:,:,target_map['sin_angle']][~output_is_hd] = torch.sin(vars['sd'])[~output_is_hd]
    targets[:,:,target_map['cos_angle']][~output_is_hd] = torch.cos(vars['sd'])[~output_is_hd]

    return inputs, targets, mask

def get_joint_vars(config):
    vars = template_0D.get_joint_vars(config)
    vars['swap_time'] = torch.ones((config.batch_size,1)).long() * (config.n_timesteps)//2
    return vars

def get_joint_unswapped_vars(config):
    vars = template_0D.get_joint_vars(config)
    vars['swap_time'] = torch.ones((config.batch_size,1)).long() * config.n_timesteps
    return vars

def get_hd_iso_vars(config):
    vars = template_0D.get_hd_iso_vars(config)
    vars['swap_time'] = torch.ones((config.batch_size,1)).long() * (config.n_timesteps)//2
    return vars

def get_sd_iso_vars(config):
    vars = template_0D.get_sd_iso_vars(config)
    vars['swap_time'] = torch.ones((config.batch_size,1)).long() * (config.n_timesteps)//2
    return vars

def get_av_vars(config):
    vars = template_0D.get_av_vars(config)
    vars['swap_time'] = torch.ones((config.batch_size,1)).long() * (config.n_timesteps)//2
    return vars

def get_metric_vars(config):
    vars = template_0D.get_metric_vars(config)
    vars['swap_time'] = torch.ones((config.batch_size,1)).long() * (config.n_timesteps)//2
    return vars

HD_SD_0D_PARAM_TASK = o2s.task.Task('HD_SD-0D-param',
                            task_specific_params=default_params, 
                            get_vars_func=get_vars,
                            create_data_func=create_data,
                            input_map=input_map,
                            target_map=target_map,
                            test_func=o2s.test.test_tuning,
                            test_func_args=dict(tuning_vars_list=['HD', 'ego_SD', 'allo_SD', 'AV']),
                            get_subtask_vars_funcs={'joint': get_joint_vars,
                                                    'joint_unswapped': get_joint_unswapped_vars,
                                                    'hd_iso': get_hd_iso_vars,
                                                    'sd_iso': get_sd_iso_vars,
                                                    'av': get_av_vars,
                                                    'metric': get_metric_vars})







