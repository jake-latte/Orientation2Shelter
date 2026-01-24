import torch

import sys

import o2s

# Task Dependencies
import o2s.Templates.vars_0D as template_0D

default_params = {
    **template_0D.default_params,
    'delay': 10
}

input_map = {
    'sx': 0,
    'sy': 1,
    'sin_hd': 2,
    'cos_hd': 3,
}

def init_func(task):
    config = task.config
    if len(task.target_map) == 0:
        task.config.n_outputs = 2 * config.delay
        for i in range(task.config.delay):
            task.target_map[f'sin_sd_{i}'] = 2*i
            task.target_map[f'cos_sd_{i}'] = 2*i + 1

    if len(task.input_map) == 4:
        task.config.n_inputs = 4 + task.config.delay
        for i in range(task.config.delay):
            task.input_map[f'av_{i}'] = i + 4


def create_data(config, vars, inputs, targets, mask):
    batch_size, n_timesteps = inputs.shape[0], inputs.shape[1]
    init_duration = config.init_duration
    
    inputs[:,:init_duration,input_map['sx']] = vars['sx'].reshape((batch_size,1)).repeat((1,init_duration))
    inputs[:,:init_duration,input_map['sy']] = vars['sy'].reshape((batch_size,1)).repeat((1,init_duration))
    
    inputs[:,:,input_map['sin_hd']] = torch.sin(vars['hd'])
    inputs[:,:,input_map['cos_hd']] = torch.cos(vars['hd'])

    for i in range(config.delay):
        if i==0:
            inputs[:,:,input_map[f'av_{i}']] = vars['av']
        else:
            inputs[:,:-i,input_map[f'av_{i}']] = vars['av'][:,i:]

    for i in range(config.delay):
        if i==0:
            targets[:,:,2*i] = torch.sin(vars['sd'])
            targets[:,:,2*i + 1] = torch.cos(vars['sd'])
        else:
            targets[:,:-i,2*i] = torch.sin(vars['sd'][:,i:])
            targets[:,:-i,2*i + 1] = torch.cos(vars['sd'][:,i:])
            mask[:,-i:,2*i] = False
            mask[:,-i:,2*i + 1] = False

    mask[:,:init_duration] = False

    return inputs, targets, mask



SD_0D_pred_TASK = o2s.task.Task('SD-0D_pred',
                    task_specific_params=default_params, 
                    get_vars_func=template_0D.get_vars,
                    create_data_func=create_data,
                    input_map=input_map,
                    target_map={},
                    init_func=init_func,
                    test_func=o2s.test.test_tuning,
                    test_func_args=dict(tuning_vars_list=['HD', 'ego_SD', 'allo_SD', 'AV']),
                    get_subtask_vars_funcs={'joint': template_0D.get_joint_vars,
                                            'hd_iso': template_0D.get_hd_iso_vars,
                                            'sd_iso': template_0D.get_sd_iso_vars,
                                            'av': template_0D.get_av_vars,
                                            'metric': template_0D.get_metric_vars})



