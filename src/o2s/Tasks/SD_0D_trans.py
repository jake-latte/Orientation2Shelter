import torch

import sys

import o2s


import o2s.Templates.vars_0D as template_0D

input_map = {
    'sin_hd': 0,
    'cos_hd': 1,
    'sx': 2,
    'sy': 3
}

target_map = {
    'sin_sd': 0,
    'cos_sd': 1
}

def create_data(config, vars, inputs, targets, mask):
    batch_size, n_timesteps = config.batch_size, config.n_timesteps

    inputs[:,:,input_map['sin_hd']] = torch.sin(vars['hd'])
    inputs[:,:,input_map['cos_hd']] = torch.cos(vars['hd'])
    inputs[:,:,input_map['sx']] = vars['sx'].reshape((batch_size,1)).repeat((1,n_timesteps))
    inputs[:,:,input_map['sy']] = vars['sy'].reshape((batch_size,1)).repeat((1,n_timesteps))

    targets[:,:,target_map['sin_sd']] = torch.sin(vars['sd'])
    targets[:,:,target_map['cos_sd']] = torch.cos(vars['sd'])

    mask[:,:config.init_duration] = False

    return inputs, targets, mask



SD_0D_TRANS_TASK = o2s.task.Task('SD-0D_trans',
                    task_specific_params=template_0D.default_params, 
                    get_vars_func=template_0D.get_vars,
                    create_data_func=create_data,
                    input_map=input_map,
                    target_map=target_map,
                    test_func=o2s.test.test_tuning,
                    test_func_args=dict(tuning_vars_list=['ego_SD', 'allo_SD', 'HD']))



