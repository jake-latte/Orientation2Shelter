import torch

import sys

import o2s

# Task Dependencies
import o2s.Templates.vars_1D as template_1D

target_map = {
    'sin_hd': 0,
    'cos_hd': 1
}


def create_data(config, vars, inputs, targets, mask):
    
    inputs, mask = template_1D.fill_inputs(config, vars, inputs, mask)

    targets[:,:,target_map['sin_hd']] = torch.sin(vars['hd'])
    targets[:,:,target_map['cos_hd']] = torch.cos(vars['hd'])

    return inputs, targets, mask



HD_1D_TASK = o2s.task.Task('HD-1D',
                    task_specific_params=template_1D.default_params, 
                    get_vars_func=template_1D.get_vars,
                    create_data_func=create_data,
                    input_map=template_1D.input_map,
                    target_map=target_map,
                    test_func=o2s.test.test_tuning,
                    test_func_args=dict(tuning_vars_list=['HD', 'AV', 'x']))



