import torch

import sys

import o2s


import o2s.Templates.vars_2D_linvel as template_2D_linvel

target_map = {
    'x': 0,
    'y': 1,
    'sin_hd': 2,
    'cos_hd': 3,
    'sin_sd': 4,
    'cos_sd': 5
}

def create_data(config, vars, inputs, targets, mask):

    inputs, mask = template_2D_linvel.fill_inputs(config, vars, inputs, mask)

    targets[:,:,target_map['x']] = vars['x']
    targets[:,:,target_map['y']] = vars['y']
    targets[:,:,target_map['sin_hd']] = torch.sin(vars['hd'])
    targets[:,:,target_map['cos_hd']] = torch.cos(vars['hd'])
    targets[:,:,target_map['sin_sd']] = torch.sin(vars['sd'])
    targets[:,:,target_map['cos_sd']] = torch.cos(vars['sd'])

    return inputs, targets, mask



PI_HD_SD_2D_linvel_TASK = o2s.task.Task('PI_HD_SD-2D_linvel',
                            task_specific_params=template_2D_linvel.default_params,
                            get_vars_func=template_2D_linvel.get_vars, 
                            create_data_func=create_data,
                            input_map=template_2D_linvel.input_map,
                            target_map=target_map,
                            test_func=o2s.test.test_tuning,
                            test_func_args=dict(tuning_vars_list=['HD', 'ego_SD', 'allo_SD', 'AV', 'x', 'y']))



