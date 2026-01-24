import torch

import sys

import o2s

import o2s.Templates.vars_2D as template_2D
import o2s.Templates.vars_EC as template_EC


target_map = {
    'sin_sd': 0,
    'cos_sd': 1
}

default_params = {
    **template_2D.default_params,
    **template_EC.default_params
}


def create_data(config, vars, inputs, targets, mask):

    inputs, mask = template_EC.fill_inputs(config, vars, inputs, mask)

    targets[:,:,target_map['sin_sd']] = torch.sin(vars['sd'])
    targets[:,:,target_map['cos_sd']] = torch.cos(vars['sd'])

    return inputs, targets, mask



SD_2D_EC_TASK = o2s.task.Task('SD-2D_EC',
                    task_specific_params=default_params, 
                    get_vars_func=template_2D.get_vars,
                    create_data_func=create_data,
                    init_func=template_EC.init_func,
                    input_map={},
                    target_map=target_map,
                    test_func=o2s.test.test_tuning,
                    test_func_args=dict(tuning_vars_list=['ego_SD', 'allo_SD', 'AV', 'x']))



