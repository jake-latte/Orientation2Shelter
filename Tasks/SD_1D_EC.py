import torch

import sys

from task import *
from build import *
from test_funcs import *

import Tasks.vars_1D_linvel as template_1D
import Tasks.vars_EC as template_EC


target_map = {
    'sin_sd': 0,
    'cos_sd': 1
}

default_params = {
    **template_1D.default_params,
    **template_EC.default_params
}


def create_data(config, inputs, targets, mask):

    vars = template_1D.create_data(config, for_training=(inputs.shape[0] == config.batch_size and inputs.shape[1] == config.n_timesteps))
    inputs, mask = template_EC.fill_inputs(config, inputs, mask, vars)

    targets[:,:,target_map['sin_sd']] = torch.sin(vars['sd'])
    targets[:,:,target_map['cos_sd']] = torch.cos(vars['sd'])

    return inputs, targets, vars, mask



SD_1D_EC_TASK = Task('SD-1D_EC',
                    task_specific_params=default_params, 
                    create_data_func=create_data,
                    init_func=template_EC.init_func,
                    input_map={},
                    target_map=target_map,
                    test_func=test_tuning,
                    test_func_args=dict(tuning_vars_list=['ego_SD', 'allo_SD', 'AV', 'x']))



