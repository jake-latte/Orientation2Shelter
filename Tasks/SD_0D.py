import torch

import sys

from task import *
from build import *
from test_funcs import *

target_map = {
    'sin_sd': 0,
    'cos_sd': 1
}

def create_data(config, inputs, targets, mask):
    
    vars = Tasks.vars_0D.create_data(config, for_training=(inputs.shape[0] == config.batch_size and inputs.shape[1] == config.n_timesteps))
    inputs, mask = Tasks.vars_0D.fill_inputs(config, inputs, mask, vars)

    targets[:,:,target_map['sin_sd']] = torch.sin(vars['sd'])
    targets[:,:,target_map['cos_sd']] = torch.cos(vars['sd'])

    return inputs, targets, vars, mask



SD_0D_TASK = Task('SD-0D', 
                    n_inputs=5, n_outputs=2, 
                    task_specific_params=Tasks.vars_0D.default_params, 
                    create_data_func=create_data,
                    input_map=Tasks.vars_0D.input_map,
                    target_map=target_map,
                    test_func=test_tuning,
                    test_func_args=dict(tuning_vars_list=['ego_SD', 'allo_SD', 'AV']))



