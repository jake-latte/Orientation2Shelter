import torch

import sys

from task import *
from build import *
from test_funcs import *

import Tasks.vars_1D as template_1D

target_map = {
    'x': 0
}

def create_data(config, inputs, targets, mask):
    
    vars = template_1D.create_data(config, for_training=(inputs.shape[0] == config.batch_size and inputs.shape[1] == config.n_timesteps))
    inputs, mask = template_1D.fill_inputs(config, inputs, mask, vars)

    targets[:,:,target_map['x']] = vars['x']

    return inputs, targets, vars, mask



PI_1D_TASK = Task('PI-1D', 
                    n_inputs=7, n_outputs=1, 
                    task_specific_params=template_1D.default_params, 
                    create_data_func=create_data,
                    input_map=template_1D.input_map,
                    target_map=target_map,
                    test_func=test_tuning,
                    test_func_args=dict(tuning_vars_list=['HD', 'AV', 'x']))



