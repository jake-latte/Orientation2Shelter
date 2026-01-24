import torch

import sys
from typing import Dict

import o2s
import o2s.Templates.vars_0D as template_0D
import o2s.Templates.vars_EC as template_EC

default_params = {
    **template_0D.default_params,
    'n_position_place_cells': 0,
    'n_shelter_place_cells': 25,
    'n_head_direction_cells': 25,
    'head_direction_cell_scale': template_EC.default_params['head_direction_cell_scale'],
    'place_cell_scale': template_EC.default_params['place_cell_scale'],
    'delay': 10
}


def init_func(task):
    config = task.config

    targets_initialised = True
    for i in range(task.config.delay):
        if not (f'sin_sd_{i}' and f'cos_sd_{i}' in task.target_map):
            targets_initialised = False
            break
    if not targets_initialised:
        for i in range(task.config.delay):
            task.target_map[f'sin_sd_{i}'] = task.config.n_outputs + 2*i
            task.target_map[f'cos_sd_{i}'] = task.config.n_outputs + 2*i + 1
        task.config.n_outputs += 2*config.delay

    inputs_initialised = True
    for i in range(task.config.delay):
        if not (f'av_{i}' in task.input_map):
            inputs_initialised = False
            break
    if not inputs_initialised:
        for i in range(task.config.delay):
            task.input_map[f'av_{i}'] = i
        task.config.n_inputs += task.config.delay
    
    template_EC.init_func(task)

def fill_head_direction_cell_inputs(config: o2s.config.Config, vars: Dict[str, torch.Tensor], inputs: torch.Tensor) -> torch.Tensor:
    init_duration, batch_size = config.init_duration, inputs.shape[0]

    head_direction = vars['hd']

    inputs[:,:,-config.n_head_direction_cells:] = template_EC.HD_activity(
        Theta=head_direction, 
        mu_h=config.head_direction_cell_centers.to(head_direction.device), scale=config.head_direction_cell_scale) 
    
    return inputs

def create_data(config, vars, inputs, targets, mask):
    batch_size, n_timesteps = inputs.shape[0], inputs.shape[1]
    init_duration = config.init_duration
    
    inputs = template_EC.fill_place_cell_inputs(config, vars, inputs)
    inputs = fill_head_direction_cell_inputs(config, vars, inputs)

    for i in range(config.delay):
        if i==0:
            inputs[:,:,i] = vars['av']
        else:
            inputs[:,:-i,i] = vars['av'][:,i:]

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



SD_0D_EC_pred_TASK = o2s.task.Task('SD-0D-EC_pred',
                    task_specific_params=default_params, 
                    get_vars_func=template_0D.get_vars,
                    create_data_func=create_data,
                    input_map={},
                    target_map={},
                    init_func=init_func,
                    test_func=o2s.test.test_tuning,
                    test_func_args=dict(tuning_vars_list=['HD', 'ego_SD', 'allo_SD', 'AV']),
                    get_subtask_vars_funcs={'joint': template_0D.get_joint_vars,
                                            'hd_iso': template_0D.get_hd_iso_vars,
                                            'sd_iso': template_0D.get_sd_iso_vars,
                                            'av': template_0D.get_av_vars,
                                            'metric': template_0D.get_metric_vars})



