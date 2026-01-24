import torch

import sys

import o2s

import o2s.Templates.vars_0D as template_0D
import o2s.Templates.vars_EC as template_EC
import o2s.Templates.vars_highdim as template_highdim

input_map = {
    'av': 0
}

target_map = {
    'sin_sd': 0,
    'cos_sd': 1,
}

default_params = {
    **template_0D.default_params,
    'n_position_place_cells': 0,
    'n_shelter_place_cells': 25,
    'n_head_direction_cells': 25,
    'head_direction_cell_scale': template_EC.default_params['head_direction_cell_scale'],
    'place_cell_scale': template_EC.default_params['place_cell_scale'],
    'n_transformed_outputs': 10,
    'transformer_seed': 0
}

def init_func(task):
    template_EC.init_func(task)
    template_highdim.init_func(task)


def create_data(config, vars, inputs, targets, mask):

    inputs, mask = template_EC.fill_inputs(config, vars, inputs, mask)

    inputs[:,:,input_map['av']] = vars['av']

    inter_targets = torch.stack((torch.sin(vars['sd']), torch.cos(vars['sd'])), dim=2)
    targets = torch.matmul(inter_targets, config.transformer.T).to(targets.device)

    return inputs, targets, mask



SD_0D_EC_HIGHDIM_TASK = o2s.task.Task('SD-0D-EC_highdim',
                    task_specific_params=default_params,
                    get_vars_func=template_0D.get_vars, 
                    create_data_func=create_data,
                    init_func=init_func,
                    input_map=input_map,
                    target_map={},
                    test_func=o2s.test.test_tuning,
                    test_func_args=dict(tuning_vars_list=['ego_SD', 'allo_SD', 'AV', 'HD']),
                    get_subtask_vars_funcs={'joint': template_0D.get_joint_vars,
                                            'hd_iso': template_0D.get_hd_iso_vars,
                                            'sd_iso': template_0D.get_sd_iso_vars,
                                            'av': template_0D.get_av_vars,
                                            'metric': template_0D.get_metric_vars})



