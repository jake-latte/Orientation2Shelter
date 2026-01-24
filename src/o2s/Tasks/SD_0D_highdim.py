import torch

import sys

import o2s

# Task Dependencies
import o2s.Templates.vars_0D as template_0D
import o2s.Templates.vars_highdim as template_highdim

default_params = {
    **template_0D.default_params,
    'n_transformed_outputs': 10,
    'transformer_seed': 0
}


def create_data(config, vars, inputs, targets, mask):
    
    inputs, mask = template_0D.fill_inputs(config, vars,inputs, mask)

    inter_targets = torch.stack((torch.sin(vars['sd']), torch.cos(vars['sd'])), dim=2)
    targets = torch.matmul(inter_targets, config.transformer.T).to(targets.device)

    return inputs, targets, mask



SD_0D_HIGHDIM_TASK = o2s.task.Task('SD-0D-highdim',
                    task_specific_params=default_params, 
                    init_func=template_highdim.init_func,
                    get_vars_func=template_0D.get_vars,
                    create_data_func=create_data,
                    input_map=template_0D.input_map,
                    target_map={},
                    test_func=o2s.test.test_tuning,
                    test_func_args=dict(tuning_vars_list=['HD', 'ego_SD', 'allo_SD', 'AV']),
                    get_subtask_vars_funcs={'joint': template_0D.get_joint_vars,
                                            'hd_iso': template_0D.get_hd_iso_vars,
                                            'sd_iso': template_0D.get_sd_iso_vars,
                                            'av': template_0D.get_av_vars,
                                            'metric': template_0D.get_metric_vars})



