import torch

import sys

import o2s

# Task Dependencies
import o2s.Templates.vars_1D_vecvel as template_1D_vecvel

target_map = {
    'sin_hd': 0,
    'cos_hd': 1,
    'sin_sd': 2,
    'cos_sd': 3
}

def create_data(config, vars, inputs, targets, mask):
    
    inputs, mask = template_1D_vecvel.fill_inputs(config, vars, inputs, mask)

    targets[:,:,target_map['sin_hd']] = torch.sin(vars['hd'])
    targets[:,:,target_map['cos_hd']] = torch.cos(vars['hd'])
    targets[:,:,target_map['sin_sd']] = torch.sin(vars['sd'])
    targets[:,:,target_map['cos_sd']] = torch.cos(vars['sd'])

    return inputs, targets, mask



HD_SD_1D_vecvel_TASK = o2s.task.Task('HD_SD-1D_vecvel',
                    task_specific_params=template_1D_vecvel.default_params,
                    get_vars_func=template_1D_vecvel.get_vars, 
                    create_data_func=create_data,
                    input_map=template_1D_vecvel.input_map,
                    target_map=target_map,
                    test_func=o2s.test.test_tuning,
                    test_func_args=dict(tuning_vars_list=['HD', 'ego_SD', 'allo_SD', 'AV', 'x']),
                    get_subtask_vars_funcs={'joint': template_1D_vecvel.get_joint_vars,
                                            'hd_iso': template_1D_vecvel.get_hd_iso_vars,
                                            'sd_iso': template_1D_vecvel.get_sd_iso_vars,
                                            'hd_iso_vel': template_1D_vecvel.get_hd_iso_vel_vars,
                                            'av': template_1D_vecvel.get_av_vars,
                                            'metric': template_1D_vecvel.get_metric_vars})



