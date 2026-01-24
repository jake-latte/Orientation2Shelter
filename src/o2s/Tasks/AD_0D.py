import torch

import sys

import o2s
# import o2s.Templates.vars_0D as template_0D
template_0D = o2s.Templates.vars_0D

target_map = {
    'sin_ad': 0,
    'cos_ad': 1
}

def create_data(config, vars, inputs, targets, mask):
    
    inputs, mask = template_0D.fill_inputs(config, vars,inputs, mask)

    targets[:,:,target_map['sin_ad']] = vars['sy'].reshape((-1,1)).repeat(1,config.n_timesteps)
    targets[:,:,target_map['cos_ad']] = vars['sx'].reshape((-1,1)).repeat(1,config.n_timesteps)

    return inputs, targets, mask



AD_0D_TASK = o2s.task.Task('AD-0D',
                    task_specific_params=template_0D.default_params, 
                    get_vars_func=template_0D.get_vars,
                    create_data_func=create_data,
                    input_map=template_0D.input_map,
                    target_map=target_map,
                    test_func=o2s.test.test_tuning,
                    test_func_args=dict(tuning_vars_list=['HD', 'ego_SD', 'allo_SD', 'AV']),
                    get_subtask_vars_funcs={'joint': template_0D.get_joint_vars,
                                            'hd_iso': template_0D.get_hd_iso_vars,
                                            'sd_iso': template_0D.get_sd_iso_vars,
                                            'av': template_0D.get_av_vars,
                                            'metric': template_0D.get_metric_vars})



