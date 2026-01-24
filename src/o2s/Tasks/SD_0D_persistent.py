import torch

import sys

import o2s

# Task Dependencies
import o2s.Templates.vars_0D as template_0D

class SD_0D_persistent(template_0D.Vars0D):
    task_name = "SD-0D_persistent"
    target_map = {
        'sin_sd': 0,
        'cos_sd': 1,
    }
    task_specific_params = {
        **template_0D.default_params,
    }
    test_func = o2s.test.test_gamut
    test_func_args = dict(tuning_vars_list=['HD', 'ego_SD', 'allo_SD', 'AV'])

    @staticmethod
    def create_data(config, vars, inputs, targets, mask):
        batch_size, n_timesteps = inputs.shape[0], inputs.shape[1]
        init_duration = config.init_duration
        
        inputs[:,:,template_0D.input_map['av']] = vars['av']
        inputs[:,:,template_0D.input_map['sin_hd_0']] = torch.sin(vars['hd'][:,0])
        inputs[:,:,template_0D.input_map['cos_hd_0']] = torch.cos(vars['hd'][:,0])
        inputs[:,:,template_0D.input_map['sx']] = vars['sx'].reshape((batch_size,1)).repeat((1,n_timesteps))
        inputs[:,:,template_0D.input_map['sy']] = vars['sy'].reshape((batch_size,1)).repeat((1,n_timesteps))

        mask[:,:init_duration] = False

        targets[:,:,SD_0D_persistent.target_map['sin_sd']] = torch.sin(vars['sd'])
        targets[:,:,SD_0D_persistent.target_map['cos_sd']] = torch.cos(vars['sd'])

        return inputs, targets, mask

    def __init__(self, **kwargs):
        self.get_vars_func = template_0D.get_vars
        self.get_subtask_vars_funcs = {
            "joint": template_0D.get_joint_vars,
            "hd_iso": template_0D.get_hd_iso_vars,
            "sd_iso": template_0D.get_sd_iso_vars,
            "av": template_0D.get_av_vars,
            "metric": template_0D.get_metric_vars,
        }
        super().__init__(
            name=self.task_name,
            task_specific_params=self.task_specific_params,
            get_vars_func=self.get_vars_func,
            create_data_func=self.create_data,
            input_map=template_0D.input_map,
            target_map=self.target_map,
            test_func=self.test_func,
            test_func_args=self.test_func_args,
            get_subtask_vars_funcs=self.get_subtask_vars_funcs,
            **kwargs
        )

# target_map = {
#     'sin_sd': 0,
#     'cos_sd': 1,
# }
#
# task_specific_params = {
#     **template_0D.default_params,
#     
# }
#
# def create_data(config, vars, inputs, targets, mask):
#     batch_size, n_timesteps = inputs.shape[0], inputs.shape[1]
#     init_duration = config.init_duration
#     
#     inputs[:,:,template_0D.input_map['av']] = vars['av']
#     inputs[:,:,template_0D.input_map['sin_hd_0']] = torch.sin(vars['hd'][:,0]).reshape((batch_size,1)).repeat((1,init_duration))
#     inputs[:,:,template_0D.input_map['cos_hd_0']] = torch.cos(vars['hd'][:,0]).reshape((batch_size,1)).repeat((1,init_duration))
#     inputs[:,:,template_0D.input_map['sx']] = vars['sx'].reshape((batch_size,1)).repeat((1,init_duration))
#     inputs[:,:,template_0D.input_map['sy']] = vars['sy'].reshape((batch_size,1)).repeat((1,init_duration))
#
#     mask[:,:init_duration] = False
#
#     targets[:,:,target_map['sin_sd']] = torch.sin(vars['sd'])
#     targets[:,:,target_map['cos_sd']] = torch.cos(vars['sd'])
#
#     return inputs, targets, mask
#
#
#
# SD_0D_PERSISTENT_TASK = o2s.task.Task('SD-0D_persistent',
#                     task_specific_params=template_0D.default_params, 
#                     get_vars_func=template_0D.get_vars,
#                     create_data_func=create_data,
#                     input_map=template_0D.input_map,
#                     target_map=target_map,
#                     test_func=o2s.test.test_gamut,
#                     test_func_args=dict(tuning_vars_list=['HD', 'ego_SD', 'allo_SD', 'AV']),
#                     get_subtask_vars_funcs={'joint': template_0D.get_joint_vars,
#                                             'hd_iso': template_0D.get_hd_iso_vars,
#                                             'sd_iso': template_0D.get_sd_iso_vars,
#                                             'av': template_0D.get_av_vars,
#                                             'metric': template_0D.get_metric_vars})


