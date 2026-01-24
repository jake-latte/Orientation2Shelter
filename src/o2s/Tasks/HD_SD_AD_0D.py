import torch
import sys
import o2s
import o2s.Templates.vars_0D as template_0D

class HD_SD_AD_0D(template_0D.Vars0D):
    task_name = "HD_SD_AD-0D"
    target_map = {
        'sin_hd': 0,
        'cos_hd': 1,
        'sin_sd': 2,
        'cos_sd': 3,
        'sin_ad': 4,
        'cos_ad': 5
    }
    default_params = template_0D.default_params
    input_map = template_0D.input_map
    get_vars = staticmethod(template_0D.get_vars)
    get_joint_vars = staticmethod(template_0D.get_joint_vars)
    get_hd_iso_vars = staticmethod(template_0D.get_hd_iso_vars)
    get_sd_iso_vars = staticmethod(template_0D.get_sd_iso_vars)
    get_av_vars = staticmethod(template_0D.get_av_vars)
    get_metric_vars = staticmethod(template_0D.get_metric_vars)
    test_func = o2s.test.test_tuning
    test_func_args = dict(tuning_vars_list=['HD', 'ego_SD', 'allo_SD', 'AV'])
    @staticmethod
    def create_data(config, vars, inputs, targets, mask):
        
        inputs, mask = template_0D.fill_inputs(config, vars,inputs, mask)
    
        targets[:,:,HD_SD_AD_0D.target_map['sin_hd']] = torch.sin(vars['hd'])
        targets[:,:,HD_SD_AD_0D.target_map['cos_hd']] = torch.cos(vars['hd'])
        targets[:,:,HD_SD_AD_0D.target_map['sin_sd']] = torch.sin(vars['sd'])
        targets[:,:,HD_SD_AD_0D.target_map['cos_sd']] = torch.cos(vars['sd'])
        targets[:,:,HD_SD_AD_0D.target_map['sin_ad']] = vars['sy'].reshape((-1,1)).repeat(1,config.n_timesteps)
        targets[:,:,HD_SD_AD_0D.target_map['cos_ad']] = vars['sx'].reshape((-1,1)).repeat(1,config.n_timesteps)
    
        return inputs, targets, mask
    def __init__(self, **kwargs):
        self.get_vars_func = self.get_vars
        self.get_subtask_vars_funcs = {
            "joint": self.get_joint_vars,
            "hd_iso": self.get_hd_iso_vars,
            "sd_iso": self.get_sd_iso_vars,
            "av": self.get_av_vars,
            "metric": self.get_metric_vars,
        }
        super().__init__(
            name=self.task_name,
            task_specific_params=self.default_params,
            get_vars_func=self.get_vars_func,
            create_data_func=self.create_data,
            input_map=self.input_map,
            target_map=self.target_map,
            test_func=self.test_func,
            test_func_args=self.test_func_args,
            get_subtask_vars_funcs=self.get_subtask_vars_funcs,
            **kwargs
        )

# target_map = {
#     'sin_hd': 0,
#     'cos_hd': 1,
#     'sin_sd': 2,
#     'cos_sd': 3,
#     'sin_ad': 4,
#     'cos_ad': 5
# }
#
# def create_data(config, vars, inputs, targets, mask):
#
#     inputs, mask = template_0D.fill_inputs(config, vars,inputs, mask)
#
#     targets[:,:,target_map['sin_hd']] = torch.sin(vars['hd'])
#     targets[:,:,target_map['cos_hd']] = torch.cos(vars['hd'])
#     targets[:,:,target_map['sin_sd']] = torch.sin(vars['sd'])
#     targets[:,:,target_map['cos_sd']] = torch.cos(vars['sd'])
#     targets[:,:,target_map['sin_ad']] = vars['sy'].reshape((-1,1)).repeat(1,config.n_timesteps)
#     targets[:,:,target_map['cos_ad']] = vars['sx'].reshape((-1,1)).repeat(1,config.n_timesteps)
#
#     return inputs, targets, mask
#
#
#
# HD_SD_AD_0D_TASK = o2s.task.Task('HD_SD_AD-0D',
#                     task_specific_params=template_0D.default_params, 
#                     get_vars_func=template_0D.get_vars,
#                     create_data_func=create_data,
#                     input_map=template_0D.input_map,
#                     target_map=target_map,
#                     test_func=o2s.test.test_tuning,
#                     test_func_args=dict(tuning_vars_list=['HD', 'ego_SD', 'allo_SD', 'AV']),
#                     get_subtask_vars_funcs={'joint': template_0D.get_joint_vars,
#                                             'hd_iso': template_0D.get_hd_iso_vars,
#                                             'sd_iso': template_0D.get_sd_iso_vars,
#                                             'av': template_0D.get_av_vars,
#                                             'metric': template_0D.get_metric_vars})
#
#
