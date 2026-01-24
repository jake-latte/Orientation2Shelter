import torch
import sys
import o2s
import o2s.Templates.vars_1D_vecvel as template_1D_vecvel

class SD_1D_vecvel(template_1D_vecvel.Vars1DVecvel):
    task_name = "SD-1D_vecvel"
    target_map = {
        'sin_sd': 0,
        'cos_sd': 1
    }
    default_params = template_1D_vecvel.default_params
    input_map = template_1D_vecvel.input_map
    get_vars = staticmethod(template_1D_vecvel.get_vars)
    get_joint_vars = staticmethod(template_1D_vecvel.get_joint_vars)
    get_hd_iso_vars = staticmethod(template_1D_vecvel.get_hd_iso_vars)
    get_sd_iso_vars = staticmethod(template_1D_vecvel.get_sd_iso_vars)
    get_hd_iso_vel_vars = staticmethod(template_1D_vecvel.get_hd_iso_vel_vars)
    get_av_vars = staticmethod(template_1D_vecvel.get_av_vars)
    get_metric_vars = staticmethod(template_1D_vecvel.get_metric_vars)
    test_func = o2s.test.test_tuning
    test_func_args = dict(tuning_vars_list=['HD', 'ego_SD', 'allo_SD', 'AV', 'x'])
    @staticmethod
    def create_data(config, vars, inputs, targets, mask):
        
        inputs, mask = template_1D_vecvel.fill_inputs(config, vars, inputs, mask)
    
        targets[:,:,SD_1D_vecvel.target_map['sin_sd']] = torch.sin(vars['sd'])
        targets[:,:,SD_1D_vecvel.target_map['cos_sd']] = torch.cos(vars['sd'])
    
        return inputs, targets, mask
    def __init__(self, **kwargs):
        self.get_vars_func = self.get_vars
        self.get_subtask_vars_funcs = {
            "joint": self.get_joint_vars,
            "hd_iso": self.get_hd_iso_vars,
            "sd_iso": self.get_sd_iso_vars,
            "hd_iso_vel": self.get_hd_iso_vel_vars,
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
#     'sin_sd': 0,
#     'cos_sd': 1
# }
#
# def create_data(config, vars, inputs, targets, mask):
#
#     inputs, mask = template_1D_vecvel.fill_inputs(config, vars, inputs, mask)
#
#     targets[:,:,target_map['sin_sd']] = torch.sin(vars['sd'])
#     targets[:,:,target_map['cos_sd']] = torch.cos(vars['sd'])
#
#     return inputs, targets, mask
#
#
#
# SD_1D_vecvel_TASK = o2s.task.Task('SD-1D_vecvel',
#                     task_specific_params=template_1D_vecvel.default_params,
#                     get_vars_func=template_1D_vecvel.get_vars, 
#                     create_data_func=create_data,
#                     input_map=template_1D_vecvel.input_map,
#                     target_map=target_map,
#                     test_func=o2s.test.test_tuning,
#                     test_func_args=dict(tuning_vars_list=['HD', 'ego_SD', 'allo_SD', 'AV', 'x']),
#                     get_subtask_vars_funcs={'joint': template_1D_vecvel.get_joint_vars,
#                                             'hd_iso': template_1D_vecvel.get_hd_iso_vars,
#                                             'sd_iso': template_1D_vecvel.get_sd_iso_vars,
#                                             'hd_iso_vel': template_1D_vecvel.get_hd_iso_vel_vars,
#                                             'av': template_1D_vecvel.get_av_vars,
#                                             'metric': template_1D_vecvel.get_metric_vars})
#
#
