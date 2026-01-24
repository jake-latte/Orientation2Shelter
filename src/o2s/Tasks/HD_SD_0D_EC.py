import torch
import sys
import o2s
import o2s.Templates.vars_0D as template_0D
import o2s.Templates.vars_EC as template_EC

class HD_SD_0D_EC(template_0D.Vars0D, template_EC.VarsEC):
    task_name = "HD_SD-0D_EC"
    input_map = {
        'av': 0
    }
    target_map = {
        'sin_hd': 0,
        'cos_hd': 1,
        'sin_sd': 2,
        'cos_sd': 3
    }
    default_params = {
        **template_0D.default_params,
        'n_position_place_cells': 0,
        'n_shelter_place_cells': 25,
        'n_head_direction_cells': 25,
        'head_direction_cell_scale': template_EC.default_params['head_direction_cell_scale'],
        'place_cell_scale': template_EC.default_params['place_cell_scale']
    }
    get_vars = staticmethod(template_0D.get_vars)
    get_joint_vars = staticmethod(template_0D.get_joint_vars)
    get_hd_iso_vars = staticmethod(template_0D.get_hd_iso_vars)
    get_sd_iso_vars = staticmethod(template_0D.get_sd_iso_vars)
    get_av_vars = staticmethod(template_0D.get_av_vars)
    get_metric_vars = staticmethod(template_0D.get_metric_vars)
    test_func = o2s.test.test_tuning
    test_func_args = dict(tuning_vars_list=['ego_SD', 'allo_SD', 'AV', 'HD'])
    init_func = template_EC.init_func
    @staticmethod
    def create_data(config, vars, inputs, targets, mask):
    
        inputs, mask = template_EC.fill_inputs(config, vars, inputs, mask)
    
        inputs[:,:,HD_SD_0D_EC.input_map['av']] = vars['av']
    
        targets[:,:,HD_SD_0D_EC.target_map['sin_hd']] = torch.sin(vars['hd'])
        targets[:,:,HD_SD_0D_EC.target_map['cos_hd']] = torch.cos(vars['hd'])
        targets[:,:,HD_SD_0D_EC.target_map['sin_sd']] = torch.sin(vars['sd'])
        targets[:,:,HD_SD_0D_EC.target_map['cos_sd']] = torch.cos(vars['sd'])
    
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
            init_func=self.init_func,
            get_subtask_vars_funcs=self.get_subtask_vars_funcs,
            **kwargs
        )

# input_map = {
#     'av': 0
# }
#
# target_map = {
#     'sin_hd': 0,
#     'cos_hd': 1,
#     'sin_sd': 2,
#     'cos_sd': 3
# }
#
# default_params = {
#     **template_0D.default_params,
#     'n_position_place_cells': 0,
#     'n_shelter_place_cells': 25,
#     'n_head_direction_cells': 25,
#     'head_direction_cell_scale': template_EC.default_params['head_direction_cell_scale'],
#     'place_cell_scale': template_EC.default_params['place_cell_scale']
# }
#
#
# def create_data(config, vars, inputs, targets, mask):
#
#     inputs, mask = template_EC.fill_inputs(config, vars, inputs, mask)
#
#     inputs[:,:,input_map['av']] = vars['av']
#
#     targets[:,:,target_map['sin_hd']] = torch.sin(vars['hd'])
#     targets[:,:,target_map['cos_hd']] = torch.cos(vars['hd'])
#     targets[:,:,target_map['sin_sd']] = torch.sin(vars['sd'])
#     targets[:,:,target_map['cos_sd']] = torch.cos(vars['sd'])
#
#     return inputs, targets, mask
#
#
#
# HD_SD_0D_EC_TASK = o2s.task.Task('HD_SD-0D_EC',
#                     task_specific_params=default_params,
#                     get_vars_func=template_0D.get_vars, 
#                     create_data_func=create_data,
#                     init_func=template_EC.init_func,
#                     input_map=input_map,
#                     target_map=target_map,
#                     test_func=o2s.test.test_tuning,
#                     test_func_args=dict(tuning_vars_list=['ego_SD', 'allo_SD', 'AV', 'HD']),
#                     get_subtask_vars_funcs={'joint': template_0D.get_joint_vars,
#                                             'hd_iso': template_0D.get_hd_iso_vars,
#                                             'sd_iso': template_0D.get_sd_iso_vars,
#                                             'av': template_0D.get_av_vars,
#                                             'metric': template_0D.get_metric_vars})
#
#
