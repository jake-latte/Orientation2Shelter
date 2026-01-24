import torch
import sys
import o2s
import o2s.Templates.vars_1D_linvel as template_1D_linvel

class PI_HD_SD_1D_linvel(template_1D_linvel.Vars1DLinvel):
    task_name = "PI_HD_SD-1D_linvel"
    target_map = {
        'sin_x': 0,
        'cos_x': 1,
        'sin_hd': 2,
        'cos_hd': 3,
        'sin_sd': 4,
        'cos_sd': 5
    }
    default_params = template_1D_linvel.default_params
    input_map = template_1D_linvel.input_map
    get_vars = staticmethod(template_1D_linvel.get_vars)
    test_func = o2s.test.test_tuning
    test_func_args = dict(tuning_vars_list=['HD', 'ego_SD', 'allo_SD', 'AV', 'x'])
    @staticmethod
    def create_data(config, vars, inputs, targets, mask):
    
        inputs, mask = template_1D_linvel.fill_inputs(config, vars, inputs, mask)
    
        targets[:,:,PI_HD_SD_1D_linvel.target_map['sin_x']] = torch.sin(vars['x'])
        targets[:,:,PI_HD_SD_1D_linvel.target_map['cos_x']] = torch.cos(vars['x'])
        targets[:,:,PI_HD_SD_1D_linvel.target_map['sin_hd']] = torch.sin(vars['hd'])
        targets[:,:,PI_HD_SD_1D_linvel.target_map['cos_hd']] = torch.cos(vars['hd'])
        targets[:,:,PI_HD_SD_1D_linvel.target_map['sin_sd']] = torch.sin(vars['sd'])
        targets[:,:,PI_HD_SD_1D_linvel.target_map['cos_sd']] = torch.cos(vars['sd'])
    
        return inputs, targets, mask
    def __init__(self, **kwargs):
        self.get_vars_func = self.get_vars
        self.get_subtask_vars_funcs = {}
        super().__init__(
            name=self.task_name,
            task_specific_params=self.default_params,
            get_vars_func=self.get_vars_func,
            create_data_func=self.create_data,
            input_map=self.input_map,
            target_map=self.target_map,
            test_func=self.test_func,
            test_func_args=self.test_func_args,
            **kwargs
        )

# target_map = {
#     'sin_x': 0,
#     'cos_x': 1,
#     'sin_hd': 2,
#     'cos_hd': 3,
#     'sin_sd': 4,
#     'cos_sd': 5
# }
#
# def create_data(config, vars, inputs, targets, mask):
#
#     inputs, mask = template_1D_linvel.fill_inputs(config, vars, inputs, mask)
#
#     targets[:,:,target_map['sin_x']] = torch.sin(vars['x'])
#     targets[:,:,target_map['cos_x']] = torch.cos(vars['x'])
#     targets[:,:,target_map['sin_hd']] = torch.sin(vars['hd'])
#     targets[:,:,target_map['cos_hd']] = torch.cos(vars['hd'])
#     targets[:,:,target_map['sin_sd']] = torch.sin(vars['sd'])
#     targets[:,:,target_map['cos_sd']] = torch.cos(vars['sd'])
#
#     return inputs, targets, mask
#
#
#
# PI_HD_SD_1D_linvel_TASK = o2s.task.Task('PI_HD_SD-1D_linvel',
#                     task_specific_params=template_1D_linvel.default_params, 
#                     get_vars_func=template_1D_linvel.get_vars,
#                     create_data_func=create_data,
#                     input_map=template_1D_linvel.input_map,
#                     target_map=target_map,
#                     test_func=o2s.test.test_tuning,
#                     test_func_args=dict(tuning_vars_list=['HD', 'ego_SD', 'allo_SD', 'AV', 'x']))
#
#
