import torch
import sys
import o2s
import o2s.Templates.vars_1D_linvel as template_1D_linvel

class SD_1D_linvel(template_1D_linvel.Vars1DLinvel):
    task_name = "SD-1D_linvel"
    target_map = {
        'sin_sd': 0,
        'cos_sd': 1
    }
    default_params = template_1D_linvel.default_params
    input_map = template_1D_linvel.input_map
    get_vars = staticmethod(template_1D_linvel.get_vars)
    test_func = o2s.test.test_tuning
    test_func_args = dict(tuning_vars_list=['HD', 'ego_SD', 'allo_SD', 'AV', 'x'])
    @staticmethod
    def create_data(config, vars, inputs, targets, mask):
        
        inputs, mask = template_1D_linvel.fill_inputs(config, vars, inputs, mask)
    
        targets[:,:,SD_1D_linvel.target_map['sin_sd']] = torch.sin(vars['sd'])
        targets[:,:,SD_1D_linvel.target_map['cos_sd']] = torch.cos(vars['sd'])
    
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
#     'sin_sd': 0,
#     'cos_sd': 1
# }
#
# def create_data(config, vars, inputs, targets, mask):
#
#     inputs, mask = template_1D_linvel.fill_inputs(config, vars, inputs, mask)
#
#     targets[:,:,target_map['sin_sd']] = torch.sin(vars['sd'])
#     targets[:,:,target_map['cos_sd']] = torch.cos(vars['sd'])
#
#     return inputs, targets, mask
#
#
#
# SD_1D_linvel_TASK = o2s.task.Task('SD-1D_linvel',
#                     task_specific_params=template_1D_linvel.default_params, 
#                     get_vars_func=template_1D_linvel.get_vars,
#                     create_data_func=create_data,
#                     input_map=template_1D_linvel.input_map,
#                     target_map=target_map,
#                     test_func=o2s.test.test_tuning,
#                     test_func_args=dict(tuning_vars_list=['HD', 'ego_SD', 'allo_SD', 'AV', 'x']))
#
#
