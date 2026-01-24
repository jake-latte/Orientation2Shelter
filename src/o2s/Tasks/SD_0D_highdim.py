import torch
import sys
import o2s
import o2s.Templates.vars_0D as template_0D
import o2s.Templates.vars_highdim as template_highdim

class SD_0D_highdim(template_0D.Vars0D, template_highdim.VarsHighdim):
    task_name = "SD-0D-highdim"
    default_params = {
        **template_0D.default_params,
        'n_transformed_outputs': 10,
        'transformer_seed': 0
    }
    input_map = template_0D.input_map
    target_map = {}
    get_vars = staticmethod(template_0D.get_vars)
    get_joint_vars = staticmethod(template_0D.get_joint_vars)
    get_hd_iso_vars = staticmethod(template_0D.get_hd_iso_vars)
    get_sd_iso_vars = staticmethod(template_0D.get_sd_iso_vars)
    get_av_vars = staticmethod(template_0D.get_av_vars)
    get_metric_vars = staticmethod(template_0D.get_metric_vars)
    test_func = o2s.test.test_tuning
    test_func_args = dict(tuning_vars_list=['HD', 'ego_SD', 'allo_SD', 'AV'])
    init_func = template_highdim.init_func
    @staticmethod
    def create_data(config, vars, inputs, targets, mask):
        
        inputs, mask = template_0D.fill_inputs(config, vars,inputs, mask)
    
        inter_targets = torch.stack((torch.sin(vars['sd']), torch.cos(vars['sd'])), dim=2)
        targets = torch.matmul(inter_targets, config.transformer.T).to(targets.device)
    
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

# default_params = {
#     **template_0D.default_params,
#     'n_transformed_outputs': 10,
#     'transformer_seed': 0
# }
#
#
# def create_data(config, vars, inputs, targets, mask):
#
#     inputs, mask = template_0D.fill_inputs(config, vars,inputs, mask)
#
#     inter_targets = torch.stack((torch.sin(vars['sd']), torch.cos(vars['sd'])), dim=2)
#     targets = torch.matmul(inter_targets, config.transformer.T).to(targets.device)
#
#     return inputs, targets, mask
#
#
#
# SD_0D_HIGHDIM_TASK = o2s.task.Task('SD-0D-highdim',
#                     task_specific_params=default_params, 
#                     init_func=template_highdim.init_func,
#                     get_vars_func=template_0D.get_vars,
#                     create_data_func=create_data,
#                     input_map=template_0D.input_map,
#                     target_map={},
#                     test_func=o2s.test.test_tuning,
#                     test_func_args=dict(tuning_vars_list=['HD', 'ego_SD', 'allo_SD', 'AV']),
#                     get_subtask_vars_funcs={'joint': template_0D.get_joint_vars,
#                                             'hd_iso': template_0D.get_hd_iso_vars,
#                                             'sd_iso': template_0D.get_sd_iso_vars,
#                                             'av': template_0D.get_av_vars,
#                                             'metric': template_0D.get_metric_vars})
#
#
