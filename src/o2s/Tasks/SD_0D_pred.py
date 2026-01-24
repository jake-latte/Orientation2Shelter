import torch
import sys
import o2s
import o2s.Templates.vars_0D as template_0D

class SD_0D_pred(template_0D.Vars0D):
    task_name = "SD-0D_pred"
    default_params = {
        **template_0D.default_params,
        'delay': 10
    }
    input_map = {
        'sx': 0,
        'sy': 1,
        'sin_hd': 2,
        'cos_hd': 3,
    }
    target_map = {}
    get_vars = staticmethod(template_0D.get_vars)
    get_joint_vars = staticmethod(template_0D.get_joint_vars)
    get_hd_iso_vars = staticmethod(template_0D.get_hd_iso_vars)
    get_sd_iso_vars = staticmethod(template_0D.get_sd_iso_vars)
    get_av_vars = staticmethod(template_0D.get_av_vars)
    get_metric_vars = staticmethod(template_0D.get_metric_vars)
    test_func = o2s.test.test_tuning
    test_func_args = dict(tuning_vars_list=['HD', 'ego_SD', 'allo_SD', 'AV'])
    @staticmethod
    def init_func(task):
        config = task.config
        if len(task.target_map) == 0:
            task.config.n_outputs = 2 * config.delay
            for i in range(task.config.delay):
                task.target_map[f'sin_sd_{i}'] = 2*i
                task.target_map[f'cos_sd_{i}'] = 2*i + 1
    
        if len(task.SD_0D_pred.input_map) == 4:
            task.config.n_inputs = 4 + task.config.delay
            for i in range(task.config.delay):
                task.SD_0D_pred.input_map[f'av_{i}'] = i + 4
    @staticmethod
    def create_data(config, vars, inputs, targets, mask):
        batch_size, n_timesteps = inputs.shape[0], inputs.shape[1]
        init_duration = config.init_duration
        
        inputs[:,:init_duration,SD_0D_pred.input_map['sx']] = vars['sx'].reshape((batch_size,1)).repeat((1,init_duration))
        inputs[:,:init_duration,SD_0D_pred.input_map['sy']] = vars['sy'].reshape((batch_size,1)).repeat((1,init_duration))
        
        inputs[:,:,SD_0D_pred.input_map['sin_hd']] = torch.sin(vars['hd'])
        inputs[:,:,SD_0D_pred.input_map['cos_hd']] = torch.cos(vars['hd'])
    
        for i in range(config.delay):
            if i==0:
                inputs[:,:,SD_0D_pred.input_map[f'av_{i}']] = vars['av']
            else:
                inputs[:,:-i,SD_0D_pred.input_map[f'av_{i}']] = vars['av'][:,i:]
    
        for i in range(config.delay):
            if i==0:
                targets[:,:,2*i] = torch.sin(vars['sd'])
                targets[:,:,2*i + 1] = torch.cos(vars['sd'])
            else:
                targets[:,:-i,2*i] = torch.sin(vars['sd'][:,i:])
                targets[:,:-i,2*i + 1] = torch.cos(vars['sd'][:,i:])
                mask[:,-i:,2*i] = False
                mask[:,-i:,2*i + 1] = False
    
        mask[:,:init_duration] = False
    
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
#     'delay': 10
# }
#
# input_map = {
#     'sx': 0,
#     'sy': 1,
#     'sin_hd': 2,
#     'cos_hd': 3,
# }
#
# def init_func(task):
#     config = task.config
#     if len(task.target_map) == 0:
#         task.config.n_outputs = 2 * config.delay
#         for i in range(task.config.delay):
#             task.target_map[f'sin_sd_{i}'] = 2*i
#             task.target_map[f'cos_sd_{i}'] = 2*i + 1
#
#     if len(task.input_map) == 4:
#         task.config.n_inputs = 4 + task.config.delay
#         for i in range(task.config.delay):
#             task.input_map[f'av_{i}'] = i + 4
#
#
# def create_data(config, vars, inputs, targets, mask):
#     batch_size, n_timesteps = inputs.shape[0], inputs.shape[1]
#     init_duration = config.init_duration
#
#     inputs[:,:init_duration,input_map['sx']] = vars['sx'].reshape((batch_size,1)).repeat((1,init_duration))
#     inputs[:,:init_duration,input_map['sy']] = vars['sy'].reshape((batch_size,1)).repeat((1,init_duration))
#
#     inputs[:,:,input_map['sin_hd']] = torch.sin(vars['hd'])
#     inputs[:,:,input_map['cos_hd']] = torch.cos(vars['hd'])
#
#     for i in range(config.delay):
#         if i==0:
#             inputs[:,:,input_map[f'av_{i}']] = vars['av']
#         else:
#             inputs[:,:-i,input_map[f'av_{i}']] = vars['av'][:,i:]
#
#     for i in range(config.delay):
#         if i==0:
#             targets[:,:,2*i] = torch.sin(vars['sd'])
#             targets[:,:,2*i + 1] = torch.cos(vars['sd'])
#         else:
#             targets[:,:-i,2*i] = torch.sin(vars['sd'][:,i:])
#             targets[:,:-i,2*i + 1] = torch.cos(vars['sd'][:,i:])
#             mask[:,-i:,2*i] = False
#             mask[:,-i:,2*i + 1] = False
#
#     mask[:,:init_duration] = False
#
#     return inputs, targets, mask
#
#
#
# SD_0D_pred_TASK = o2s.task.Task('SD-0D_pred',
#                     task_specific_params=default_params, 
#                     get_vars_func=template_0D.get_vars,
#                     create_data_func=create_data,
#                     input_map=input_map,
#                     target_map={},
#                     init_func=init_func,
#                     test_func=o2s.test.test_tuning,
#                     test_func_args=dict(tuning_vars_list=['HD', 'ego_SD', 'allo_SD', 'AV']),
#                     get_subtask_vars_funcs={'joint': template_0D.get_joint_vars,
#                                             'hd_iso': template_0D.get_hd_iso_vars,
#                                             'sd_iso': template_0D.get_sd_iso_vars,
#                                             'av': template_0D.get_av_vars,
#                                             'metric': template_0D.get_metric_vars})
#
#
