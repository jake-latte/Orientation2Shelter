import torch
import sys
import o2s
import o2s.Templates.vars_0D as template_0D

class HD_SD_0D_cued(template_0D.Vars0D):
    task_name = "HD_SD-0D-cued"
    default_params = {
        **template_0D.default_params,
        'cue_time': 20
    }
    input_map = {
        'av': 0,
        'sin_hd_0': 1,
        'cos_hd_0': 2,
        'sx': 3,
        'sy': 4,
        'cue_hd': 5,
        'cue_sd': 6,
    }
    target_map = {
        'sin_a': 0,
        'cos_a': 1,
        'sin_b': 2,
        'cos_b': 3
    }
    test_func = o2s.test.test_tuning
    test_func_args = dict(tuning_vars_list=['HD', 'ego_SD', 'allo_SD', 'AV'])
    @staticmethod
    def get_vars(config):
        vars = template_0D.get_vars(config)
    
        vars['cue'] = torch.randint(0, 3, (config.batch_size,1))
    
        return vars
    @staticmethod
    def create_data(config, vars, inputs, targets, mask):
        
        inputs, mask = template_0D.fill_inputs(config, vars,inputs, mask)
        mask[:,config.cue_time:] = True
    
        cues = vars['cue']
        for i, cue in enumerate(cues.squeeze()):
            if cue == 0:
                inputs[i, config.cue_time:, HD_SD_0D_cued.input_map['cue_sd']] = 0
                inputs[i, config.cue_time:, HD_SD_0D_cued.input_map['cue_hd']] = 0
    
                targets[i,:,HD_SD_0D_cued.target_map['sin_a']] = torch.sin(vars['hd'][i])
                targets[i,:,HD_SD_0D_cued.target_map['cos_a']] = torch.cos(vars['hd'][i])
                targets[i,:,HD_SD_0D_cued.target_map['sin_b']] = torch.sin(vars['sd'][i])
                targets[i,:,HD_SD_0D_cued.target_map['cos_b']] = torch.cos(vars['sd'][i])
            elif cue == 1:
                inputs[i, config.cue_time:, HD_SD_0D_cued.input_map['cue_sd']] = 1
                inputs[i, config.cue_time:, HD_SD_0D_cued.input_map['cue_hd']] = 0
    
                targets[i,:,HD_SD_0D_cued.target_map['sin_a']] = torch.sin(vars['sd'][i])
                targets[i,:,HD_SD_0D_cued.target_map['cos_a']] = torch.cos(vars['sd'][i])
                targets[i,:,HD_SD_0D_cued.target_map['sin_b']] = torch.zeros_like(vars['sd'][i])
                targets[i,:,HD_SD_0D_cued.target_map['cos_b']] = torch.zeros_like(vars['sd'][i])
            elif cue == 2:
                inputs[i, config.cue_time:, HD_SD_0D_cued.input_map['cue_sd']] = 0
                inputs[i, config.cue_time:, HD_SD_0D_cued.input_map['cue_hd']] = 1
    
                targets[i,:,HD_SD_0D_cued.target_map['sin_a']] = torch.sin(vars['hd'][i])
                targets[i,:,HD_SD_0D_cued.target_map['cos_a']] = torch.cos(vars['hd'][i])
                targets[i,:,HD_SD_0D_cued.target_map['sin_b']] = torch.zeros_like(vars['hd'][i])
                targets[i,:,HD_SD_0D_cued.target_map['cos_b']] = torch.zeros_like(vars['hd'][i])
    
        return inputs, targets, mask
    @staticmethod
    def get_joint_vars(config):
        vars = template_0D.get_joint_vars(config)
        vars['cue'] = torch.zeros((config.batch_size,1)).long()
        return vars
    @staticmethod
    def get_joint_sd_only_vars(config):
        vars = template_0D.get_joint_vars(config)
        vars['cue'] = 1*torch.ones((config.batch_size,1)).long()
        return vars
    @staticmethod
    def get_joint_hd_only_vars(config):
        vars = template_0D.get_joint_vars(config)
        vars['cue'] = 2*torch.ones((config.batch_size,1)).long()
        return vars
    @staticmethod
    def get_joint_unswapped_vars(config):
        vars = template_0D.get_joint_vars(config)
        vars['cue'] = torch.zeros((config.batch_size,1)).long()
        return vars
    @staticmethod
    def get_hd_iso_vars(config):
        vars = template_0D.get_hd_iso_vars(config)
        vars['cue'] = torch.zeros((config.batch_size,1)).long()
        return vars
    @staticmethod
    def get_sd_iso_vars(config):
        vars = template_0D.get_sd_iso_vars(config)
        vars['cue'] = torch.zeros((config.batch_size,1)).long()
        return vars
    @staticmethod
    def get_av_vars(config):
        vars = template_0D.get_av_vars(config)
        vars['cue'] = torch.zeros((config.batch_size,1)).long()
        return vars
    @staticmethod
    def get_metric_vars(config):
        vars = template_0D.get_metric_vars(config)
        vars['cue'] = torch.zeros((config.batch_size,1)).long()
        return vars
    def __init__(self, **kwargs):
        self.get_vars_func = self.get_vars
        self.get_subtask_vars_funcs = {
            "joint": self.get_joint_vars,
            "joint_sd_only": self.get_joint_sd_only_vars,
            "joint_hd_only": self.get_joint_hd_only_vars,
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

# default_params = {
#     **template_0D.default_params,
#     'cue_time': 20
# }
#
# input_map = {
#     'av': 0,
#     'sin_hd_0': 1,
#     'cos_hd_0': 2,
#     'sx': 3,
#     'sy': 4,
#     'cue_hd': 5,
#     'cue_sd': 6,
# }
#
# target_map = {
#     'sin_a': 0,
#     'cos_a': 1,
#     'sin_b': 2,
#     'cos_b': 3
# }
#
# def get_vars(config):
#     vars = template_0D.get_vars(config)
#
#     vars['cue'] = torch.randint(0, 3, (config.batch_size,1))
#
#     return vars
#
# def create_data(config, vars, inputs, targets, mask):
#
#     inputs, mask = template_0D.fill_inputs(config, vars,inputs, mask)
#     mask[:,config.cue_time:] = True
#
#     cues = vars['cue']
#     for i, cue in enumerate(cues.squeeze()):
#         if cue == 0:
#             inputs[i, config.cue_time:, input_map['cue_sd']] = 0
#             inputs[i, config.cue_time:, input_map['cue_hd']] = 0
#
#             targets[i,:,target_map['sin_a']] = torch.sin(vars['hd'][i])
#             targets[i,:,target_map['cos_a']] = torch.cos(vars['hd'][i])
#             targets[i,:,target_map['sin_b']] = torch.sin(vars['sd'][i])
#             targets[i,:,target_map['cos_b']] = torch.cos(vars['sd'][i])
#         elif cue == 1:
#             inputs[i, config.cue_time:, input_map['cue_sd']] = 1
#             inputs[i, config.cue_time:, input_map['cue_hd']] = 0
#
#             targets[i,:,target_map['sin_a']] = torch.sin(vars['sd'][i])
#             targets[i,:,target_map['cos_a']] = torch.cos(vars['sd'][i])
#             targets[i,:,target_map['sin_b']] = torch.zeros_like(vars['sd'][i])
#             targets[i,:,target_map['cos_b']] = torch.zeros_like(vars['sd'][i])
#         elif cue == 2:
#             inputs[i, config.cue_time:, input_map['cue_sd']] = 0
#             inputs[i, config.cue_time:, input_map['cue_hd']] = 1
#
#             targets[i,:,target_map['sin_a']] = torch.sin(vars['hd'][i])
#             targets[i,:,target_map['cos_a']] = torch.cos(vars['hd'][i])
#             targets[i,:,target_map['sin_b']] = torch.zeros_like(vars['hd'][i])
#             targets[i,:,target_map['cos_b']] = torch.zeros_like(vars['hd'][i])
#
#     return inputs, targets, mask
#
# def get_joint_vars(config):
#     vars = template_0D.get_joint_vars(config)
#     vars['cue'] = torch.zeros((config.batch_size,1)).long()
#     return vars
#
# def get_joint_sd_only_vars(config):
#     vars = template_0D.get_joint_vars(config)
#     vars['cue'] = 1*torch.ones((config.batch_size,1)).long()
#     return vars
#
# def get_joint_hd_only_vars(config):
#     vars = template_0D.get_joint_vars(config)
#     vars['cue'] = 2*torch.ones((config.batch_size,1)).long()
#     return vars
#
# def get_joint_unswapped_vars(config):
#     vars = template_0D.get_joint_vars(config)
#     vars['cue'] = torch.zeros((config.batch_size,1)).long()
#     return vars
#
# def get_hd_iso_vars(config):
#     vars = template_0D.get_hd_iso_vars(config)
#     vars['cue'] = torch.zeros((config.batch_size,1)).long()
#     return vars
#
# def get_sd_iso_vars(config):
#     vars = template_0D.get_sd_iso_vars(config)
#     vars['cue'] = torch.zeros((config.batch_size,1)).long()
#     return vars
#
# def get_av_vars(config):
#     vars = template_0D.get_av_vars(config)
#     vars['cue'] = torch.zeros((config.batch_size,1)).long()
#     return vars
#
# def get_metric_vars(config):
#     vars = template_0D.get_metric_vars(config)
#     vars['cue'] = torch.zeros((config.batch_size,1)).long()
#     return vars
#
# HD_SD_0D_CUED_TASK = o2s.task.Task('HD_SD-0D-cued',
#                             task_specific_params=default_params, 
#                             get_vars_func=get_vars,
#                             create_data_func=create_data,
#                             input_map=input_map,
#                             target_map=target_map,
#                             test_func=o2s.test.test_tuning,
#                             test_func_args=dict(tuning_vars_list=['HD', 'ego_SD', 'allo_SD', 'AV']),
#                             get_subtask_vars_funcs={'joint': get_joint_vars,
#                                                     'joint_sd_only': get_joint_sd_only_vars,
#                                                     'joint_hd_only': get_joint_hd_only_vars,
#                                                     'hd_iso': get_hd_iso_vars,
#                                                     'sd_iso': get_sd_iso_vars,
#                                                     'av': get_av_vars,
#                                                     'metric': get_metric_vars})
#
#
#
#
#
#
