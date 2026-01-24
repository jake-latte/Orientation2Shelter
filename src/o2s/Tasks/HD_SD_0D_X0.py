import torch
import numpy as np
import sys
from typing import Tuple
import o2s
import o2s.Templates.vars_0D as template_0D

class HD_SD_0D_X0(template_0D.Vars0D):
    task_name = "HD_SD-0D-X0"
    target_map = {
        'sin_hd': 0,
        'cos_hd': 1,
        'sin_sd': 2,
        'cos_sd': 3
    }
    default_params = template_0D.default_params
    input_map = template_0D.input_map
    get_joint_vars = staticmethod(template_0D.get_joint_vars)
    get_hd_iso_vars = staticmethod(template_0D.get_hd_iso_vars)
    get_sd_iso_vars = staticmethod(template_0D.get_sd_iso_vars)
    get_av_vars = staticmethod(template_0D.get_av_vars)
    get_metric_vars = staticmethod(template_0D.get_metric_vars)
    test_func = o2s.test.test_tuning
    test_func_args = dict(tuning_vars_list=['HD', 'ego_SD', 'allo_SD', 'AV'])
    @staticmethod
    def get_vars(config):
        vars = template_0D.get_vars(config)
        x_0 = torch.normal(mean=0, std=config.hidden_g/np.sqrt(config.n_neurons), size=(config.batch_size, config.n_neurons))
        vars['x_0'] = x_0
    
        return vars
    @staticmethod
    def create_data(config, vars, inputs, targets, mask):
        
        inputs, mask = template_0D.fill_inputs(config, vars,inputs, mask)
    
        targets[:,:,HD_SD_0D_X0.target_map['sin_hd']] = torch.sin(vars['hd'])
        targets[:,:,HD_SD_0D_X0.target_map['cos_hd']] = torch.cos(vars['hd'])
        targets[:,:,HD_SD_0D_X0.target_map['sin_sd']] = torch.sin(vars['sd'])
        targets[:,:,HD_SD_0D_X0.target_map['cos_sd']] = torch.cos(vars['sd'])
    
        return inputs, targets, mask
    @staticmethod
    def loss_func(task: o2s.task.Task, net: 'RNN', batch: dict) -> Tuple[torch.tensor, torch.tensor]:
        _, activity, outputs = net(batch['inputs'], noise=batch['noise'], x_0=batch['vars']['x_0'])
    
        # MSE of (masked) outputs
        loss_prediction = torch.sum(torch.square(outputs - batch['targets'])[batch['mask']]) / torch.sum(batch['mask']==1)
    
        # Rate Loss
        if task.config.rate_loss_type == 1:
            loss_activity = task.config.rate_lambda * torch.mean(torch.abs(activity))
        else:
            loss_activity = task.config.rate_lambda * torch.mean(torch.square(activity))
    
        # Weight Loss
        if task.config.weight_loss_type == 1:
            loss_weight = 0
            for weight, include in zip([net.W_rec.weight, net.W_in.weight, net.W_out.weight], [task.config.learn_W_rec, task.config.learn_W_in, task.config.learn_W_out]):
                if include:
                    loss_weight += task.config.weight_lambda * torch.mean(torch.abs(weight))
        else:
            loss_weight = 0
            for weight, include in zip([net.W_rec.weight, net.W_in.weight, net.W_out.weight], [task.config.learn_W_rec, task.config.learn_W_in, task.config.learn_W_out]):
                if include:
                    loss_weight += task.config.weight_lambda * torch.mean(torch.square(weight))
    
        loss = loss_prediction + loss_activity + loss_weight
    
        return loss, outputs
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
            loss_func=self.loss_func,
            test_func=self.test_func,
            test_func_args=self.test_func_args,
            get_subtask_vars_funcs=self.get_subtask_vars_funcs,
            **kwargs
        )

# target_map = {
#     'sin_hd': 0,
#     'cos_hd': 1,
#     'sin_sd': 2,
#     'cos_sd': 3
# }
#
# def get_vars(config):
#     vars = template_0D.get_vars(config)
#     x_0 = torch.normal(mean=0, std=config.hidden_g/np.sqrt(config.n_neurons), size=(config.batch_size, config.n_neurons))
#     vars['x_0'] = x_0
#
#     return vars
#
# def create_data(config, vars, inputs, targets, mask):
#
#     inputs, mask = template_0D.fill_inputs(config, vars,inputs, mask)
#
#     targets[:,:,target_map['sin_hd']] = torch.sin(vars['hd'])
#     targets[:,:,target_map['cos_hd']] = torch.cos(vars['hd'])
#     targets[:,:,target_map['sin_sd']] = torch.sin(vars['sd'])
#     targets[:,:,target_map['cos_sd']] = torch.cos(vars['sd'])
#
#     return inputs, targets, mask
#
# def loss_func(task: o2s.task.Task, net: 'RNN', batch: dict) -> Tuple[torch.tensor, torch.tensor]:
#     _, activity, outputs = net(batch['inputs'], noise=batch['noise'], x_0=batch['vars']['x_0'])
#
#     # MSE of (masked) outputs
#     loss_prediction = torch.sum(torch.square(outputs - batch['targets'])[batch['mask']]) / torch.sum(batch['mask']==1)
#
#     # Rate Loss
#     if task.config.rate_loss_type == 1:
#         loss_activity = task.config.rate_lambda * torch.mean(torch.abs(activity))
#     else:
#         loss_activity = task.config.rate_lambda * torch.mean(torch.square(activity))
#
#     # Weight Loss
#     if task.config.weight_loss_type == 1:
#         loss_weight = 0
#         for weight, include in zip([net.W_rec.weight, net.W_in.weight, net.W_out.weight], [task.config.learn_W_rec, task.config.learn_W_in, task.config.learn_W_out]):
#             if include:
#                 loss_weight += task.config.weight_lambda * torch.mean(torch.abs(weight))
#     else:
#         loss_weight = 0
#         for weight, include in zip([net.W_rec.weight, net.W_in.weight, net.W_out.weight], [task.config.learn_W_rec, task.config.learn_W_in, task.config.learn_W_out]):
#             if include:
#                 loss_weight += task.config.weight_lambda * torch.mean(torch.square(weight))
#
#     loss = loss_prediction + loss_activity + loss_weight
#
#     return loss, outputs
#
#
#
# HD_SD_0D_X0_TASK = o2s.task.Task('HD_SD-0D-X0',
#                     task_specific_params=template_0D.default_params, 
#                     get_vars_func=get_vars,
#                     create_data_func=create_data,
#                     loss_func=loss_func,
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
