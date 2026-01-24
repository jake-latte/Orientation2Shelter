import torch
import numpy as np
import sys
from typing import Tuple
import o2s
import o2s.Templates.vars_2D_vecvel as template_2D_vecvel

class PI_HD_SD_2D_vecvel(template_2D_vecvel.Vars2DVecvel):
    task_name = "PI_HD_SD-2D_vecvel"
    default_params = {
        **template_2D_vecvel.default_params,
        'PI_penalty_lambda': 10
    }
    target_map = {
        'x': 0,
        'y': 1,
        'sin_hd': 2,
        'cos_hd': 3,
        'sin_sd': 4,
        'cos_sd': 5
    }
    input_map = template_2D_vecvel.input_map
    get_vars = staticmethod(template_2D_vecvel.get_vars)
    test_func = o2s.test.test_tuning
    test_func_args = dict(tuning_vars_list=['HD', 'ego_SD', 'allo_SD', 'AV', 'x', 'y'])
    @staticmethod
    def loss_func(task: o2s.task.Task, net: 'RNN', batch: dict) -> Tuple[torch.tensor, torch.tensor]:
    
        _, activity, outputs = net(batch['inputs'], noise=batch['noise'])
    
        loss_all = torch.sum(torch.square(outputs - batch['targets'])[batch['mask']]) / torch.sum(batch['mask']==1)
        loss_PI = torch.sum(torch.square(
            outputs[:,:,[task.PI_HD_SD_2D_vecvel.target_map['x'], task.PI_HD_SD_2D_vecvel.target_map['y']]] - batch['targets'][:,:,[task.PI_HD_SD_2D_vecvel.target_map['x'], task.PI_HD_SD_2D_vecvel.target_map['y']]])[batch['mask'][:,:,[task.PI_HD_SD_2D_vecvel.target_map['x'], task.PI_HD_SD_2D_vecvel.target_map['y']]]]) / torch.sum(batch['mask'][:,:,[task.PI_HD_SD_2D_vecvel.target_map['x'], task.PI_HD_SD_2D_vecvel.target_map['y']]]==1)
        
        loss_prediction = loss_all + (task.config.PI_penalty_lambda - 1) * loss_PI
    
        # Rate L2
        loss_activity = task.config.rate_lambda * torch.mean(activity**2)
    
        # I think more appropriate:
        loss_weight = task.config.weight_lambda * (torch.mean(net.W_rec.weight**2) + torch.mean(net.W_in.weight**2) + torch.mean(net.W_out.weight**2))
    
        loss = loss_prediction + loss_activity + loss_weight
    
        return loss, outputs
    @staticmethod
    def create_data(config, vars, inputs, targets, mask):
    
        inputs, mask = template_2D_vecvel.fill_inputs(config, vars, inputs, mask)
    
        targets[:,:,PI_HD_SD_2D_vecvel.target_map['x']] = vars['x']
        targets[:,:,PI_HD_SD_2D_vecvel.target_map['y']] = vars['y']
        targets[:,:,PI_HD_SD_2D_vecvel.target_map['sin_hd']] = torch.sin(vars['hd'])
        targets[:,:,PI_HD_SD_2D_vecvel.target_map['cos_hd']] = torch.cos(vars['hd'])
        targets[:,:,PI_HD_SD_2D_vecvel.target_map['sin_sd']] = torch.sin(vars['sd'])
        targets[:,:,PI_HD_SD_2D_vecvel.target_map['cos_sd']] = torch.cos(vars['sd'])
    
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

# default_params = {
#     **template_2D_vecvel.default_params,
#     'PI_penalty_lambda': 10
# }
#
# def loss_func(task: o2s.task.Task, net: 'RNN', batch: dict) -> Tuple[torch.tensor, torch.tensor]:
#
#     _, activity, outputs = net(batch['inputs'], noise=batch['noise'])
#
#     loss_all = torch.sum(torch.square(outputs - batch['targets'])[batch['mask']]) / torch.sum(batch['mask']==1)
#     loss_PI = torch.sum(torch.square(
#         outputs[:,:,[task.target_map['x'], task.target_map['y']]] - batch['targets'][:,:,[task.target_map['x'], task.target_map['y']]])[batch['mask'][:,:,[task.target_map['x'], task.target_map['y']]]]) / torch.sum(batch['mask'][:,:,[task.target_map['x'], task.target_map['y']]]==1)
#
#     loss_prediction = loss_all + (task.config.PI_penalty_lambda - 1) * loss_PI
#
#     # Rate L2
#     loss_activity = task.config.rate_lambda * torch.mean(activity**2)
#
#     # I think more appropriate:
#     loss_weight = task.config.weight_lambda * (torch.mean(net.W_rec.weight**2) + torch.mean(net.W_in.weight**2) + torch.mean(net.W_out.weight**2))
#
#     loss = loss_prediction + loss_activity + loss_weight
#
#     return loss, outputs
#
# target_map = {
#     'x': 0,
#     'y': 1,
#     'sin_hd': 2,
#     'cos_hd': 3,
#     'sin_sd': 4,
#     'cos_sd': 5
# }
#
# def create_data(config, vars, inputs, targets, mask):
#
#     inputs, mask = template_2D_vecvel.fill_inputs(config, vars, inputs, mask)
#
#     targets[:,:,target_map['x']] = vars['x']
#     targets[:,:,target_map['y']] = vars['y']
#     targets[:,:,target_map['sin_hd']] = torch.sin(vars['hd'])
#     targets[:,:,target_map['cos_hd']] = torch.cos(vars['hd'])
#     targets[:,:,target_map['sin_sd']] = torch.sin(vars['sd'])
#     targets[:,:,target_map['cos_sd']] = torch.cos(vars['sd'])
#
#     return inputs, targets, mask
#
#
#
# PI_HD_SD_2D_vecvel_TASK = o2s.task.Task('PI_HD_SD-2D_vecvel',
#                             task_specific_params=default_params, 
#                             get_vars_func=template_2D_vecvel.get_vars,
#                             create_data_func=create_data,
#                             input_map=template_2D_vecvel.input_map,
#                             target_map=target_map,
#                             test_func=o2s.test.test_tuning,
#                             test_func_args=dict(tuning_vars_list=['HD', 'ego_SD', 'allo_SD', 'AV', 'x', 'y']))
#
#
