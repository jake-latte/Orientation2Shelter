import torch
import numpy as np
import sys
from typing import Tuple
import o2s
import o2s.Templates.vars_0D as template_0D
import o2s.Templates.vars_1D_vecvel as template_1D
import o2s.Templates.vars_2D_vecvel as template_2D

class SD_2D_curriculum(template_0D.Vars0D, template_1D.Vars1DVecvel, template_2D.Vars2DVecvel):
    task_name = "SD-2D_curriculum"
    default_params = {
        'stage': 1e-9,
        '0D_loss_threshold': 0.05,
        '1D_loss_threshold': 0.05,
        '2D_loss_threshold': 0.05,
    
    
        'v_step_shape': 2,
        'v_step_scale': 0.005,
        'v_step_momentum': 0.3,
        'v_step_zero_prob': 0.5,
        'min_xy': 0,
        'max_xy': 2.5,
    
        'v_step_std': 0.001,
        'v_step_hd_bias': 0.01,
    
        'av_step_std': 0.1, 
        'av_step_momentum': 0.5,
        'av_step_zero_prob': 0.5,
        'init_duration': 10
    }
    target_map = {
        'sin_sd': 0,
        'cos_sd': 1
    }
    input_map = template_2D.input_map
    input_map = input_map
    get_joint_vars = staticmethod(template_2D.get_joint_vars)
    get_hd_iso_vars = staticmethod(template_2D.get_hd_iso_vars)
    get_sd_iso_vars = staticmethod(template_2D.get_sd_iso_vars)
    get_av_vars = staticmethod(template_2D.get_av_vars)
    get_metric_vars = staticmethod(template_2D.get_metric_vars)
    test_func = o2s.test.test_tuning
    test_func_args = dict(tuning_vars_list=['HD', 'ego_SD', 'allo_SD', 'AV', 'x', 'y'])
    @staticmethod
    def loss_func(task: o2s.task.Task, net: 'RNN', batch: dict) -> Tuple[torch.tensor, torch.tensor]:
        for_training = (batch['inputs'].shape[0] == task.config.batch_size and batch['inputs'].shape[1] == task.config.n_timesteps)
    
        loss, outputs = o2s.task.default_loss_func(task, net, batch)
    
        if 0<task.config.stage<1 and loss.item() < task.config['0D_loss_threshold']:
            task.config.stage += 0.1
            print(f'Pushing to 1D stage {task.config.stage}')
        elif 1<=task.config.stage<2 and loss.item() < task.config['1D_loss_threshold']:
            task.config.stage += 0.1
            print(f'Pushing to 2D stage {task.config.stage}')
        elif 2<=task.config.stage and loss.item() < task.config['2D_loss_threshold']:
            if task.config.stage<102:
                task.config.stage += 1
                print(f'Bumping 2D curriculum stage {task.config.stage}')
    
    
        return loss, outputs
    @staticmethod
    def get_curriculum_vars(config):
        batch_size, n_timesteps = config.batch_size, config.n_timesteps
        init_duration = config.init_duration
        v_step_std, v_step_momentum, v_step_hd_bias, v_step_zero_prob = config.v_step_std, config.v_step_momentum, config.v_step_hd_bias, config.v_step_zero_prob
    
        vars = template_0D.get_vars(config)
    
        angular_velocity = torch.zeros((batch_size, n_timesteps))
        angle_0 = 2*np.pi*torch.rand(batch_size)
        angle = torch.tile(angle_0.reshape(batch_size,1), dims=(1,n_timesteps))
    
        shelter_x = (2*torch.rand((batch_size,1)) - 1).repeat((1,n_timesteps))
        shelter_y = (2*torch.rand((batch_size,1)) - 1).repeat((1,n_timesteps))
    
        position_0 = torch.stack((torch.cos(angle_0), torch.sin(angle_0)), dim=1)
        position = torch.tile(position_0.reshape((batch_size,1,2)), dims=(1,n_timesteps,1)) + torch.stack((shelter_x, shelter_y), dim=2)
    
        zero_trials = torch.where(torch.rand((batch_size,)) < v_step_zero_prob)
    
        step_adjust_scale = (config.stage-2)/100
    
        for t in range(init_duration, n_timesteps):
            normal = torch.distributions.normal.Normal(loc=v_step_hd_bias*(vars['hd'][:,t] - angle[:,t]), scale=torch.ones((batch_size,))*v_step_std)
    
            v_step = normal.sample() + v_step_momentum * angular_velocity[:,t-1]
            if t > n_timesteps*(1/4) and t < n_timesteps*(3/4):
                v_step[zero_trials] = 0
    
            angular_velocity[:, t] = v_step
            pre_adjust_angle = angle[:,t] + angular_velocity[:,t]
    
            angle_step = torch.stack((torch.cos(pre_adjust_angle), torch.sin(pre_adjust_angle)), dim=1) - torch.stack((torch.cos(angle[:,t]), torch.sin(angle[:,t])), dim=1)
            
            head_direction_step = torch.stack(((1.2-step_adjust_scale) * v_step * torch.cos(vars['hd'][:,t]), (1.2-step_adjust_scale) * v_step * torch.sin(vars['hd'][:,t])), dim=1)
    
            post_adjust_step = angle_step + step_adjust_scale * (head_direction_step - angle_step)
            position[:,t:] += torch.tile(post_adjust_step.reshape((batch_size,1,2)), dims=(1,n_timesteps-t,1))
    
            post_adjust_angle = torch.atan2(position[:,t,1] - shelter_y[:,0], position[:,t,0] - shelter_x[:,0])
            angle[:,t:] = torch.tile(post_adjust_angle.reshape((batch_size,1)), dims=(1,n_timesteps-t))
    
    
        velocity = torch.cat((torch.zeros((batch_size,1,2)), torch.diff(position, dim=1)), dim=1)
        angle = torch.remainder(angle, 2*np.pi)
    
        d_x = shelter_x - position[:,:,0]
        d_y = shelter_y - position[:,:,1]
        dist = torch.sqrt(d_x**2 + d_y**2)
        pert = 10e-6 * torch.ones((batch_size,n_timesteps))
        dist[torch.where(dist==0)[0]] += (pert * np.random.choice([1, -1]))[torch.where(dist==0)[0]]
        allo_shelter_angle = torch.atan2(d_y, d_x)
        allo_shelter_angle[allo_shelter_angle<0] += 2*np.pi
    
        ego_angle = allo_shelter_angle - vars['hd']
        ego_angle[ego_angle<0] += 2*np.pi
    
        vars['sx'] = shelter_x[:,0]
        vars['sy'] = shelter_y[:,0]
        vars['sd'] = ego_angle
        vars['x'] = position[:,:,0]
        vars['y'] = position[:,:,1]
        vars['xv'] = velocity[:,:,0]
        vars['yv'] = velocity[:,:,1]
    
        return vars
    @staticmethod
    def get_vars(config):
        batch_size, n_timesteps = config.batch_size, config.n_timesteps
        max_xy, min_xy = config.max_xy, config.min_xy
    
        if 0<config.stage<1:
    
            vars = template_0D.get_vars(config)
    
            vars['x'] = ((max_xy-min_xy)*torch.rand((batch_size,1)) + min_xy).repeat((1,n_timesteps))
            vars['y'] = ((max_xy-min_xy)*torch.rand((batch_size,1)) + min_xy).repeat((1,n_timesteps))
    
            vars['sx'] = vars['sx'] + vars['x'][:,0]
            vars['sy'] = vars['sy'] + vars['y'][:,0]
    
            vars['xv'] = vars['yv'] = torch.zeros((batch_size, n_timesteps))
    
        elif 1<=config.stage<2:
    
            vars = template_1D.get_vars(config)
    
            vars['sx'] = ((max_xy-min_xy)*torch.rand((batch_size,1)) + min_xy)
            vars['sy'] = ((max_xy-min_xy)*torch.rand((batch_size,1)) + min_xy)
    
            vars['y'] = vars['sy'].repeat((1,n_timesteps)) + torch.sin(vars['x'])
            vars['x'] = vars['sx'].repeat((1,n_timesteps)) + torch.cos(vars['x'])
    
        elif 2<=config.stage:
            vars = get_curriculum_vars(config)
        
        else:
            vars = template_2D.get_vars(config)
    
        return vars
    @staticmethod
    def create_data(config, vars, inputs, targets, mask):
        batch_size, n_timesteps = config.batch_size, config.n_timesteps
    
        inputs, mask = template_2D.fill_inputs(config, vars, inputs, mask)
    
        targets[:,:,SD_2D_curriculum.target_map['sin_sd']] = torch.sin(vars['sd'])
        targets[:,:,SD_2D_curriculum.target_map['cos_sd']] = torch.cos(vars['sd'])
    
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
            loss_func=self.loss_func,
            test_func=self.test_func,
            test_func_args=self.test_func_args,
            get_subtask_vars_funcs=self.get_subtask_vars_funcs,
            **kwargs
        )

# default_params = {
#     'stage': 1e-9,
#     '0D_loss_threshold': 0.05,
#     '1D_loss_threshold': 0.05,
#     '2D_loss_threshold': 0.05,
#
#
#     'v_step_shape': 2,
#     'v_step_scale': 0.005,
#     'v_step_momentum': 0.3,
#     'v_step_zero_prob': 0.5,
#     'min_xy': 0,
#     'max_xy': 2.5,
#
#     'v_step_std': 0.001,
#     'v_step_hd_bias': 0.01,
#
#     'av_step_std': 0.1, 
#     'av_step_momentum': 0.5,
#     'av_step_zero_prob': 0.5,
#     'init_duration': 10
# }
#
# input_map = template_2D.input_map
#
# target_map = {
#     'sin_sd': 0,
#     'cos_sd': 1
# }
#
# def loss_func(task: o2s.task.Task, net: 'RNN', batch: dict) -> Tuple[torch.tensor, torch.tensor]:
#     for_training = (batch['inputs'].shape[0] == task.config.batch_size and batch['inputs'].shape[1] == task.config.n_timesteps)
#
#     loss, outputs = o2s.task.default_loss_func(task, net, batch)
#
#     if 0<task.config.stage<1 and loss.item() < task.config['0D_loss_threshold']:
#         task.config.stage += 0.1
#         print(f'Pushing to 1D stage {task.config.stage}')
#     elif 1<=task.config.stage<2 and loss.item() < task.config['1D_loss_threshold']:
#         task.config.stage += 0.1
#         print(f'Pushing to 2D stage {task.config.stage}')
#     elif 2<=task.config.stage and loss.item() < task.config['2D_loss_threshold']:
#         if task.config.stage<102:
#             task.config.stage += 1
#             print(f'Bumping 2D curriculum stage {task.config.stage}')
#
#
#     return loss, outputs
#
#
#
#
#
# def get_curriculum_vars(config):
#     batch_size, n_timesteps = config.batch_size, config.n_timesteps
#     init_duration = config.init_duration
#     v_step_std, v_step_momentum, v_step_hd_bias, v_step_zero_prob = config.v_step_std, config.v_step_momentum, config.v_step_hd_bias, config.v_step_zero_prob
#
#     vars = template_0D.get_vars(config)
#
#     angular_velocity = torch.zeros((batch_size, n_timesteps))
#     angle_0 = 2*np.pi*torch.rand(batch_size)
#     angle = torch.tile(angle_0.reshape(batch_size,1), dims=(1,n_timesteps))
#
#     shelter_x = (2*torch.rand((batch_size,1)) - 1).repeat((1,n_timesteps))
#     shelter_y = (2*torch.rand((batch_size,1)) - 1).repeat((1,n_timesteps))
#
#     position_0 = torch.stack((torch.cos(angle_0), torch.sin(angle_0)), dim=1)
#     position = torch.tile(position_0.reshape((batch_size,1,2)), dims=(1,n_timesteps,1)) + torch.stack((shelter_x, shelter_y), dim=2)
#
#     zero_trials = torch.where(torch.rand((batch_size,)) < v_step_zero_prob)
#
#     step_adjust_scale = (config.stage-2)/100
#
#     for t in range(init_duration, n_timesteps):
#         normal = torch.distributions.normal.Normal(loc=v_step_hd_bias*(vars['hd'][:,t] - angle[:,t]), scale=torch.ones((batch_size,))*v_step_std)
#
#         v_step = normal.sample() + v_step_momentum * angular_velocity[:,t-1]
#         if t > n_timesteps*(1/4) and t < n_timesteps*(3/4):
#             v_step[zero_trials] = 0
#
#         angular_velocity[:, t] = v_step
#         pre_adjust_angle = angle[:,t] + angular_velocity[:,t]
#
#         angle_step = torch.stack((torch.cos(pre_adjust_angle), torch.sin(pre_adjust_angle)), dim=1) - torch.stack((torch.cos(angle[:,t]), torch.sin(angle[:,t])), dim=1)
#
#         head_direction_step = torch.stack(((1.2-step_adjust_scale) * v_step * torch.cos(vars['hd'][:,t]), (1.2-step_adjust_scale) * v_step * torch.sin(vars['hd'][:,t])), dim=1)
#
#         post_adjust_step = angle_step + step_adjust_scale * (head_direction_step - angle_step)
#         position[:,t:] += torch.tile(post_adjust_step.reshape((batch_size,1,2)), dims=(1,n_timesteps-t,1))
#
#         post_adjust_angle = torch.atan2(position[:,t,1] - shelter_y[:,0], position[:,t,0] - shelter_x[:,0])
#         angle[:,t:] = torch.tile(post_adjust_angle.reshape((batch_size,1)), dims=(1,n_timesteps-t))
#
#
#     velocity = torch.cat((torch.zeros((batch_size,1,2)), torch.diff(position, dim=1)), dim=1)
#     angle = torch.remainder(angle, 2*np.pi)
#
#     d_x = shelter_x - position[:,:,0]
#     d_y = shelter_y - position[:,:,1]
#     dist = torch.sqrt(d_x**2 + d_y**2)
#     pert = 10e-6 * torch.ones((batch_size,n_timesteps))
#     dist[torch.where(dist==0)[0]] += (pert * np.random.choice([1, -1]))[torch.where(dist==0)[0]]
#     allo_shelter_angle = torch.atan2(d_y, d_x)
#     allo_shelter_angle[allo_shelter_angle<0] += 2*np.pi
#
#     ego_angle = allo_shelter_angle - vars['hd']
#     ego_angle[ego_angle<0] += 2*np.pi
#
#     vars['sx'] = shelter_x[:,0]
#     vars['sy'] = shelter_y[:,0]
#     vars['sd'] = ego_angle
#     vars['x'] = position[:,:,0]
#     vars['y'] = position[:,:,1]
#     vars['xv'] = velocity[:,:,0]
#     vars['yv'] = velocity[:,:,1]
#
#     return vars
#
# def get_vars(config):
#     batch_size, n_timesteps = config.batch_size, config.n_timesteps
#     max_xy, min_xy = config.max_xy, config.min_xy
#
#     if 0<config.stage<1:
#
#         vars = template_0D.get_vars(config)
#
#         vars['x'] = ((max_xy-min_xy)*torch.rand((batch_size,1)) + min_xy).repeat((1,n_timesteps))
#         vars['y'] = ((max_xy-min_xy)*torch.rand((batch_size,1)) + min_xy).repeat((1,n_timesteps))
#
#         vars['sx'] = vars['sx'] + vars['x'][:,0]
#         vars['sy'] = vars['sy'] + vars['y'][:,0]
#
#         vars['xv'] = vars['yv'] = torch.zeros((batch_size, n_timesteps))
#
#     elif 1<=config.stage<2:
#
#         vars = template_1D.get_vars(config)
#
#         vars['sx'] = ((max_xy-min_xy)*torch.rand((batch_size,1)) + min_xy)
#         vars['sy'] = ((max_xy-min_xy)*torch.rand((batch_size,1)) + min_xy)
#
#         vars['y'] = vars['sy'].repeat((1,n_timesteps)) + torch.sin(vars['x'])
#         vars['x'] = vars['sx'].repeat((1,n_timesteps)) + torch.cos(vars['x'])
#
#     elif 2<=config.stage:
#         vars = get_curriculum_vars(config)
#
#     else:
#         vars = template_2D.get_vars(config)
#
#     return vars
#
# def create_data(config, vars, inputs, targets, mask):
#     batch_size, n_timesteps = config.batch_size, config.n_timesteps
#
#     inputs, mask = template_2D.fill_inputs(config, vars, inputs, mask)
#
#     targets[:,:,target_map['sin_sd']] = torch.sin(vars['sd'])
#     targets[:,:,target_map['cos_sd']] = torch.cos(vars['sd'])
#
#     return inputs, targets, mask
#
#
#
# SD_2D_CURRICULUM_TASK = o2s.task.Task('SD-2D_curriculum',
#                                     task_specific_params=default_params, 
#                                     get_vars_func=get_vars,
#                                     create_data_func=create_data,
#                                     input_map=input_map,
#                                     target_map=target_map,
#                                     test_func=o2s.test.test_tuning,
#                                     test_func_args=dict(tuning_vars_list=['HD', 'ego_SD', 'allo_SD', 'AV', 'x', 'y']),
#                                     loss_func=loss_func,
#                                     get_subtask_vars_funcs={'joint': template_2D.get_joint_vars,
#                                                             'hd_iso': template_2D.get_hd_iso_vars,
#                                                             'sd_iso': template_2D.get_sd_iso_vars,
#                                                             'av': template_2D.get_av_vars,
#                                                             'metric': template_2D.get_metric_vars})
#
#
