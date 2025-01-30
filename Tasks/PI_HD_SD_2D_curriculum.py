import torch

import sys

from task import *
from build import *
from test_funcs import *

import Tasks.vars_0D as template_0D
import Tasks.vars_1D_vecvel as template_1D
import Tasks.vars_2D_vecvel as template_2D

default_params = {
    'stage': 0,
    '0D_loss_threshold': 0.05,
    '1D_loss_threshold': 0.05,
    '2D_loss_threshold': 0.05,


    'v_step_shape': 2,
    'v_step_scale': 0.005,
    'v_step_momentum': 0.3,
    'v_step_zero_prob': 0.5,

    'v_step_std': 0.001,
    'v_step_hd_bias': 0.01,

    'av_step_std': 0.1, 
    'av_step_momentum': 0.5,
    'av_step_zero_prob': 0.5,
    'init_duration': 10
}

input_map = template_2D.input_map

target_map = {
    'x': 0,
    'y': 1,
    'sin_hd': 2,
    'cos_hd': 3,
    'sin_sd': 4,
    'cos_sd': 5
}

def loss_func(task: Task, net: 'RNN', batch: dict) -> Tuple[torch.tensor, torch.tensor]:
    for_training = (batch['inputs'].shape[0] == task.config.batch_size and batch['inputs'].shape[1] == task.config.n_timesteps)

    loss, outputs = default_loss_func(task, net, batch)

    if 0<=task.config.stage<1 and loss.item() < task.config['0D_loss_threshold']:
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

def init_func(task):
    assert task.config.n_timesteps != task.config.test_n_timesteps or task.config.batch_size != task.config.test_batch_size, 'test and training batches must be different'







def create_curriculum_vars(config, for_training=True):
    batch_size, n_timesteps = config.batch_size if for_training else config.test_batch_size, config.n_timesteps if for_training else config.test_n_timesteps
    init_duration = config.init_duration
    v_step_std, v_step_momentum, v_step_hd_bias, v_step_zero_prob = config.v_step_std, config.v_step_momentum, config.v_step_hd_bias, config.v_step_zero_prob


    vars = template_0D.create_data(config, for_training=for_training)

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

def create_data(config, inputs, targets, mask):
    for_training = (inputs.shape[0] == config.batch_size and inputs.shape[1] == config.n_timesteps)
    batch_size, n_timesteps = config.batch_size if for_training else config.test_batch_size, config.n_timesteps if for_training else config.test_n_timesteps

    vars = {}
    if for_training:
        if 0<=config.stage<1:

            vars = template_0D.create_data(config)

            vars['x'] = (2*torch.rand((batch_size,1)) - 1).repeat((1,n_timesteps))
            vars['y'] = (2*torch.rand((batch_size,1)) - 1).repeat((1,n_timesteps))

            vars['sx'] = vars['sx'] + vars['x'][:,0]
            vars['sy'] = vars['sy'] + vars['y'][:,0]

            vars['xv'] = vars['yv'] = torch.zeros((batch_size, n_timesteps))

        elif 1<=config.stage<2:

            vars = template_1D.create_data(config)

            vars['sx'] = 2*torch.rand((batch_size,1)) - 1
            vars['sy'] = 2*torch.rand((batch_size,1)) - 1

            vars['y'] = vars['sy'].repeat((1,n_timesteps)) + torch.sin(vars['x'])
            vars['x'] = vars['sx'].repeat((1,n_timesteps)) + torch.cos(vars['x'])

        else:
            vars = create_curriculum_vars(config)

    else:
        vars = template_2D.create_data(config, for_training=False)


    inputs, mask = template_2D.fill_inputs(config, inputs, mask, vars)

    targets[:,:,target_map['x']] = vars['x']
    targets[:,:,target_map['y']] = vars['y']
    targets[:,:,target_map['sin_hd']] = torch.sin(vars['hd'])
    targets[:,:,target_map['cos_hd']] = torch.cos(vars['hd'])
    targets[:,:,target_map['sin_sd']] = torch.sin(vars['sd'])
    targets[:,:,target_map['cos_sd']] = torch.cos(vars['sd'])

    return inputs, targets, vars, mask



PI_HD_SD_2D_CURRICULUM_TASK = Task('PI_HD_SD-2D_curriculum',
                                    task_specific_params=default_params, 
                                    create_data_func=create_data,
                                    input_map=input_map,
                                    target_map=target_map,
                                    test_func=test_tuning,
                                    test_func_args=dict(tuning_vars_list=['HD', 'ego_SD', 'allo_SD', 'AV', 'x', 'y']),
                                    loss_func=loss_func,
                                    init_func=init_func)



