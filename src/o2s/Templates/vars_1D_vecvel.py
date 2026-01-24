import torch
import numpy as np

from typing import Dict, Tuple

import o2s
import o2s.Templates.vars_0D as template_0D

default_params = {
    
    **template_0D.default_params,
    # Standard deviation of noise in angular velocity input
    'v_step_std': 0.01,
    # Momentum of previous step's angular velocity
    'v_step_momentum': 0.6,
    'v_step_hd_bias': 0.01,
    'v_step_zero_prob': 0.5
}

input_map = {
    'av': 0,
    'sin_hd_0': 1,
    'cos_hd_0': 2,
    'xv': 3,
    'yv': 4,
    'sin_x_0': 5,
    'cos_x_0': 6,
    'sx': 7,
    'sy': 8
}

def get_vars(config):
    batch_size, n_timesteps = config.batch_size, config.n_timesteps
    init_duration = config.init_duration
    v_step_std, v_step_momentum, v_step_hd_bias = config.v_step_std, config.v_step_momentum, config.v_step_hd_bias

    vars = template_0D.get_vars(config)

    pos_0 = 2*np.pi*torch.rand(batch_size)
    velocity = torch.zeros((batch_size, n_timesteps))
    angle = torch.tile(pos_0.reshape(batch_size,1), dims=(1,n_timesteps))

    zero_trials = torch.where(torch.rand((batch_size,)) < config.v_step_zero_prob)

    normal = torch.distributions.normal.Normal(loc=torch.zeros((batch_size,)), scale=torch.ones((batch_size,))*v_step_std)
    for t in range(init_duration, n_timesteps):
        # normal = torch.distributions.normal.Normal(loc=v_step_hd_bias*(vars['hd'][:,t] - angle[:,t]), scale=torch.ones((batch_size,))*v_step_std)

        v_step = normal.sample() + v_step_momentum * velocity[:,t-1]
        if t > n_timesteps*(1/4) and t < n_timesteps*(3/4):
            v_step[zero_trials] = 0

        velocity[:, t] = v_step
        angle[:, t:] += torch.tile(v_step.reshape((batch_size,1)), dims=(1,n_timesteps-t))

    x_velocity = torch.cat((torch.zeros((batch_size,1)), torch.diff(torch.cos(angle), dim=1)), dim=1)
    y_velocity = torch.cat((torch.zeros((batch_size,1)), torch.diff(torch.sin(angle), dim=1)), dim=1)
    angle = torch.remainder(angle, 2*np.pi)

    allo_shelter_angle = torch.remainder(angle - np.pi, 2*np.pi)
    ego_angle = torch.remainder(allo_shelter_angle - vars['hd'], 2*np.pi)

    vars['sx'] = torch.zeros((batch_size,))
    vars['sy'] = torch.zeros((batch_size,))
    vars['sd'] = ego_angle
    vars['x'] = angle
    vars['xv'] = x_velocity
    vars['yv'] = y_velocity

    return vars


def fill_inputs(config: o2s.config.Config, vars: Dict[str, torch.Tensor], inputs: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    batch_size = inputs.shape[0]
    init_duration = config.init_duration
    
    inputs[:,:,input_map['av']] = vars['av']
    inputs[:,:init_duration,input_map['sin_hd_0']] = torch.sin(vars['hd'][:,0]).reshape((batch_size,1)).repeat((1,init_duration))
    inputs[:,:init_duration,input_map['cos_hd_0']] = torch.cos(vars['hd'][:,0]).reshape((batch_size,1)).repeat((1,init_duration))
    inputs[:,:,input_map['xv']] = vars['xv']
    inputs[:,:,input_map['yv']] = vars['yv']
    inputs[:,:init_duration,input_map['sin_x_0']] = torch.sin(vars['x'][:,0].reshape((batch_size, 1)).repeat((1, init_duration)))
    inputs[:,:init_duration,input_map['cos_x_0']] = torch.cos(vars['x'][:,0].reshape((batch_size, 1)).repeat((1, init_duration)))
    inputs[:,:init_duration,input_map['sx']] = vars['sx'].reshape((batch_size,1)).repeat((1,init_duration))
    inputs[:,:init_duration,input_map['sy']] = vars['sy'].reshape((batch_size,1)).repeat((1,init_duration))

    mask[:,:init_duration] = False

    return inputs, mask



def get_joint_vars(config):
    batch_size, n_timesteps = config.batch_size, config.n_timesteps
    n_steps = int(np.sqrt(batch_size))
    hd, sd = torch.meshgrid(torch.linspace(0, 2*np.pi, n_steps+1)[:-1], torch.linspace(0, 2*np.pi, n_steps+1)[:-1], indexing='ij')
    hd, sd = hd.reshape((-1,1)).repeat((1,n_timesteps)), sd.reshape((-1,1)).repeat((1,n_timesteps))
    ad = torch.remainder(hd + sd, 2*np.pi)
    x = torch.remainder(ad + np.pi, 2*np.pi)
    vars = {
        'hd': hd,
        'sd': sd,
        'sx': torch.zeros((batch_size,1)),
        'sy': torch.zeros((batch_size,1)),
        'x': x,
        'av': torch.zeros_like(hd),
        'xv': torch.zeros_like(hd),
        'yv': torch.zeros_like(hd)
    }
    return vars


def get_hd_iso_vars(config):
    batch_size, n_timesteps = config.batch_size, config.n_timesteps
    hd = torch.linspace(0, 2*np.pi, batch_size+1)[:-1].reshape((-1,1)).repeat((1,n_timesteps))
    sd = torch.zeros((batch_size, n_timesteps))
    ad = torch.remainder(hd + sd, 2*np.pi)
    x = torch.remainder(ad + np.pi, 2*np.pi)
    vars = {
        'hd': hd,
        'sd': sd,
        'sx': torch.zeros((batch_size,1)),
        'sy': torch.zeros((batch_size,1)),
        'x': x,
        'av': torch.zeros_like(hd),
        'xv': torch.zeros_like(hd),
        'yv': torch.zeros_like(hd)
    }
    return vars


def get_sd_iso_vars(config):
    batch_size, n_timesteps = config.batch_size, config.n_timesteps
    hd = torch.zeros((batch_size, n_timesteps))
    sd = torch.linspace(0, 2*np.pi, batch_size+1)[:-1].reshape((-1,1)).repeat((1,n_timesteps))
    ad = torch.remainder(hd + sd, 2*np.pi)
    x = torch.remainder(ad + np.pi, 2*np.pi)
    vars = {
        'hd': hd,
        'sd': sd,
        'sx': torch.zeros((batch_size,1)),
        'sy': torch.zeros((batch_size,1)),
        'x': x,
        'av': torch.zeros_like(hd),
        'xv': torch.zeros_like(hd),
        'yv': torch.zeros_like(hd)
    }
    return vars

def get_hd_iso_vel_vars(config):

    batch_size, n_timesteps = config.batch_size, config.n_timesteps
    sd = torch.zeros((batch_size, n_timesteps))
    x_0 = torch.linspace(0, 2*np.pi, batch_size+1)[:-1].reshape((-1,1)).repeat((1,n_timesteps))

    period_length = (n_timesteps - config.init_duration) // 5
    init_mov = torch.zeros((batch_size, config.init_duration))
    null_mov = torch.zeros((batch_size, period_length))
    cw_mov = torch.ones((batch_size, period_length)) * np.pi / period_length
    ccw_mov = torch.ones((batch_size, period_length)) * -np.pi / period_length
    mov = torch.cat((init_mov, null_mov, cw_mov, null_mov, ccw_mov, null_mov), dim=1)
    
    x = x_0 + torch.cumsum(mov, dim=1)
    xv = torch.cat((torch.zeros((batch_size,1)), torch.diff(torch.cos(x), dim=1)), dim=1)
    yv = torch.cat((torch.zeros((batch_size,1)), torch.diff(torch.sin(x), dim=1)), dim=1)
    xv = torch.cat((torch.zeros((batch_size,1)), torch.diff(torch.cos(x), dim=1)), dim=1)
    yv = torch.cat((torch.zeros((batch_size,1)), torch.diff(torch.sin(x), dim=1)), dim=1)

    hd = x + np.pi
    av = torch.cat((torch.zeros((batch_size,1)), torch.diff(hd, dim=1)), dim=1)
    av = torch.cat((torch.zeros((batch_size,1)), torch.diff(hd, dim=1)), dim=1)

    vars = {
        'hd': torch.remainder(hd, 2*np.pi),
        'sd': sd,
        'sx': torch.zeros((batch_size,1)),
        'sy': torch.zeros((batch_size,1)),
        'x': torch.remainder(x, 2*np.pi),
        'av': av,
        'xv': xv,
        'yv': yv
    }
    return vars



def get_sd_iso_vel_vars(config):
    batch_size, n_timesteps = config.batch_size, config.n_timesteps
    hd = torch.zeros((batch_size, n_timesteps))
    x_0 = torch.linspace(0, 2*np.pi, batch_size+1)[:-1].reshape((-1,1)).repeat((1,n_timesteps))

    period_length = (n_timesteps - config.init_duration) // 5
    init_mov = torch.zeros((batch_size, config.init_duration))
    null_mov = torch.zeros((batch_size, period_length))
    cw_mov = torch.ones((batch_size, period_length)) * np.pi / period_length
    ccw_mov = torch.ones((batch_size, period_length)) * -np.pi / period_length
    mov = torch.cat((init_mov, null_mov, cw_mov, null_mov, ccw_mov, null_mov), dim=1)
    
    x = x_0 + torch.cumsum(mov, dim=1)
    xv = torch.cat((torch.zeros((batch_size,1)), torch.diff(torch.cos(x), dim=1)), dim=1)
    yv = torch.cat((torch.zeros((batch_size,1)), torch.diff(torch.sin(x), dim=1)), dim=1)

    sd = x + np.pi

    vars = {
        'hd': hd,
        'sd': torch.remainder(sd, 2*np.pi),
        'sx': torch.zeros((batch_size,1)),
        'sy': torch.zeros((batch_size,1)),
        'x': torch.remainder(x, 2*np.pi),
        'av': torch.zeros((batch_size, n_timesteps)),
        'xv': xv,
        'yv': yv
    }
    return vars



def get_av_vars(config):
    batch_size, n_timesteps = config.batch_size, config.n_timesteps
    n_steps = int(np.sqrt(batch_size))
    hd_0, sd_0 = torch.meshgrid(torch.linspace(0, 2*np.pi, n_steps+1)[:-1], torch.linspace(0, 2*np.pi, n_steps+1)[:-1], indexing='ij')
    hd_0, sd_0 = hd_0.flatten(), sd_0.flatten()
    ad = torch.remainder(hd_0 + sd_0, 2*np.pi).reshape((-1,1)).repeat((1,n_timesteps))
    x = torch.remainder(ad + np.pi, 2*np.pi)

    period_length = (n_timesteps - config.init_duration) // 5
    init_rot = torch.zeros((batch_size, config.init_duration))
    null_rot = torch.zeros((batch_size, period_length))
    cw_rot = torch.ones((batch_size, period_length)) * np.pi / period_length
    ccw_rot = torch.ones((batch_size, period_length)) * -np.pi / period_length
    av = torch.cat((init_rot, null_rot, cw_rot, null_rot, ccw_rot, null_rot), dim=1)

    hd = torch.remainder(hd_0.reshape((-1,1)).repeat((1,n_timesteps)) + torch.cumsum(av, dim=1), 2*np.pi)
    sd = torch.remainder(ad - hd, 2*np.pi)

    vars = {
        'hd': hd,
        'sd': sd,
        'sx': torch.zeros((batch_size,1)),
        'sy': torch.zeros((batch_size,1)),
        'x': x,
        'av': av,
        'xv': torch.zeros_like(hd),
        'yv': torch.zeros_like(hd)
    }
    return vars


def get_metric_vars(config):
    batch_size, n_timesteps, init_duration = config.batch_size, config.n_timesteps, config.init_duration
    n_steps = int(np.sqrt(batch_size))
    dtheta_1, dtheta_2 = config.dtheta_1, config.dtheta_2
    theta_2_is_SD = config.theta_2_is_SD

    theta_1, theta_2 = torch.linspace(dtheta_1, 2*np.pi+dtheta_1, n_steps+1)[:-1], torch.linspace(dtheta_2, 2*np.pi+dtheta_2, n_steps+1)[:-1]
    theta_1, theta_2 = torch.meshgrid(theta_1, theta_2, indexing='ij')
    theta_1, theta_2 = torch.remainder(theta_1.flatten(), 2*np.pi), torch.remainder(theta_2.flatten(), 2*np.pi)

    hd = theta_1
    sd = theta_2 if theta_2_is_SD else torch.remainder(theta_2 - theta_1, 2*np.pi)
    ad = torch.remainder(theta_1 + theta_2, 2*np.pi) if theta_2_is_SD else theta_2
    x = torch.remainder(ad + np.pi, 2*np.pi)
    vars = {
        'hd': hd.reshape((-1,1)).repeat((1,config.n_timesteps)),
        'sd': sd.reshape((-1,1)).repeat((1,config.n_timesteps)),
        'sx': torch.zeros((batch_size,1)),
        'sy': torch.zeros((batch_size,1)),
        'x': x.reshape((-1,1)).repeat((1,config.n_timesteps)),
        'av': torch.zeros((batch_size, n_timesteps)),
        'xv': torch.zeros((batch_size, n_timesteps)),
        'yv': torch.zeros((batch_size, n_timesteps))
    }
    return vars