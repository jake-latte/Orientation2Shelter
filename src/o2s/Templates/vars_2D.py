import torch
import numpy as np
from typing import Dict, Tuple

import o2s
import o2s.Templates.vars_0D as template_0D


default_params = {
    **template_0D.default_params,
    # Standard deviation of noise in angular velocity input
    'v_step_shape': 2,
    'v_step_scale': 0.005,
    'v_step_momentum': 0.3,
    'v_step_zero_prob': 0.5,
    'min_xy': -1,
    'max_xy': 1
}

input_map = {
    'av': 0,
    'sin_hd_0': 1,
    'cos_hd_0': 2,
    'sx': 3,
    'sy': 4,
    'x_0': 5,
    'y_0': 6,
    'v': 7
}


def get_vars(config):
    batch_size, n_timesteps = config.batch_size, config.n_timesteps
    init_duration = config.init_duration
    av_step_std, av_step_momentum = config.av_step_std, config.av_step_momentum
    v_step_shape, v_step_scale, v_step_momentum = config.v_step_shape, config.v_step_scale, config.v_step_momentum
    max_xy, min_xy = config.max_xy, config.min_xy

    HD_0 = (torch.rand(batch_size, 1)) * 2 * np.pi
    HD_0 = torch.tile(HD_0, dims=(1, n_timesteps))
    HD, AV = HD_0, torch.zeros((batch_size, n_timesteps))

    pos_0 = (max_xy-min_xy) * torch.rand((batch_size, 1, 2)) + min_xy
    pos_0 = torch.tile(pos_0, dims=(1, n_timesteps, 1))
    pos, vel = pos_0, torch.zeros((batch_size, n_timesteps))

    zero_trials = torch.rand((batch_size,)) < config.v_step_zero_prob

    normal = torch.distributions.normal.Normal(loc=torch.zeros((batch_size,)), scale=torch.ones((batch_size,))*av_step_std)
    gamma = torch.distributions.gamma.Gamma(concentration=torch.ones((batch_size,))*v_step_shape, rate=torch.ones((batch_size,))/v_step_scale)    
    for t in range(init_duration, n_timesteps):

        av_step = normal.sample() + av_step_momentum * AV[:, t-1]

        AV[:, t] = av_step
        HD[:, t:] += torch.tile(av_step.reshape((batch_size,1)), dims=(1, n_timesteps-t))

        v_step = gamma.sample() + v_step_momentum * vel[:, t-1] 
        if t > n_timesteps*(1/4) and t < n_timesteps*(3/4):
            v_step[zero_trials] = 0

        xv_step = torch.cos(HD[:, t]) * v_step
        yv_step = torch.sin(HD[:, t]) * v_step

        # Compute the maximum scaling factors for x and y bounds
        max_k_x = torch.where(
            xv_step > 0,
            (max_xy - pos[:,t,0]) / xv_step,
            (min_xy - pos[:,t,0]) / xv_step
        )

        max_k_y = torch.where(
            yv_step > 0,
            (max_xy - pos[:,t,1]) / yv_step,
            (min_xy - pos[:,t,1]) / yv_step
        )

        # Replace invalid scaling factors with infinity (e.g., division by zero)
        max_k_x = torch.where(xv_step == 0, torch.full_like(max_k_x, float('inf')), max_k_x)
        max_k_y = torch.where(yv_step == 0, torch.full_like(max_k_y, float('inf')), max_k_y)

        # Compute the maximum scaling factor k for each agent
        max_k = torch.minimum(max_k_x, max_k_y)

        # Ensure k is bounded between 0 and 1 (agents cannot scale beyond their velocity or reverse direction)
        max_k = torch.clamp(max_k, 0, 1)

        v_step = max_k * v_step
        xv_step = torch.cos(HD[:, t]) * v_step
        yv_step = torch.sin(HD[:, t]) * v_step

        vel[:, t] = v_step
        pos[:, t:, 0] += torch.tile(xv_step.reshape((batch_size, 1)), dims=(1, n_timesteps - t))
        pos[:, t:, 1] += torch.tile(yv_step.reshape((batch_size, 1)), dims=(1, n_timesteps - t))

    HD = torch.remainder(HD, 2*np.pi)
    # pos = torch.clamp(pos, min=min_xy, max=max_xy)

    shelter_x_0 = 2*torch.rand(batch_size) - 1
    shelter_y_0 = 2*torch.rand(batch_size) - 1

    shelter_x = torch.tile(shelter_x_0.reshape(batch_size,1), dims=(1,n_timesteps))
    shelter_y = torch.tile(shelter_y_0.reshape(batch_size,1), dims=(1,n_timesteps))

    d_x = shelter_x - pos[:,:,0]
    d_y = shelter_y - pos[:,:,1]
    dist = torch.sqrt(d_x**2 + d_y**2)
    pert = 10e-6 * torch.ones((batch_size,n_timesteps))
    dist[torch.where(dist==0)[0]] += (pert * np.random.choice([1, -1]))[torch.where(dist==0)[0]]
    allo_shelter_angle = torch.atan2(d_y, d_x)
    allo_shelter_angle[allo_shelter_angle<0] += 2*np.pi

    ego_angle = allo_shelter_angle - HD
    ego_angle[ego_angle<0] += 2*np.pi

    return {
        'av': AV,
        'hd': HD,
        'sd': ego_angle,
        'x': pos[:,:,0],
        'y': pos[:,:,1],
        'v': vel,
        'sx': shelter_x[:,0],
        'sy': shelter_y[:,0]
    }



def fill_inputs(config: o2s.config.Config, vars: Dict[str, torch.Tensor], inputs: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    init_duration, batch_size = config.init_duration, inputs.shape[0]

    inputs[:,:,input_map['av']] = vars['av']
    inputs[:,:init_duration,input_map['sin_hd_0']] = torch.sin(vars['hd'][:,:init_duration])
    inputs[:,:init_duration,input_map['cos_hd_0']] = torch.cos(vars['hd'][:,:init_duration])
    inputs[:,:init_duration,input_map['sx']] = vars['sx'].reshape((batch_size,1)).repeat((1,init_duration))
    inputs[:,:init_duration,input_map['sy']] = vars['sy'].reshape((batch_size,1)).repeat((1,init_duration))
    inputs[:,:init_duration,input_map['x_0']] = vars['x'][:,0].reshape((batch_size,1)).repeat((1,init_duration))
    inputs[:,:init_duration,input_map['y_0']] = vars['y'][:,0].reshape((batch_size,1)).repeat((1,init_duration))
    inputs[:,:,input_map['v']] = vars['v']

    mask[:,:init_duration] = False

    return inputs, mask





def get_joint_vars(config):
    batch_size, n_timesteps = config.batch_size, config.n_timesteps
    n_steps = int(np.sqrt(batch_size))
    hd, sd = torch.meshgrid(torch.linspace(0, 2*np.pi, n_steps+1)[:-1], torch.linspace(0, 2*np.pi, n_steps+1)[:-1], indexing='ij')
    hd, sd = hd.reshape((-1,1)).repeat((1,n_timesteps)), sd.reshape((-1,1)).repeat((1,n_timesteps))
    ad = torch.remainder(hd + sd, 2*np.pi)
    vars = {
        'hd': hd,
        'sd': sd,
        'sx': torch.cos(ad[:,0]).reshape((-1,1)),
        'sy': torch.sin(ad[:,0]).reshape((-1,1)),
        'av': torch.zeros_like(hd),
        'x': torch.zeros((batch_size, n_timesteps)),
        'y': torch.zeros((batch_size, n_timesteps)),
        'v': torch.zeros((batch_size, n_timesteps))
    }
    return vars


def get_hd_iso_vars(config):
    batch_size, n_timesteps = config.batch_size, config.n_timesteps
    hd = torch.linspace(0, 2*np.pi, batch_size+1)[:-1].reshape((-1,1)).repeat((1,n_timesteps))
    sd = torch.zeros((batch_size, n_timesteps))
    ad = torch.remainder(hd + sd, 2*np.pi)
    vars = {
        'hd': hd,
        'sd': sd,
        'sx': torch.cos(ad[:,0]).reshape((-1,1)),
        'sy': torch.sin(ad[:,0]).reshape((-1,1)),
        'av': torch.zeros_like(hd),
        'x': torch.zeros((batch_size, n_timesteps)),
        'y': torch.zeros((batch_size, n_timesteps)),
        'v': torch.zeros((batch_size, n_timesteps))
    }
    return vars

def get_sd_iso_vars(config):
    batch_size, n_timesteps = config.batch_size, config.n_timesteps
    hd = torch.zeros((batch_size, n_timesteps))
    sd = torch.linspace(0, 2*np.pi, batch_size+1)[:-1].reshape((-1,1)).repeat((1,n_timesteps))
    ad = torch.remainder(hd + sd, 2*np.pi)
    vars = {
        'hd': hd,
        'sd': sd,
        'sx': torch.cos(ad[:,0]).reshape((-1,1)),
        'sy': torch.sin(ad[:,0]).reshape((-1,1)),
        'av': torch.zeros_like(hd),
        'x': torch.zeros((batch_size, n_timesteps)),
        'y': torch.zeros((batch_size, n_timesteps)),
        'v': torch.zeros((batch_size, n_timesteps))
    }
    return vars

def get_av_vars(config):
    batch_size, n_timesteps = config.batch_size, config.n_timesteps
    n_steps = int(np.sqrt(batch_size))
    hd_0, sd_0 = torch.meshgrid(torch.linspace(0, 2*np.pi, n_steps+1)[:-1], torch.linspace(0, 2*np.pi, n_steps+1)[:-1], indexing='ij')
    hd_0, sd_0 = hd_0.flatten(), sd_0.flatten()
    ad = torch.remainder(hd_0 + sd_0, 2*np.pi).reshape((-1,1)).repeat((1,n_timesteps))

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
        'sx': torch.cos(ad[:,0]).reshape((-1,1)),
        'sy': torch.sin(ad[:,0]).reshape((-1,1)),
        'av': av,
        'x': torch.zeros((batch_size, n_timesteps)),
        'y': torch.zeros((batch_size, n_timesteps)),
        'v': torch.zeros((batch_size, n_timesteps))
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
    vars = {
        'hd': hd.reshape((-1,1)).repeat((1,config.n_timesteps)),
        'sd': sd.reshape((-1,1)).repeat((1,config.n_timesteps)),
        'sx': torch.cos(ad).reshape((-1,1)),
        'sy': torch.sin(ad).reshape((-1,1)),
        'av': torch.zeros((batch_size, n_timesteps)),
        'x': torch.zeros((batch_size, n_timesteps)),
        'y': torch.zeros((batch_size, n_timesteps)),
        'v': torch.zeros((batch_size, n_timesteps))
    }
    return vars