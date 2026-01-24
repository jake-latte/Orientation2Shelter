import torch
import numpy as np
from typing import Dict, Tuple

import o2s


default_params = {

    # For task:
    # Number of timesteps at beginning of trial where angular velocity is 0
    'init_duration': 10,
    # Standard deviation of noise in angular velocity input
    'av_step_std': 0.03,
    # Momentum of previous step's angular velocity
    'av_step_momentum': 0.8,
    'av_step_zero_prob': 0.5
}

input_map = {
    'av': 0,
    'sin_hd_0': 1,
    'cos_hd_0': 2,
    'sx': 3,
    'sy': 4
}


def get_vars(config):
    # Create local copies of parameter properties (for brevity's sake)
    batch_size, n_timesteps = config.batch_size, config.n_timesteps
    init_duration = config.init_duration
    av_step_std, av_step_momentum = config.av_step_std, config.av_step_momentum
    
    # Randomly select starting angle for each sequence
    angle_0 = (torch.rand(batch_size)) * 2 * np.pi

    # Initialise tensors to store the target angle and input angular velocity for each sequence
    angle, angular_velocity = torch.zeros((batch_size, n_timesteps)), torch.zeros((batch_size, n_timesteps))

    zero_trials = torch.where(torch.rand((batch_size,)) < config.av_step_zero_prob)

    normal = torch.distributions.normal.Normal(loc=torch.zeros((batch_size,)), scale=torch.ones((batch_size,))*av_step_std)
    for t in range(init_duration, n_timesteps):
        av_step = normal.sample() + av_step_momentum * angular_velocity[:, t-1]

        if t > n_timesteps*(1/4) and t < n_timesteps*(3/4):
            av_step[zero_trials] = 0

        angular_velocity[:,t] = av_step
    
    # Compute sequence's target angle as its initial angle + integral of angular velocity up to each timestep
    angle = torch.tile(angle_0.reshape((batch_size,1)), dims=(1,n_timesteps)) + torch.cumsum(angular_velocity, dim=1)
    angle = torch.remainder(angle, 2*np.pi)

    # Initialise allocentric target angle (relative to zero head-direction) for each sequence
    allo_shelter_angle_0 = (torch.rand(batch_size) - 1) * 2 * np.pi
    # Create time-varying allocentric angle as difference between constant allocentric target and
    # current head direction
    ego_sheler_angle = allo_shelter_angle_0.reshape((batch_size,1)).repeat((1,n_timesteps)) - angle
    ego_sheler_angle = torch.remainder(ego_sheler_angle, 2 * np.pi)

    dtype = torch.float64 if config.precise else torch.float32
    return {'av': angular_velocity.type(dtype), 
            'hd': angle.type(dtype), 
            'sd': ego_sheler_angle.type(dtype), 
            'sx': torch.cos(allo_shelter_angle_0).type(dtype), 
            'sy': torch.sin(allo_shelter_angle_0).type(dtype)}

def fill_inputs(config: o2s.config.Config, vars: Dict[str, torch.Tensor], inputs: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    batch_size, n_timesteps = inputs.shape[0], inputs.shape[1]
    init_duration = config.init_duration
    
    inputs[:,:,input_map['av']] = vars['av']
    inputs[:,:init_duration,input_map['sin_hd_0']] = torch.sin(vars['hd'][:,0]).reshape((batch_size,1)).repeat((1,init_duration))
    inputs[:,:init_duration,input_map['cos_hd_0']] = torch.cos(vars['hd'][:,0]).reshape((batch_size,1)).repeat((1,init_duration))
    inputs[:,:init_duration,input_map['sx']] = vars['sx'].reshape((batch_size,1)).repeat((1,init_duration))
    inputs[:,:init_duration,input_map['sy']] = vars['sy'].reshape((batch_size,1)).repeat((1,init_duration))

    mask[:,:init_duration] = False

    return inputs, mask






def get_joint_vars(config):
    batch_size, n_timesteps = config.batch_size, config.n_timesteps
    shuffle = False if 'shuffle' not in config.dict else config.shuffle

    if shuffle:
        hd = torch.rand((batch_size,1)).repeat((1,n_timesteps)) * 2 * np.pi
        sd = torch.rand((batch_size,1)).repeat((1,n_timesteps)) * 2 * np.pi
        ad = torch.remainder(hd + sd, 2*np.pi)
    else:
        n_steps = int(np.sqrt(batch_size))
        hd, sd = torch.meshgrid(torch.linspace(0, 2*np.pi, n_steps+1)[:-1], torch.linspace(0, 2*np.pi, n_steps+1)[:-1], indexing='ij')
        hd, sd = hd.reshape((-1,1)).repeat((1,n_timesteps)), sd.reshape((-1,1)).repeat((1,n_timesteps))
        ad = torch.remainder(hd + sd, 2*np.pi)
    vars = {
        'hd': hd,
        'sd': sd,
        'sx': torch.cos(ad[:,0]).reshape((-1,1)),
        'sy': torch.sin(ad[:,0]).reshape((-1,1)),
        'av': torch.zeros_like(hd)
    }
    return vars


def get_hd_iso_vars(config):
    batch_size, n_timesteps = config.batch_size, config.n_timesteps
    fix_sd = 0 if 'fix_sd' not in config.dict else config.fix_sd
    hd = torch.linspace(0, 2*np.pi, batch_size+1)[:-1].reshape((-1,1)).repeat((1,n_timesteps))
    sd = torch.ones((batch_size, n_timesteps)) * fix_sd
    ad = torch.remainder(hd + sd, 2*np.pi)
    vars = {
        'hd': hd,
        'sd': sd,
        'sx': torch.cos(ad[:,0]).reshape((-1,1)),
        'sy': torch.sin(ad[:,0]).reshape((-1,1)),
        'av': torch.zeros_like(hd)
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
        'av': torch.zeros_like(hd)
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
        'av': av
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
        'av': torch.zeros((batch_size, n_timesteps))
    }
    return vars

class Vars0D(o2s.task.Task):
    default_params = default_params
    input_map = input_map
    get_vars_func = staticmethod(get_vars)
    fill_inputs = staticmethod(fill_inputs)
    get_joint_vars = staticmethod(get_joint_vars)
    get_hd_iso_vars = staticmethod(get_hd_iso_vars)
    get_sd_iso_vars = staticmethod(get_sd_iso_vars)
    get_av_vars = staticmethod(get_av_vars)
    get_metric_vars = staticmethod(get_metric_vars)
