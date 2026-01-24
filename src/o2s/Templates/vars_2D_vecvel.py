import torch
import numpy as np
from typing import Dict, Tuple

import o2s
from o2s.Templates.vars_2D_linvel import default_params
from o2s.Templates.vars_2D_linvel import get_vars as linvel_get_vars



input_map = {
    'av': 0,
    'sin_hd_0': 1,
    'cos_hd_0': 2,
    'sx': 3,
    'sy': 4,
    'x_0': 5,
    'y_0': 6,
    'xv': 7,
    'yv': 8
}

def get_vars(config):

    vars = linvel_get_vars(config)

    hd = vars['hd']
    v = vars['v']
    del vars['v']

    vars['xv'] = v * torch.cos(hd)
    vars['yv'] = v * torch.sin(hd)

    return vars

def fill_inputs(config: o2s.config.Config, vars: Dict[str, torch.Tensor], inputs: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    init_duration, batch_size = config.init_duration, inputs.shape[0]

    inputs[:,:,input_map['av']] = vars['av']
    inputs[:,:init_duration,input_map['sin_hd_0']] = torch.sin(vars['hd'][:,:init_duration])
    inputs[:,:init_duration,input_map['cos_hd_0']] = torch.cos(vars['hd'][:,:init_duration])
    inputs[:,:init_duration,input_map['sx']] = vars['sx'].reshape((batch_size,1)).repeat((1,init_duration))
    inputs[:,:init_duration,input_map['sy']] = vars['sy'].reshape((batch_size,1)).repeat((1,init_duration))
    inputs[:,:init_duration,input_map['x_0']] = vars['x'][:,0].reshape((batch_size,1)).repeat((1,init_duration))
    inputs[:,:init_duration,input_map['y_0']] = vars['y'][:,0].reshape((batch_size,1)).repeat((1,init_duration))
    inputs[:,:,input_map['xv']] = vars['xv']
    inputs[:,:,input_map['yv']] = vars['yv']

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
        'xv': torch.zeros((batch_size, n_timesteps)),
        'yv': torch.zeros((batch_size, n_timesteps))
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
        'xv': torch.zeros((batch_size, n_timesteps)),
        'yv': torch.zeros((batch_size, n_timesteps))
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
        'xv': torch.zeros((batch_size, n_timesteps)),
        'yv': torch.zeros((batch_size, n_timesteps))
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
        'xv': torch.zeros((batch_size, n_timesteps)),
        'yv': torch.zeros((batch_size, n_timesteps))
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
        'xv': torch.zeros((batch_size, n_timesteps)),
        'yv': torch.zeros((batch_size, n_timesteps))
    }
    return vars

class Vars2DVecvel(o2s.task.Task):
    input_map = input_map
    get_vars_func = staticmethod(get_vars)
    fill_inputs = staticmethod(fill_inputs)
    get_joint_vars = staticmethod(get_joint_vars)
    get_hd_iso_vars = staticmethod(get_hd_iso_vars)
    get_sd_iso_vars = staticmethod(get_sd_iso_vars)
    get_av_vars = staticmethod(get_av_vars)
    get_metric_vars = staticmethod(get_metric_vars)
