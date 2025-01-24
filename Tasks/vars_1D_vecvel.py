import torch
import numpy as np

from typing import Dict, Tuple

from task import *
from build import *
from config import *
from test_funcs import *

default_params = {
    
    **Tasks.vars_0D.default_params,
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

def create_data(config, for_training=True):
    # Create local copies of parameter properties (for brevity's sake)
    batch_size, n_timesteps = config.batch_size if for_training else config.test_batch_size, config.n_timesteps if for_training else config.test_n_timesteps
    init_duration = config.init_duration
    v_step_std, v_step_momentum, v_step_hd_bias = config.v_step_std, config.v_step_momentum, config.v_step_hd_bias

    vars = Tasks.vars_0D.create_data(config, for_training=for_training)

    pos_0 = 2*np.pi*torch.rand(batch_size)
    velocity = torch.zeros((batch_size, n_timesteps))
    angle = torch.tile(pos_0.reshape(batch_size,1), dims=(1,n_timesteps))

    zero_trials = torch.where(torch.rand((batch_size,)) < config.v_step_zero_prob)

    for t in range(init_duration, n_timesteps):
        normal = torch.distributions.normal.Normal(loc=v_step_hd_bias*(vars['hd'][:,t] - angle[:,t]), scale=torch.ones((batch_size,))*v_step_std)

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


def fill_inputs(config: Config, inputs: torch.Tensor, mask: torch.Tensor, vars: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
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





