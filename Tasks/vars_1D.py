import torch

import sys

import Tasks.vars_0D
from ratinabox.Environment import Environment
from ratinabox.Agent import Agent

from task import *
from build import *
from test_funcs import *


default_params = {
    **Tasks.vars_0D.default_params,
    # Standard deviation of noise in angular velocity input
    'xv_step_std': 0.01,
    # Momentum of previous step's angular velocity
    'xv_step_momentum': 0.3,
    'xv_step_hd_bias': 0.005,
    'xv_step_zero_prob': 0.5
}

input_map = {
    'av': 0,
    'sin_hd_0': 1,
    'cos_hd_0': 2,
    'sx': 3,
    'sy': 4,
    'v': 5,
    'x_0': 6
}


def create_data(config, for_training=True):
    # Create local copies of parameter properties (for brevity's sake)
    batch_size, n_timesteps = config.batch_size if for_training else config.test_batch_size, config.n_timesteps if for_training else config.test_n_timesteps
    angle_0_duration = config.angle_0_duration
    xv_step_std, xv_step_momentum, xv_step_hd_bias = config.xv_step_std, config.xv_step_momentum, config.xv_step_hd_bias

    vars = Tasks.vars_0D.create_data(config, for_training=for_training)

    x_0 = torch.rand(batch_size)
    x_velocity = torch.zeros((batch_size, n_timesteps))
    x_position = torch.tile(x_0.reshape(batch_size,1), dims=(1,n_timesteps))

    for t in range(angle_0_duration, n_timesteps):
        normal = torch.distributions.normal.Normal(loc=xv_step_hd_bias*torch.cos(vars['hd'][:,t]), scale=torch.ones((batch_size,))*xv_step_std)
        xv_step = normal.sample() + xv_step_momentum * x_velocity[:,t-1]
        xv_step[torch.rand((batch_size,)) < config.xv_step_zero_prob] = 0

        max_xv_step = 1 - x_position[:, t] 
        min_xv_step = -1 - x_position[:, t]

        xv_step = torch.where(xv_step > max_xv_step, max_xv_step, xv_step)
        xv_step = torch.where(xv_step < min_xv_step, min_xv_step, xv_step)

        x_velocity[:, t] = xv_step
        x_position[:, t:] += torch.tile(xv_step.reshape((batch_size,1)), dims=(1,n_timesteps-t))

    x_position = torch.clamp(x_position, min=-1, max=1)

    shelter_x = torch.tile(torch.rand(batch_size).reshape(batch_size,1), dims=(1,n_timesteps))

    d_x = shelter_x - x_position
    d_y = 0.1 * torch.ones_like(d_x)
    dist = torch.sqrt(d_x**2 + d_y**2)
    pert = 10e-6 * torch.ones((batch_size,n_timesteps))
    dist[torch.where(dist==0)[0]] += (pert * np.random.choice([1, -1]))[torch.where(dist==0)[0]]
    allo_shelter_angle = torch.atan2(d_y, d_x)

    ego_angle = allo_shelter_angle - vars['hd']

    vars['sx'] = shelter_x[:,0]
    vars['sy'] = 0.1* torch.ones((batch_size,))
    vars['sd'] = ego_angle
    vars['x'] = x_position
    vars['v'] = x_velocity

    return vars




