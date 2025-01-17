import torch

import sys

from ratinabox.Environment import Environment
from ratinabox.Agent import Agent

from task import *
from build import *
from test_funcs import *


default_params = {
    **Tasks.vars_0D.default_params,
    # Standard deviation of noise in angular velocity input
    'v_step_shape': 2,
    'v_step_scale': 0.005,
    'v_step_momentum': 0.3,
    'v_step_hd_bias': 0.005,
    'v_step_zero_prob': 0.5,
    'n_timesteps': 1000
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


def create_data(config, for_training=True):
    # Create local copies of parameter properties (for brevity's sake)
    batch_size, n_timesteps = config.batch_size if for_training else config.test_batch_size, config.n_timesteps if for_training else config.test_n_timesteps
    angle_0_duration = config.angle_0_duration
    av_step_std, av_step_momentum = config.av_step_std, config.av_step_momentum
    v_step_shape, v_step_scale, v_step_momentum = config.v_step_shape, config.v_step_scale, config.v_step_momentum

    HD_0 = (torch.rand(batch_size, 1)) * 2 * np.pi
    HD_0 = torch.tile(HD_0, dims=(1, n_timesteps))
    HD, AV = HD_0, torch.zeros((batch_size, n_timesteps))

    pos_0 = torch.rand((batch_size, 1, 2)) * 2 - 1
    pos_0 = torch.tile(pos_0, dims=(1, n_timesteps, 1))
    pos, vel = pos_0, torch.zeros((batch_size, n_timesteps))

    zero_trials = torch.where(torch.rand((batch_size,)) < config.v_step_zero_prob)

    normal = torch.distributions.normal.Normal(loc=torch.zeros((batch_size,)), scale=torch.ones((batch_size,))*av_step_std)
    gamma = torch.distributions.gamma.Gamma(concentration=torch.ones((batch_size,))*v_step_shape, rate=torch.ones((batch_size,))/v_step_scale)    
    for t in range(angle_0_duration, n_timesteps):

        max_xv_step = 1 - pos[:, t, 0]
        min_xv_step = -1 - pos[:, t, 0]
        max_yv_step = 1 - pos[:, t, 1]
        min_yv_step = -1 - pos[:, t, 1]

        av_step = normal.sample() + av_step_momentum * AV[:, t-1]

        AV[:, t] = av_step
        HD[:, t:] += torch.tile(av_step.reshape((batch_size,1)), dims=(1, n_timesteps-t))

        v_step = gamma.sample() + v_step_momentum * vel[:, t-1] 
        # if t > n_timesteps*(1/4) and t < n_timesteps*(3/4):
        #     v_step[zero_trials] = 0

        xv_step = torch.cos(HD[:, t]) * v_step
        yv_step = torch.sin(HD[:, t]) * v_step

        xv_step = torch.where(xv_step > max_xv_step, torch.ones((batch_size,)), xv_step)
        xv_step = torch.where(xv_step < min_xv_step, -torch.ones((batch_size,)), xv_step)
        yv_step = torch.where(yv_step > max_yv_step, torch.ones((batch_size,)), yv_step)
        yv_step = torch.where(yv_step < min_yv_step, -torch.ones((batch_size,)), yv_step)

        vel[:, t] = torch.sqrt(xv_step**2 + yv_step**2)
        pos[:, t:, 0] += torch.tile(xv_step.reshape((batch_size, 1)), dims=(1, n_timesteps - t))
        pos[:, t:, 1] += torch.tile(yv_step.reshape((batch_size, 1)), dims=(1, n_timesteps - t))

    HD = torch.remainder(HD, 2*np.pi)
    pos = torch.clamp(pos, min=-1, max=1)

    shelter_x_0 = torch.rand(batch_size)
    shelter_y_0 = torch.rand(batch_size)
    shelter_wall_select_probs = torch.rand(batch_size)
    shelter_wall_select_horizontal = torch.where(shelter_wall_select_probs>0.5)[0]
    shelter_wall_select_vertical = torch.where(shelter_wall_select_probs<=0.5)[0]
    shelter_wall_select_directions = torch.randint(low=0, high=2, size=(batch_size,), dtype=shelter_x_0.dtype) * 2 - 1

    shelter_x_0[shelter_wall_select_vertical] = shelter_wall_select_directions[shelter_wall_select_vertical]
    shelter_y_0[shelter_wall_select_horizontal] = shelter_wall_select_directions[shelter_wall_select_horizontal]

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





