import torch

from typing import Dict, Tuple

from task import *
from build import *
from config import *
from test_funcs import *


import Tasks.vars_0D as template_0D


default_params = {
    **template_0D.default_params,
    # Standard deviation of noise in angular velocity input
    'v_step_std': 0.01,
    # Momentum of previous step's angular velocity
    'v_step_momentum': 0.3,
    'v_step_hd_bias': 0.005,
    'v_step_zero_prob': 0.5
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
    init_duration = config.init_duration
    v_step_std, v_step_momentum, v_step_hd_bias = config.v_step_std, config.v_step_momentum, config.v_step_hd_bias

    vars = template_0D.create_data(config, for_training=for_training)

    x_0 = torch.rand(batch_size)
    x_velocity = torch.zeros((batch_size, n_timesteps))
    x_position = torch.tile(x_0.reshape(batch_size,1), dims=(1,n_timesteps))

    zero_trials = torch.where(torch.rand((batch_size,)) < config.v_step_zero_prob)

    for t in range(init_duration, n_timesteps):
        normal = torch.distributions.normal.Normal(loc=v_step_hd_bias*torch.cos(vars['hd'][:,t]), scale=torch.ones((batch_size,))*v_step_std)

        v_step = normal.sample() + v_step_momentum * x_velocity[:,t-1]
        if t > n_timesteps*(1/4) and t < n_timesteps*(3/4):
            v_step[zero_trials] = 0

        max_v_step = 1 - x_position[:, t] 
        min_v_step = -1 - x_position[:, t]

        v_step = torch.where(v_step > max_v_step, max_v_step, v_step)
        v_step = torch.where(v_step < min_v_step, min_v_step, v_step)



        x_velocity[:, t] = v_step
        x_position[:, t:] += torch.tile(v_step.reshape((batch_size,1)), dims=(1,n_timesteps-t))

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

def fill_inputs(config: Config, inputs: torch.Tensor, mask: torch.Tensor, vars: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    batch_size = inputs.shape[0]
    init_duration = config.init_duration
    
    inputs[:,:,input_map['av']] = vars['av']
    inputs[:,:init_duration,input_map['sin_hd_0']] = torch.sin(vars['hd'][:,0]).reshape((batch_size,1)).repeat((1,init_duration))
    inputs[:,:init_duration,input_map['cos_hd_0']] = torch.cos(vars['hd'][:,0]).reshape((batch_size,1)).repeat((1,init_duration))
    inputs[:,:init_duration,input_map['sx']] = vars['sx'].reshape((batch_size,1)).repeat((1,init_duration))
    inputs[:,:init_duration,input_map['sy']] = vars['sy'].reshape((batch_size,1)).repeat((1,init_duration))
    inputs[:,:,input_map['v']] = vars['v']
    inputs[:,:init_duration,input_map['x_0']] = vars['x'][:,0].reshape((batch_size, 1)).repeat((1, init_duration))

    mask[:,:init_duration] = False

    return inputs, mask




