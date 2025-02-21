import torch

import sys

from task import *
from build import *
from test_funcs import *

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
    'av_1': 0,
    'av_2': 1,
    'sin_theta_1': 2,
    'cos_theta_1': 3,
    'sin_theta_2': 4,
    'cos_theta_2': 5
}


def create_data(config, for_training=True):
    # Create local copies of parameter properties (for brevity's sake)
    batch_size, n_timesteps = config.batch_size if for_training else config.test_batch_size, config.n_timesteps if for_training else config.test_n_timesteps
    init_duration = config.init_duration
    av_step_std, av_step_momentum = config.av_step_std, config.av_step_momentum
    
    # Randomly select starting angle for each sequence
    theta_1_0 = (torch.rand(batch_size)) * 2 * np.pi
    theta_2_0 = (torch.rand(batch_size)) * 2 * np.pi

    # Initialise tensors to store the target angle and input angular velocity for each sequence
    theta_1, av_1 = torch.zeros((batch_size, n_timesteps)), torch.zeros((batch_size, n_timesteps))
    theta_2, av_2 = torch.zeros((batch_size, n_timesteps)), torch.zeros((batch_size, n_timesteps))

    zero_trials = torch.where(torch.rand((batch_size,)) < config.av_step_zero_prob)

    normal = torch.distributions.normal.Normal(loc=torch.zeros((batch_size,)), scale=torch.ones((batch_size,))*av_step_std)
    for t in range(init_duration, n_timesteps):
        av_1_step = normal.sample() + av_step_momentum * av_1[:, t-1]
        av_2_step = normal.sample() + av_step_momentum * av_2[:, t-1]

        if t > n_timesteps*(1/4) and t < n_timesteps*(3/4):
            av_1_step[zero_trials] = 0
            av_2_step[zero_trials] = 0

        av_1[:,t] = av_1_step
        av_2[:,t] = av_2_step
    
    # Compute sequence's target angle as its initial angle + integral of angular velocity up to each timestep
    theta_1 = torch.tile(theta_1_0.reshape((batch_size,1)), dims=(1,n_timesteps)) + torch.cumsum(av_1, dim=1)
    theta_1 = torch.remainder(theta_1, 2*np.pi)
    theta_2 = torch.tile(theta_2_0.reshape((batch_size,1)), dims=(1,n_timesteps)) + torch.cumsum(av_2, dim=1)
    theta_2 = torch.remainder(theta_2, 2*np.pi)

    return {'av_1': av_1, 'av_2': av_2, 'theta_1': theta_1, 'theta_2': theta_2}

def fill_inputs(config: Config, inputs: torch.Tensor, mask: torch.Tensor, vars: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    batch_size = inputs.shape[0]
    init_duration = config.init_duration
    
    inputs[:,:,input_map['av_1']] = vars['av_1']
    inputs[:,:,input_map['av_2']] = vars['av_2']
    inputs[:,:init_duration,input_map['sin_theta_1']] = torch.sin(vars['theta_1'][:,0]).reshape((batch_size,1)).repeat((1,init_duration))
    inputs[:,:init_duration,input_map['cos_theta_1']] = torch.cos(vars['theta_1'][:,0]).reshape((batch_size,1)).repeat((1,init_duration))
    inputs[:,:init_duration,input_map['sin_theta_2']] = torch.sin(vars['theta_2'][:,0]).reshape((batch_size,1)).repeat((1,init_duration))
    inputs[:,:init_duration,input_map['cos_theta_2']] = torch.cos(vars['theta_2'][:,0]).reshape((batch_size,1)).repeat((1,init_duration))

    mask[:,:init_duration] = False

    return inputs, mask


target_map = {
    'sin_theta_1': 0,
    'cos_theta_1': 1,
    'sin_theta_2': 2,
    'cos_theta_2': 3
}

def create_data(config, inputs, targets, mask):
    
    vars = create_data(config, for_training=(inputs.shape[0] == config.batch_size and inputs.shape[1] == config.n_timesteps))
    inputs, mask = fill_inputs(config, inputs, mask, vars)

    targets[:,:,target_map['sin_theta_1']] = torch.sin(vars['theta_1'])
    targets[:,:,target_map['cos_theta_1']] = torch.cos(vars['theta_1'])
    targets[:,:,target_map['sin_theta_2']] = torch.sin(vars['theta_2'])
    targets[:,:,target_map['cos_theta_2']] = torch.cos(vars['theta_2'])

    return inputs, targets, vars, mask



HD_AD_0D_TASK = Task('HD_AD-0D',
                    task_specific_params=default_params, 
                    create_data_func=create_data,
                    input_map=input_map,
                    target_map=target_map,
                    test_func=test_general)



