import torch

from typing import Dict, Tuple

from task import *
from build import *
from config import *
from test_funcs import test_allo


default_params = {

    # For task:
    # Number of timesteps at beginning of trial where angular velocity is 0
    'angle_0_duration': 10,
    # Standard deviation of noise in angular velocity input
    'av_step_std': 0.03,
    # Momentum of previous step's angular velocity
    'av_step_momentum': 0.8,
}

input_map = {
    'av': 0,
    'sin_hd_0': 1,
    'cos_hd_0': 2,
    'sx': 3,
    'sy': 4
}


def create_data(config, for_training=True):
    # Create local copies of parameter properties (for brevity's sake)
    batch_size, n_timesteps = config.batch_size if for_training else config.test_batch_size, config.n_timesteps if for_training else config.test_n_timesteps
    angle_0_duration = config.angle_0_duration
    av_step_std, av_step_momentum = config.av_step_std, config.av_step_momentum
    
    # Randomly select starting angle for each sequence
    angle_0 = (torch.rand(batch_size)) * 2 * np.pi

    # Initialise tensors to store the target angle and input angular velocity for each sequence
    angle, angular_velocity = torch.zeros((batch_size, n_timesteps)), torch.zeros((batch_size, n_timesteps))

    normal = torch.distributions.normal.Normal(loc=torch.zeros((batch_size,)), scale=torch.ones((batch_size,))*av_step_std)
    for t in range(angle_0_duration, n_timesteps):

        angular_velocity[:,t] = normal.sample() + av_step_momentum * angular_velocity[:, t-1]
    
    # Compute sequence's target angle as its initial angle + integral of angular velocity up to each timestep
    angle = torch.tile(angle_0.reshape((batch_size,1)), dims=(1,n_timesteps)) + torch.cumsum(angular_velocity, dim=1)
    angle = torch.remainder(angle, 2*np.pi)

    # Initialise allocentric target angle (relative to zero head-direction) for each sequence
    allo_shelter_angle_0 = (torch.rand(batch_size) - 1) * 2 * np.pi
    # Create time-varying allocentric angle as difference between constant allocentric target and
    # current head direction
    ego_sheler_angle = allo_shelter_angle_0.reshape((batch_size,1)).repeat((1,n_timesteps)) - angle
    ego_sheler_angle = torch.remainder(ego_sheler_angle, 2 * np.pi)

    return {'av': angular_velocity, 'hd': angle, 'sd': ego_sheler_angle, 'sx': torch.cos(allo_shelter_angle_0), 'sy': torch.sin(allo_shelter_angle_0)}

def fill_inputs(config: Config, inputs: torch.Tensor, mask: torch.Tensor, vars: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    batch_size = inputs.shape[0]
    angle_0_duration = config.angle_0_duration
    
    inputs[:,:,input_map['av']] = vars['av']
    inputs[:,:angle_0_duration,input_map['sin_hd_0']] = torch.sin(vars['hd'][:,0]).reshape((batch_size,1)).repeat((1,angle_0_duration))
    inputs[:,:angle_0_duration,input_map['cos_hd_0']] = torch.cos(vars['hd'][:,0]).reshape((batch_size,1)).repeat((1,angle_0_duration))
    inputs[:,:angle_0_duration,input_map['sx']] = vars['sx'].reshape((batch_size,1)).repeat((1,angle_0_duration))
    inputs[:,:angle_0_duration,input_map['sy']] = vars['sy'].reshape((batch_size,1)).repeat((1,angle_0_duration))

    mask[:,:angle_0_duration] = False

    return inputs, mask