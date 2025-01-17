import torch

import sys

from task import *
from build import *
from test_funcs import test_allo


default_params = {

    # For task:
    # Number of timesteps at beginning of trial where angular velocity is 0
    'angle_0_duration': 10,
    # Standard deviation of noise in angular velocity input
    'av_step_std': 0.03,
    # Momentum of previous step's angular velocity
    'av_step_momentum': 0.8,

    # For tuning analysis:
    # Number of bins in which to discretize target/model angle
    'n_angle_bins': 360,
    # Number of bins in which to discretize input angular velocity
    'n_AV_bins': 100,
    # Number of standard deviations of input angular velociy to keep in those bins
    'n_AV_std': 3,
    # Maximum differential between neuron maximum and minimum angle-tuned activity for classifying as untuned
    # (implicit minimum at 0)
    'max_dif_for_untuned': 0.1,
    # Maximum (absolute) slope of neuron velocity-tuned activity for classifying as untuned
    'max_slope_for_untuned': 0.003,
    # Maximum (absolute) slope of neuron velocity-tuned activity for classifying as compass
    'max_slope_for_compass': 0.005,
    # Minimum differential between neuron maximum and minimum angle-tuned activity for classifying as compass
    'min_dif_for_compass': 0.4,
    # Maximum (absolute) slope of neuron velocity-tuned activity for classifying as weakly tuned
    'max_slope_for_weakly_tuned': 0.005,
    # Minimum differential between neuron maximum and minimum angle-tuned activity for classifying as weakly tuned
    'min_dif_for_weakly_tuned': 0.1,
    # Maximum differential between neuron maximum and minimum angle-tuned activity for classifying as weakly tuned
    'max_dif_for_weakly_tuned': 0.3,
    # Number of timesteps to use in lesion analysis
    'n_lesion_timesteps': 500,
    # Number of ignored timesteps at beginnning of each lesion trial to account for transient activity
    'n_lesion_transient': 50,
}

input_map = {
    'av': 0,
    'sin_hd_0': 1,
    'cos_hd_0': 2
}

target_map = {
    'sin_hd': 0,
    'cos_hd': 1
}


def create_data(config, inputs, targets, mask):
    # Make local copies of parameter properties (just for brevity's sake)
    batch_size, n_timesteps = inputs.shape[0], inputs.shape[1]
    angle_0_duration = config.angle_0_duration
    av_step_std, av_step_momentum = config.av_step_std, config.av_step_momentum

    # Randomly select half of sequences (by index) to have zero angular velocity during some random middle-ish period
    middle_av_zero_indices = torch.randperm(batch_size)[:batch_size//2]
    start_av_zero = torch.randint(low=n_timesteps//4, high=n_timesteps//3, size=(batch_size,))
    end_av_zero = torch.randint(low=2*(n_timesteps//3), high=3*(n_timesteps//4), size=(batch_size,))
    
    # Randomly select starting angle for each sequence
    angle_0 = (torch.rand(batch_size)) * 2 * np.pi

    # Initialise tensors to store the target angle and input angular velocity for each sequence
    angle, angular_velocity = torch.zeros((batch_size, n_timesteps)), torch.zeros((batch_size, n_timesteps))

    for trial in range(batch_size):

        for t in range(angle_0_duration, n_timesteps):
            if trial in middle_av_zero_indices:
                if t>=start_av_zero[trial] and t<=end_av_zero[trial]:
                    angular_velocity[trial, t] = 0
                    continue

            angular_velocity[trial, t] = av_step_std * np.random.randn() + av_step_momentum * angular_velocity[trial, t-1]
        
        # Compute sequence's target angle as its initial angle + integral of angular velocity up to each timestep
        angle[trial] = angle_0[trial] + torch.cumsum(angular_velocity[trial], dim=0)
    # Take angle as mod 2pi
    angle = torch.remainder(angle, 2*np.pi)

        
    # Save input streams
    inputs[:,:,input_map['av']] = angular_velocity
    inputs[:,:angle_0_duration,input_map['sin_hd_0']] = torch.sin(angle_0).repeat((angle_0_duration,1)).T
    inputs[:,:angle_0_duration,input_map['cos_hd_0']] = torch.cos(angle_0).repeat((angle_0_duration,1)).T

    # Save output streams
    targets[:,:,target_map['sin_hd']] = torch.sin(angle)
    targets[:,:,target_map['cos_hd']] = torch.cos(angle)

    mask[:,:angle_0_duration,:] = 0

    return inputs, targets, mask



ALLO0D_TASK = Task('allo0D', 
                    n_inputs=3, n_outputs=2, 
                    task_specific_params=default_params, 
                    create_data_func=create_data,
                    test_func=test_allo,
                    input_map=input_map,
                    target_map=target_map)




