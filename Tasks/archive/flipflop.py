import torch

import sys

from task import *
from build import *
from test_funcs import *


default_params = {

    'transient_duration': 10,
    'min_pulses': 10,
    'max_pulses': 30,
    'pulse_width': 6,


    'n_fit_examples': 5
}


def create_data(config, inputs, targets, mask):
    batch_size, n_timesteps = inputs.shape[0], inputs.shape[1]

    min_pulses = config.min_pulses
    max_pulses = config.max_pulses
    pulse_width = config.pulse_width
    
    n_pulses = torch.randint(min_pulses, max_pulses, (batch_size,))
    for i in range(3):
        for s in range(config.batch_size):
            T = n_timesteps - config.transient_duration

            # pick random number of indices at which to have a pulse, for the current trial
            pulse_indices = torch.randperm(T-(2*pulse_width))[:n_pulses[s]]
            pulse_indices = pulse_indices[torch.sort(pulse_indices).indices]

            # pick either 1 or -1 to correspond to those pulses
            pulse_heights = 2*torch.randint(low=0, high=2, size=(n_pulses[s],)) - 1

            in_pulses = torch.zeros((T,))
            used_pulses, used_pulse_heights = [], []

            # for each pulse index...
            for j in range(n_pulses[s]):
                p_i = pulse_indices[j]

                # if this and the subsequent pulse do not overlap, keep it
                if j > 0 and p_i - pulse_indices[j-1] <= pulse_width:
                    continue

                # create the pulse shape
                in_pulses[p_i:p_i+pulse_width] = pulse_heights[j]

                used_pulses.append(p_i)
                used_pulse_heights.append(pulse_heights[j])

            # reset trial data based on which pulses were used (i.e. not overlapping)
            pulse_indices = torch.tensor(used_pulses)
            pulse_heights = torch.tensor(used_pulse_heights)
            n_pulses[s] = len(pulse_indices)

            out_pulses = torch.zeros((T+1,))
            if n_pulses[s] < 2:
                out_pulses[pulse_indices[0]:] = pulse_heights[0]
            else:
                out_pulses[pulse_indices[0]:pulse_indices[1]] = pulse_heights[0]
                for j in range(n_pulses[s]):
                    p_i = pulse_indices[j]
                    next_p_i = -1 if j==n_pulses[s]-1 else pulse_indices[j+1]
                    out_pulses[p_i:next_p_i] = pulse_heights[j]

            inputs[s,config.transient_duration:,i] = in_pulses
            targets[s,config.transient_duration:,i] = out_pulses[:-1]

    return inputs, targets, mask




FLIPFLOP_TASK = Task('flipflop', 
                    n_inputs=3, n_outputs=3, 
                    task_specific_params=default_params, 
                    create_data_func=create_data)





