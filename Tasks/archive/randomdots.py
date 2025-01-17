import torch

import sys

from task import *
from build import *
from test_funcs import *


default_params = {

    'min_coherence': -0.1875,
    'max_coherence': 0.1875,
    'min_pulses': 10,
    'max_pulses': 30,
    'pulse_width': 6,
    'transient_duration': 10,


    'n_fit_examples': 5
}

def create_data(config, inputs, targets, mask):

    batch_size, n_timesteps = inputs.shape[0], inputs.shape[1]
    
    d_m = torch.FloatTensor(batch_size).uniform_(config.min_coherence, config.max_coherence)
    d_c = torch.FloatTensor(batch_size).uniform_(config.min_coherence, config.max_coherence)

    t0 = config.transient_duration
    T = n_timesteps - t0
    ctx = torch.round(torch.rand(config.batch_size))
    for i in range(config.batch_size):
        inputs[i,t0:,0] = d_m[i] + torch.normal(mean=0, std=1, size=(T,))
        inputs[i,t0:,1] = d_c[i] + torch.normal(mean=0, std=1, size=(T,))
        
        inputs[i,t0:,2] = torch.ones(T) * ctx[i]
        inputs[i,t0:,3] = torch.ones(T) * (1 - ctx[i])

        targets[i,-1,0] = torch.sign(d_m[i]) if ctx[i] else torch.sign(d_c[i])

    mask[:,:t0,:] = 0
    mask[:,t0+1:-1,:] = 0

    return inputs, targets, mask



RANDOMDOTS_TASK = Task('randomdots', 
                    n_inputs=4, n_outputs=1, 
                    task_specific_params=default_params, 
                    create_data_func=create_data)






