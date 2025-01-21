import torch
from torch.utils.data import  Dataset

import numpy as np

from PIL import Image, ImageDraw

from typing import Tuple, Dict, Any

from task import Task




############################################################################################################################################
##################################################### TASK DATASET SUPERCLASS ##############################################################
############################################################################################################################################

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# Superclass for capturing the generation and presentation of data to RNNs                                                                 #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


class TaskDataset(Dataset):
    '''
    __init__
    Store the config parameters and functions which will allow data generation for a given task
    ---------------------------------------------------------------------------------------------
    Receives
        task :
            Task to generate data for
        for_training (optional) :
            flag indicating whether training or testing parameters should be used
        kwargs : 
            additional keyword-argument pairs which can override those in task.config
    '''

    def __init__(self, task: Task, for_training: bool = True, **kwargs):

        self.config = task.config

        # Initialise properties from task.config/kwargs (see config.py)
        self.n_inputs = kwargs.get('n_inputs', task.config.n_inputs)
        self.n_neurons = kwargs.get('n_neurons', task.config.n_neurons)
        self.n_outputs = kwargs.get('n_outputs', task.config.n_outputs)

        self.device = kwargs.get('device', task.config.device)
        self.dt = kwargs.get('dt', task.config.dt)

        self.state_noise_std = kwargs.get('state_noise_std', task.config.state_noise_std)
        self.rate_noise_std = kwargs.get('rate_noise_std', task.config.rate_noise_std)
        self.output_noise_std = kwargs.get('output_noise_std', task.config.output_noise_std)

        # Save relevant batch shape parameters based on training or testing
        if for_training:
            self.n_epochs = kwargs.get('n_epochs', task.config.n_epochs)
            # If no epoch number is set, training should be able to continue until the maximum allowed number of epochs
            if self.n_epochs <= 0:
                self.n_epochs = task.config.max_epochs
            self.batch_size = kwargs.get('batch_size', task.config.batch_size)
            self.n_timesteps = kwargs.get('n_timesteps', task.config.n_timesteps)
        else:
            self.n_epochs = 1
            self.batch_size = kwargs.get('test_batch_size', task.config.test_batch_size)
            self.n_timesteps = kwargs.get('test_n_timesteps', task.config.test_n_timesteps)

        # Store task-specific data creation routine
        self.create_data_worker = task.create_data_func

        super(Dataset, self).__init__()


    '''
    create_data
    Create task-dependent input and output sequences (here a dummy function to be overriden by subclasses)
    ---------------------------------------------------------------------------------------------
    
    Returns 
        input :
            tensor of shape [num sequences, num timesteps, num inputs] capturing dataset of task input sequnces (on cpu)
        targets :
            tensor of shape [num sequences, num timesteps, num outputs] capturing dataset of task target outputs (on cpu)
        vars :
            dictionary of task variables
        mask :
            tensor of same shape as target defining which timesteps should contribute to loss calculation
        state_, rate_, output_noise :
            noise values to add to each moment in the trials; corresponds to whether noise should be added to hidden states or rates,
            or the output units
        Note: all tensors will be on cpu
    '''
    def create_data(self) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
        # Initialise input and target tensors
        inputs = torch.zeros((self.batch_size, self.n_timesteps, self.n_inputs))
        targets = torch.zeros((self.batch_size, self.n_timesteps, self.n_outputs))
        mask = torch.ones_like(targets, dtype=torch.bool)

        inputs, targets, vars, mask = self.create_data_worker(self.config, inputs, targets, mask)
        
        # Generate noise corresponding to dataset
        state_noise = torch.normal(mean=0, std=self.state_noise_std, size=(self.batch_size, self.n_timesteps, self.n_neurons))
        rate_noise = torch.normal(mean=0, std=self.rate_noise_std, size=(self.batch_size, self.n_timesteps, self.n_neurons))
        output_noise = torch.normal(mean=0, std=self.output_noise_std, size=(self.batch_size, self.n_timesteps, self.n_outputs))

        return (inputs, targets, vars, mask, state_noise, rate_noise, output_noise)
        
    '''
    get_batch
    Wraps create data outputs into a single batch dictioanry
    ---------------------------------------------------------------------------------------------
    
    Returns
        dict w/ keys :
            inputs, targets, vars, mask : corresponding to outputs of create_data
            noise : tuple of (state_noise, rate_noise, output_noise)
        Note: all tensors will be on training device
    '''
    def get_batch(self) -> Dict['str', Any]:
        inputs, targets, vars, mask, state_noise, rate_noise, output_noise = self.create_data()
        return {
            'inputs': inputs.to(self.device),
            'targets': targets.to(self.device),
            'vars': {key: var.to(self.device) for key, var in vars.items()},
            'mask': mask.to(self.device),
            'noise': (
                state_noise.to(self.device),
                rate_noise.to(self.device),
                output_noise.to(self.device)
            )
        }

    # Internal functions for compatibility with Dataset superclass and DataLoader Wrapper
    def __len__(self) -> int:
        return self.n_epochs * self.batch_size

    def __getitem__(self, index: int) -> list:
        return []
    
    def __getitems__(self, indices: int) -> list:
        return []
    
    def collate_fn(self, data: Any) -> Dict[str, Any]:
        inputs, targets, vars, mask, state_noise, rate_noise, output_noise = self.create_data()

        return {
            'inputs': inputs,
            'targets': targets,
            'vars': vars,
            'mask': mask,
            'noise': (
                state_noise,
                rate_noise,
                output_noise
            )
        }
        