import torch
from torch.utils.data import  Dataset

import numpy as np


from PIL import Image, ImageDraw




############################################################################################################################################
##################################################### TASK DATASET SUPERCLASS ##############################################################
############################################################################################################################################

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# Superclass for capturing the data similarities between variations of the ego and allocentric tasks                                      #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


class TaskDataset(Dataset):
    '''
    __init__
    Saves all the configuration parameters which will be relevant
    ---------------------------------------------------------------------------------------------
    Receives
        config :
            a Config object which determines properties of the data
        for_training (optional) :
            flag indicating whether training or testing parameters should be used
        kwargs : 
            additional keyword-argument pairs which can override those in config
    
    Result (by modification of self - see TaskDataset.create_data)
        input (torch.tensor) :
            tensor of shape [num sequences, num timesteps, num inputs] capturing dataset of task input sequnces (on cpu)
        targets (torch.tensor) :
            tensor of shape [num sequences, num timesteps, num outputs] capturing dataset of task target outputs (on cpu)
        state_, rate_, output_noise (torch.tensor) :
            tensors which represent the noise to be added to the states, rates, and outputs of the network
            for each trial in the dataset (on cpu)
    '''

    def __init__(self, task, for_training=True, **kwargs):

        self.config = task.config

        # Initialise properties from config/kwargs (see Config.py)
        self.n_inputs = kwargs.get('n_inputs', task.config.n_inputs)
        self.n_neurons = kwargs.get('n_neurons', task.config.n_neurons)
        self.n_outputs = kwargs.get('n_outputs', task.config.n_outputs)

        self.device = kwargs.get('device', task.config.device)
        self.dt = kwargs.get('dt', task.config.dt)

        self.test_batch_size = kwargs.get('test_batch_size', task.config.test_batch_size)
        self.test_n_timesteps = kwargs.get('test_n_timesteps', task.config.test_n_timesteps)

        self.state_noise_std = kwargs.get('state_noise_std', task.config.state_noise_std)
        self.rate_noise_std = kwargs.get('rate_noise_std', task.config.rate_noise_std)
        self.output_noise_std = kwargs.get('output_noise_std', task.config.output_noise_std)

        if for_training:
            self.n_epochs = kwargs.get('n_epochs', task.config.n_epochs)
            if self.n_epochs <= 0:
                self.n_epochs = task.config.max_epochs
            self.batch_size = kwargs.get('batch_size', task.config.batch_size)
            self.n_timesteps = kwargs.get('n_timesteps', task.config.n_timesteps)
        else:
            self.n_epochs = 1
            self.batch_size = kwargs.get('test_batch_size', task.config.test_batch_size)
            self.n_timesteps = kwargs.get('test_n_timesteps', task.config.test_n_timesteps)

        self.create_data_worker = task.create_data_func

        super(Dataset, self).__init__()


    '''
    create_data
    Create task-dependent input and output sequences (here a dummy function to be overriden by subclasses)
    ---------------------------------------------------------------------------------------------
    Receives
        config :
            a Config object which determines properties of the data
    
    Result (by modification of self)
        input (torch.tensor) :
            tensor of shape [num sequences, num timesteps, num inputs] capturing dataset of task input sequnces (on cpu)
        targets (torch.tensor) :
            tensor of shape [num sequences, num timesteps, num outputs] capturing dataset of task target outputs (on cpu)
    '''
    def create_data(self):
        # Initialise input and target tensors
        inputs = torch.zeros((self.batch_size, self.n_timesteps, self.n_inputs))
        targets = torch.zeros((self.batch_size, self.n_timesteps, self.n_outputs))
        mask = torch.ones_like(targets, dtype=torch.bool)
        
        # Generate and save noise corresponding to dataset
        state_noise = torch.normal(mean=0, std=self.state_noise_std, size=(self.batch_size, self.n_timesteps, self.n_neurons))
        rate_noise = torch.normal(mean=0, std=self.rate_noise_std, size=(self.batch_size, self.n_timesteps, self.n_neurons))
        output_noise = torch.normal(mean=0, std=self.output_noise_std, size=(self.batch_size, self.n_timesteps, self.n_outputs))

        inputs, targets, vars, mask = self.create_data_worker(self.config, inputs, targets, mask)

        return (inputs, targets, vars, mask, state_noise, rate_noise, output_noise)
        
    
    def get_batch(self):
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


    def __len__(self):
        return self.n_epochs * self.batch_size

    def __getitem__(self, index):
        return []
    
    def __getitems__(self, indices):
        return []
    
    def collate_fn(self, data):
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
        