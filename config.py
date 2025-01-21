import torch

import random
import time

from typing import Set


############################################################################################################################################
######################################################## CONFIGURATION #####################################################################
############################################################################################################################################

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# Object storing various parameters of a given model build  (essentially a convenience wrapper for a dict)                                 #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


class Config:
    '''
    __init__
    ---------------------------------------------------------------------------------------------
    Receives
        kwargs : 
            keyword-argument pairs which capture parameters of the build
            (typical call signature will unwrap either default_params or existing saved config state)
    
    '''
    def __init__(self, **kwargs):
        # Initialise with default parameters (to ensure backward compatability)
        self.__dict__.update(**default_params)
        if len(kwargs) > 0:
                # Store given parameters
                self.update(**kwargs)
        # Create space for name of build
        self.name = None
        # Store time build was made
        if 'time' in self.__dict__:
            self.__dict__['time'] = time.time()



    '''
    get_name
    Gives the name of the configuration, or creates a default one if it does not exist
    ---------------------------------------------------------------------------------------------
    Returns
        str
            Name of configuration
    '''
    def get_name(self) -> str:
        if self.name is None:
            self.make_name(include=[])

        return self.name
    



    '''
    make_name
    Creates a name for the configuration based on important parameter setting
    ---------------------------------------------------------------------------------------------
    Receives
        include (optional) :
            list of parameter keys to include in name
            if not supplied, only task is used
        exclude (optional) :
            list of parameter keys to exclude from name
            if not supplied, none are excluded
    
    Returns
        str
            name created for configuration (also saved to self.name)
            format is 'task:<task>-<key>:<value>-....

    '''
    def make_name(self, include: Set[str] = None, exclude: Set[str] =None) -> str:

        if include is None:
            include = []
        if exclude is None:
            exclude = []
        
        parts = [f'task:{self.task}']
        for key in include:
            if key in self.__dict__ and key not in exclude and key != 'task':
                parts.append(f'{key}:{self.__dict__[key]}')
        
        self.name = '-'.join(parts)
        return self.name
    



    # Allows parameter access directly by key (i.e. config_obj[key])
    def __getitem__(self, key):
        return self.__dict__[key]

    # Allows update of parameter dictionary
    def update(self, **kwargs):
        self.__dict__.update(kwargs)
        
    # Prints all parameters
    def __str__(self):

        res = f'\n{"-"*25} CONFIG {"-"*25}\n'
        for key, val in self.__dict__.items():
            flag = '\t'
            res += (f'{flag}{key}: {val}\n')
        res += ('\n' + '-'*58)

        return res

    # Creates seed if none supplied
    def seed(self):
        if self.build_seed == -1:
            self.build_seed = random.randint(0, 2**32 - 1)













############################################################################################################################################
######################################################## DEFAULT CONFIGURATION #############################################################
############################################################################################################################################

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# Dictionary storing default parameter values of model build                                                                               #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


default_params = {

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Network Structural Parameters~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
    
    # No. neurons in hidden layer
    'n_neurons': 100,  
    # No. neurons in input layer                                                                 
    'n_inputs': 3,   
    # No. neurons in output layer                                                               
    'n_outputs': 2,
    # Name of activaion function (ReTanh, Tanh, ReLU)
    'activation_func_name': 'ReTanh',
    # Euler step size
    'dt': 50.0,
    # Neuron time constant
    'tau': 500.0,
    # Initialisation method for input and recurrent weights (normal, orthogonal)
    'W_in_init': 'normal',
    'W_rec_init': 'normal',
    # Constant controlling standard deviation of recurrent weight matrix (std = hidden_g / n_neurons^2)
    'hidden_g': 1.1, 
    # Flags detemining which parameter sets to learn
    'learn_x_0': True,
    'learn_W_in': True,
    'learn_W_in_bias': True,
    'learn_W_rec': True,
    'learn_W_out': True,
    'learn_W_out_bias': True,

    # Standard deviation of noise terms (zero-mean normallly distributed) in continuous time equation
    'state_noise_std': 0.1,
    'rate_noise_std': 0.0,
    'output_noise_std': 0.0,

    # Coeefficient of L2 regularisation term in loss function on network weights and activity
    'weight_lambda': 0.1,
    'rate_lambda': 0.1,  

    # Rank of recurrent matrix (for use with low-rank RNNs)
    'rank': None,


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Build Parameters~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

    # Time of creation (overridden at build time)
    'time':  -1,
    # Seed for network generation (-1 will be set a build time)
    'build_seed': -1, 
    # Maximum duration of build training
    'max_hours': 48,
    # Device trained on
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    # Name of training algorithm to use (Adam, AdamW or HF)
    'optimiser_name': 'Adam',  
    # Number of processes to use concurrently for data generation
    'num_loader_workers': 4,
    # Parameters for assessing training convergence
    'training_threshold': 0.0,
    'training_convergence_std_threshold': 10e-3,
    'training_convergence_std_threshold_window': 5000,
    # Save directory for model checkpoints
    'savedir': 'trained-models',
    # Flag indicating whether untrained model should be saved
    'save_initial_net': False,
    # Loss thresholds at which to save model
    'save_losses': [0.2, 0.1, 0.05, 0.025, 0.01, 0.005, 0.001, 0.0008, 0.0006, 0.0004, 0.0002, 0.00001],
    # Epoch multiples at which to save model
    'save_epochs': 1000,
    # Epoch multiples at which to test model
    'test_epochs': 100,
    # HF parameters
    'HF_damping': 0.5, 
    'HF_delta_decay': 0.95, 
    'HF_cg_max_iter': 100, 


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Training Parameters~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

    # Length of training sequences
    'n_timesteps': 500,
    # Number of epochs to train on (if -1, will train until threshold loss/convergence)
    'n_epochs': -1,
    # Maximum number of epochs to train for (when n_epochs = -1)
    'max_epochs': 500000,
    # Number of times to repeat a given batch
    'num_batch_repeats': 1,
    # Number of examples in one batch
    'batch_size': 500,
    # Size of minibatches to train on
    'minibatch_size': 500,
    # Maximum learning rate
    'max_lr': 1.0,
    # Mimumum learning rate
    'min_lr': 0.000001,
    # Initial learning rate schedule: higher values means increases to max quicker
    'lr_initial_schedule': 1e-9,
    # Anneal learning rate schedule: higher values means decreases to min quicker
    'lr_anneal_schedule': 1e-9,
    


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Testing Parameters~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
    
    # Length of sequences to store in testing dataset
    'test_n_timesteps': 500,
    # Number of sequences to store in testing dataset
    'test_batch_size': 500,

    # (Matplotlib) standard dimensions of figures created during testing
    'test_fig_width': 20,
    'test_fig_height': 10,
    'test_fig_margin': 0.06,
}