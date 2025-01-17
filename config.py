import torch

import random
import time


############################################################################################################################################
######################################################## CONFIGURATION #####################################################################
############################################################################################################################################

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# Object storing various (ideally all) parameters of a given model build  (essentially a convenience wrapper for a dict)                   #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


class Config:
    '''
    __init__
    ---------------------------------------------------------------------------------------------
    Receives
        kwargs : 
            dictionary of keyword-argument pairs which capture parameters of the build
            (typical call signature will unwrap either default_params or existing saved config state)
    
    Returns
        None
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
    Receives
        None

    Returns
        str
            Name of configuration
    '''
    def get_name(self):
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
            if not supplied, used all keys
        exclude (optional) :
            list of parameter keys to exclude from name
            if not supplied, none are excluded
    
    Returns
        str
            name created for configuration (also saved to object .name)
            format is 'task:<task>-<key>:<value>-....

    '''
    def make_name(self, include=None, exclude=None):

        if include is None:
            include = self.__dict__.keys()
        if exclude is None:
            exclude=[]
        
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
    def print(self):
        print('Instance Parameters')
        for key, value in self.__dict__.items():
            print(f'\t{key}: {value}')

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

    # Coeefficient of regularisation term in loss function on network activities
    'weight_lambda': 0.1,
    'rate_lambda': 0.1,  


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Build Parameters~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

    # Time of creation (this time is overriden at config init)
    'time':  -1,
    # (PyTorch) Seed for network generation
    'build_seed': -1, 
    'max_hours': 48,
    # Device to train on
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    # Name of training algorithm to use (Adam)
    'optimiser_name': 'Adam',  

    'num_loader_workers': 4,

    'training_threshold': 0.0,
    'training_convergence_std_threshold': 10e-3,
    'training_convergence_std_threshold_window': 5000,
    
    'savedir': 'trained-models',
    'save_initial_net': False,
    'save_losses': [0.2, 0.1, 0.05, 0.025, 0.01, 0.005, 0.001, 0.0008, 0.0006, 0.0004, 0.0002, 0.00001],
    'save_epochs': 1000,

    'HF_damping': 0.5, 
    'HF_delta_decay': 0.95, 
    'HF_cg_max_iter': 100, 


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Training Parameters~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

    'n_timesteps': 500,
    'n_epochs': -1,
    'max_epochs': 500000,
    'num_batch_repeats': 3,
    'batch_size': 500,
    'minibatch_size': 250,
    'max_lr': 1.0,
    'min_lr': 0.000001,
    'lr_initial_schedule': 1.0,
    'lr_anneal_schedule': 0.1,
    'test_epochs': 100,


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Testing Parameters~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
    
    # Length of sequences to store in testing dataset
    'test_n_timesteps': 500,
    # Number of sequences to store in testing dataset
    'test_batch_size': 500,

    # (Matplotlib) standard dimensions of figures created during testing
    'test_fig_width': 20,
    'test_fig_height': 10,
    'test_fig_margin': 0.06,



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Experimnetal~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

    'rank': None
}