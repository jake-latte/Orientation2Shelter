import torch
import torch.nn as nn

from config import Config

from typing import Callable, Tuple

import numpy as np


def get_activation_func_from_name(name):
    name = name.lower()
    if name == 'retanh':
        return lambda x: torch.maximum(torch.zeros_like(x), torch.tanh(x))
    elif name == 'tanh':
        return lambda x: torch.tanh(x)
    elif name == 'relu':
        return lambda x: torch.maximum(torch.zeros_like(x), x)
    else:
        raise ValueError('Invalid activation function name: use ReTanh, Tanh, or ReLU')


############################################################################################################################################
################################################### CONTINUOUS TIME RNN ####################################################################
############################################################################################################################################

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# PyTorch RNN Module                                                                                                                       #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #



class RNN(nn.Module):
    '''
    __init__
    Create RNN per specified structure and initialise weights
    ---------------------------------------------------------------------------------------------
    Receives
        config : 
            a config object which determines structure and other properties of the net (usual call signature will use task.config)
        activation_func (optional):
            custom activation function; if not supplied, activation_func_name in config is used with get_activation_func_from_name above
        kwargs : 
            keyword arguments to override any properties in config
    
    Returns
        None
    '''
    def __init__(self, config: Config, activation_func: Callable[[torch.Tensor], torch.Tensor] = None, **kwargs):

        # Initialise properties from config/kwargs (see config.py)
        self.n_neurons = kwargs.get('n_neurons', config.n_neurons)
        self.n_inputs = kwargs.get('n_inputs', config.n_inputs)
        self.n_outputs = kwargs.get('n_outputs', config.n_outputs)
        self.activation_func_name = kwargs.get('activation_func_name', config.activation_func_name)
        self.dt = kwargs.get('dt', config.dt)
        self.tau = kwargs.get('tau', config.tau)

        self.W_in_init = kwargs.get('W_in_init', config.W_in_init)
        self.W_rec_init = kwargs.get('W_rec_init', config.W_rec_init)
        self.hidden_g = kwargs.get('hidden_g', config.hidden_g)

        self.learn_x_0 = kwargs.get('learn_x_0', config.learn_x_0)
        self.learn_W_in = kwargs.get('learn_W_in', config.learn_W_in)
        self.learn_W_in_bias = kwargs.get('learn_W_in_bias', config.learn_W_in_bias)
        self.learn_W_rec = kwargs.get('learn_W_rec', config.learn_W_rec)
        self.learn_W_out = kwargs.get('learn_W_out', config.learn_W_out)
        self.learn_W_out_bias = kwargs.get('learn_W_out_bias', config.learn_W_out_bias)

        self.state_noise_std = kwargs.get('state_noise_std', config.state_noise_std)
        self.rate_noise_std = kwargs.get('rate_noise_std', config.rate_noise_std)
        self.output_noise_std = kwargs.get('output_noise_std', config.output_noise_std)

        self.device = kwargs.get('device', config.device)

        super(RNN, self).__init__()

        self.activation_func = get_activation_func_from_name(self.activation_func_name) if activation_func is None else activation_func

        # W_in (Feed-forward input weights)
        self.W_in = nn.Linear(self.n_inputs, self.n_neurons, bias=True)
        # W_in weights
        if self.W_in_init == 'normal':
            nn.init.normal_(self.W_in.weight, mean=0, std=1 / np.sqrt(self.n_inputs))
        elif self.W_in_init == 'orthogonal':
            nn.init.orthogonal_(self.W_in.weight, gain=torch.nn.init.calculate_gain('relu') * self.hidden_g)
        else:
            raise(f'Unsupported input weight initialisation. Use "normal". You used: {self.W_in_init}' )
        self.W_in.weight.requires_grad = self.learn_W_in
        # W_in bias
        input_bias = 0.1 + 0.01*torch.randn(self.n_neurons)
        self.W_in.bias = torch.nn.Parameter(torch.squeeze(input_bias))
        self.W_in.bias.requires_grad = self.learn_W_in_bias



        # W_rec (recurrent weights)
        self.W_rec = nn.Linear(self.n_neurons, self.n_neurons, bias=False, device=self.device)
        # W_rec weights (no bias)
        if self.W_rec_init == 'normal':
            nn.init.normal_(self.W_rec.weight, mean=0, std=self.hidden_g / np.sqrt(self.n_neurons)) 
        elif self.W_rec_init == 'orthogonal':
            nn.init.orthogonal_(self.W_rec.weight, gain=torch.nn.init.calculate_gain('relu') * self.hidden_g)
        else:
            raise(f'Unsupported input weight initialisation. Use "normal" or "orthogonal". You used: {self.W_rec_init}' )
        # Prohibit self-connections
        self.W_rec.weight = nn.Parameter(
            (1 - torch.eye(self.n_neurons, device=self.device)) * self.W_rec.weight.data, requires_grad=self.learn_W_rec
        )

        # W_out (feed-forward output weights)
        self.W_out = nn.Linear(self.n_neurons, self.n_outputs, bias=True)
        self.W_out.weight.requires_grad = self.learn_W_out
        self.W_out.bias.requires_grad = self.learn_W_out_bias


        # Initialise initial state
        self.x_0 = torch.nn.Parameter(torch.zeros(self.n_neurons), requires_grad=self.learn_x_0)


        self.to(self.device)
        self.train(False)




# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

    '''
    forward
    nn.Module.forward implementation for Continuous-Time RNN
    ---------------------------------------------------------------------------------------------
    Receives
        u (torch.Tensor): 
            input to the rnn, of shape [batch_size, n_timesteps, n_inputs]
        x_0 (optional) (torch.Tensor) :
            initial state of the rnn to use (of shape [n_neurons] or [batch_size, n_neurons]); if not supplied, self.x_0 is used
        noise (optional) (Tuple(torch.Tensor, torch.Tensor, torch.Tensor)) : 
            tuple of noise tensors (state noise, rate noise, output noise) matched to the input (generally from same batch)
        batch_first (default=True) :
            flag indicating whether batch index is first (as above) or second
    
    Returns
        Tuple(torch.Tensor, torch.Tensor, torch.Tensor) :
            states :
                tensor of shape [batch_size, n_timesteps, n_neurons] representing non-activated states of hidden units during batch
            activity :
                hidden unit activities, corresponding to states (i.e., activation_func(states))
            outputs :
                tensor of shape [batch_size, n_timesteps, n_outputs] corresponding to network output during batch
    '''
    def forward(self, u: torch.Tensor, x_0: torch.Tensor = None, noise: Tuple[torch.Tensor, torch.Tensor, torch.Tensor] = None, batch_first: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        # If dimension of input is 2, assume is one trial (batch_size=1) of [n_timesteps, n_inputs]
        if len(u.shape)==2:
            n_trials, n_timesteps = 1, u.shape[0]
            u = torch.unsqueeze(u, 0)
        elif batch_first:
            # Transpose shape of input tensor (to make iteration cleaner)
            n_trials, n_timesteps = u.shape[0], u.shape[1]
            u = u.transpose(0, 1)
        else:
            n_timesteps, n_trials = u.shape[0], u.shape[1]
        
        # Initialise network state to chosen starting point
        x_0 = self.x_0 if x_0 is None else x_0
        if len(x_0.shape)==1:
            x_0 = x_0.repeat((n_trials,1))
        else:
            assert x_0.shape[0]==n_trials

        # Generate noise if not supplied
        if noise is None:
            state_noise = torch.normal(mean=0, std=self.state_noise_std, size=(n_timesteps, n_trials, self.n_neurons), dtype=self.x_0.dtype).to(self.device)
            rate_noise = torch.normal(mean=0, std=self.rate_noise_std, size=(n_timesteps, n_trials, self.n_neurons), dtype=self.x_0.dtype).to(self.device)
            output_noise = torch.normal(mean=0, std=self.output_noise_std, size=(n_timesteps, n_trials, self.n_outputs), dtype=self.x_0.dtype).to(self.device)

        else:
            state_noise = noise[0].transpose(0, 1)
            rate_noise = noise[1].transpose(0, 1)
            output_noise = noise[2].transpose(0, 1)

        # Initialise lists to store tensors corresponding to state of net at each point in input sequence
        # X : net hidden unit states (where X[i] is net state after receiving ith step of input)
        # R : net hidden unit rates (where R[i] is net activity after receiving ith step of input)
        # Z : net output unit states (where Z[i] is net output after receiving ith step of input)
        X = []
        R = []
        Z = []

        for t in range(n_timesteps):
            if t==0:
                x_t, r_t = x_0, self.activation_func(x_0)
            else:
                x_t, r_t = X[-1], R[-1]

            # Continuous-Time RNN Update Funcion:
            x_next = x_t + (self.dt/self.tau) * (-x_t + self.W_rec(r_t) + self.W_in(u[t]) + state_noise[t])
            r_next = self.activation_func(x_next) + rate_noise[t]
            z_next = self.W_out(r_next) + output_noise[t]

            X.append(x_next)
            R.append(r_next)
            Z.append(z_next)

        # Convert X, R, Z to corrsponding output tensors
        states, activity, outputs = torch.stack(X, dim=1), torch.stack(R, dim=1), torch.stack(Z, dim=1)

        return states, activity, outputs
    

    # Helper function for iterating over all tensors requiring gradient
    def parameters(self):
        for params in super(RNN, self).parameters():
            if params.requires_grad:
                yield params
    



























############################################################################################################################################
########################################################### LOW-RANK RNN ###################################################################
############################################################################################################################################

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# Low-Rank version of above RNN                                                                                                            #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


# Module class for capturing low-rank recurrent weight matrix, storing parameters in 2R Nx1 vectors (whose sum of outer products gives weight matrix)
class LowRankRecurrentLayer(nn.Module):
    '''
    __init__
    Create low-rank recurrent weight matrix
    ---------------------------------------------------------------------------------------------
    Receives
        rank : 
            rank of recurrency
        n_neurons :
            number of neurons in recurrency
        requires_grad :
            should weights be tracked by autograd
        device :
            device to put matrix onto
    
    '''
    def __init__(self, rank: int, n_neurons: int, requires_grad: bool, device: torch.device):
        super().__init__()

        self.rank, self.requires_grad, self.device, self.n_neurons = rank, requires_grad, device, n_neurons

        # Initialise two matrix, M and N, each of row of which is a pair of vectors (of size n_neurons) whose outer products will be summed
        self.M = nn.Parameter(data=torch.normal(mean=0, std=1, size=(rank, n_neurons), device=device), requires_grad=requires_grad)
        self.N = nn.Parameter(data=torch.normal(mean=0, std=1, size=(rank, n_neurons), device=device), requires_grad=requires_grad)

        self.update_weight()

    # Function performing the sum-of-outer-products routine to create W_rec 
    # Must be called before any forward pass so that W_rec is updated after last optimiser step
    def update_weight(self):
        W_rec = torch.zeros((self.n_neurons, self.n_neurons), device=self.device)

        for r in range(self.rank):
            m = self.M[r].unsqueeze(1)
            n = self.N[r].unsqueeze(0)
            W_rec += m@n
        W_rec *= (1/self.n_neurons)

        self.weight = W_rec

    # Forward pass for performing W_rec @ x
    def forward(self, x):
        return x @ (self.weight).T
    



class LowRankRNN(RNN):
    '''
    __init__
    Create Low-Rank RNN per specified structure and initialise weights
    ---------------------------------------------------------------------------------------------
    Receives
        config : 
            a config object which determines structure and other properties of the net (usual call signature will use task.config)
            config must have 'rank' set
        activation_func (optional):
            custom activation function; if not supplied, activation_func_name in config is used with get_activation_func_from_name above
        kwargs : 
            keyword arguments to override any properties in config
    '''
    def __init__(self, config: Config, activation_func: Callable[[torch.Tensor], torch.Tensor] = None, **kwargs):
        self.rank = kwargs.get('rank', config.rank)
        assert self.rank is not None

        # Use RNN superclass to initialise (will create full-rank W_rec, which is then overridden below)
        super().__init__(config, activation_func=activation_func, **kwargs)

        # W_rec override (recurrent weights constrained to be low-rank)
        self.W_rec = LowRankRecurrentLayer(
            rank=self.rank, n_neurons=config.n_neurons, requires_grad=self.learn_W_rec, device=self.device
        )

        # W_in (initialise without bias and with normal feed-forward weights)
        nn.init.normal_(self.W_in.weight, mean=0, std=1)
        self.W_in.bias = None

    # Identical to above, but first updates W_rec (based on its defining vector products) before forward above
    def forward(self, u: torch.Tensor, x_0: torch.Tensor = None, noise: Tuple[torch.Tensor, torch.Tensor, torch.Tensor] = None, batch_first: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self.W_rec.update_weight()

        return super().forward(u, x_0=x_0, noise=noise, batch_first=batch_first)
    











    '''
    __init__
    Create Reinforcement Learning RNN per specified structure and initialise weights
    ---------------------------------------------------------------------------------------------
    Receives
        config : 
            a Config object which determines structure and other properties of the net
        kwargs : 
            keyword arguments to override any properties which may be contained in config
    
    Returns
        None
    '''
    def __init__(self, config, activation_func=None, **kwargs):

        # Initialise properties from config/kwargs (see Config.py)
        self.n_neurons = kwargs.get('n_neurons', config.n_neurons)
        self.n_inputs = kwargs.get('n_inputs', config.n_inputs)
        self.n_outputs = kwargs.get('n_outputs', config.n_outputs) # n actions
        self.activation_func_name = kwargs.get('activation_func_name', config.activation_func_name)
        self.dt = kwargs.get('dt', config.dt)
        self.tau = kwargs.get('tau', config.tau)

        self.W_in_init = kwargs.get('W_in_init', config.W_in_init)
        self.W_rec_init = kwargs.get('W_rec_init', config.W_rec_init)
        self.hidden_g = kwargs.get('hidden_g', config.hidden_g)

        self.learn_x_0 = kwargs.get('learn_x_0', config.learn_x_0)
        self.learn_W_in = kwargs.get('learn_W_in', config.learn_W_in)
        self.learn_W_in_bias = kwargs.get('learn_W_in_bias', config.learn_W_in_bias)
        self.learn_W_rec = kwargs.get('learn_W_rec', config.learn_W_rec)
        self.learn_W_out = kwargs.get('learn_W_out', config.learn_W_out)
        self.learn_W_out_bias = kwargs.get('learn_W_out_bias', config.learn_W_out_bias)

        self.state_noise_std = kwargs.get('state_noise_std', config.state_noise_std)
        self.rate_noise_std = kwargs.get('rate_noise_std', config.rate_noise_std)
        self.output_noise_std = kwargs.get('output_noise_std', config.output_noise_std)

        self.device = kwargs.get('device', config.device)

        super(RNN, self).__init__()


        self.activation_func = get_activation_func_from_name(self.activation_func_name) if activation_func is None else activation_func

        # Initialise input weights (W_in)
        self.W_in = nn.Linear(self.n_inputs, self.n_neurons, bias=True)

        if self.W_in_init == 'normal':
            nn.init.normal_(self.W_in.weight, mean=0, std=1 / np.sqrt(self.n_inputs))
        elif self.W_in_init == 'orthogonal':
            nn.init.orthogonal_(self.W_in.weight, gain=torch.nn.init.calculate_gain('relu') * self.hidden_g)
        else:
            raise(f'Unsupported input weight initialisation. Use "normal". You used: {self.W_in_init}' )

        self.W_in.weight.requires_grad = self.learn_W_in
        
        input_bias = 0.1 + 0.01*torch.randn(self.n_neurons)
        self.W_in.bias = torch.nn.Parameter(torch.squeeze(input_bias))
        self.W_in.bias.requires_grad = self.learn_W_in_bias



        # Initialise recurrent weights (W_rec)
        self.W_rec = nn.Linear(self.n_neurons, self.n_neurons, bias=False, device=self.device)
        
        if self.W_rec_init == 'normal':
            nn.init.normal_(self.W_rec.weight, mean=0, std=self.hidden_g / np.sqrt(self.n_neurons)) 
        elif self.W_rec_init == 'orthogonal':
            nn.init.orthogonal_(self.W_rec.weight, gain=torch.nn.init.calculate_gain('relu') * self.hidden_g)
        else:
            raise(f'Unsupported input weight initialisation. Use "normal" or "orthogonal". You used: {self.W_rec_init}' )
        
        self.W_rec.weight = nn.Parameter(
            (1 - torch.eye(self.n_neurons, device=self.device)) * self.W_rec.weight.data, requires_grad=self.learn_W_rec
        )

        # Initialise output weights (W_out)
        self.policy_linear = nn.Linear(self.n_neurons, self.n_outputs, bias=True)
        self.policy_linear.weight.requires_grad = self.learn_W_out
        self.policy_linear.bias.requires_grad = self.learn_W_out_bias 

        self.value_linear = nn.Linear(self.n_neurons, 1, bias=True)
        self.value_linear.weight.requires_grad = self.learn_W_out
        self.value_linear.bias.requires_grad = self.learn_W_out_bias


        # Initialise initial state
        self.x_0 = torch.nn.Parameter(torch.zeros(self.n_neurons), requires_grad=self.learn_x_0)


        self.to(self.device)
        # self.train(False)




# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

    '''
    forward
    nn.Module.forward implementation for Continuous-Time RNN
    ---------------------------------------------------------------------------------------------
    Receives
        input : 
            a batched input tensor of shape [num sequences, num timesteps, num inputs]
        noise (optional) : 
            tuple of noise tensors (state noise, rate noise, output noise) matched to the input, each of shape
            [num sequences, num timesteps, num units (hidden units or output units)]
            if not supplied, are generated from a normal distribution ~ N(0, config.---_noise_std^2)
    
    Returns
        tuple[ torch.tensor ] (states, activations, outputs) :
            states :
                tensor of shape [num sequences, num timesteps, num hidden units] represnting non-activated states of
                hidden units during trial
            activations :
                hidden units activities, corresponding to states (i.e., activation_func(states))
            outputs :
                tensor of shape [num sequences, num timesteps, num output units] corresponding to network output over
                input sequence
    '''
    def forward(self, state, x_0=None, noise=None):

        assert len(state.shape) <= 2

        if len(state.shape)==0:
            u = torch.unsqueeze(u, 0)
        
        # Initialise network state to chosen starting point
        x_0 = self.x_0 if x_0 is None else x_0

        # Generate noise if not supplied, or transpose that given
        if noise is None:
            state_noise = torch.normal(mean=0, std=self.state_noise_std, size=(self.n_neurons,), dtype=self.x_0.dtype).to(self.device)
            rate_noise = torch.normal(mean=0, std=self.rate_noise_std, size=(self.n_neurons,), dtype=self.x_0.dtype).to(self.device)
            policy_noise = torch.normal(mean=0, std=self.output_noise_std, size=(self.n_outputs,), dtype=self.x_0.dtype).to(self.device)
            value_noise = torch.normal(mean=0, std=self.output_noise_std, size=(1,), dtype=self.x_0.dtype).to(self.device)

        x_t, r_t = x_0, self.activation_func(x_0)

        # Continuous-Time RNN Update Funcion:
        x_next = x_t + (self.dt/self.tau) * (-x_t + self.W_rec(r_t) + self.W_in(u) + state_noise)
        r_next = self.activation_func(x_next) + rate_noise
        p_next = self.policy_linear(r_next) + policy_noise
        v_next = self.value_linear(r_next) + value_noise

        states, activity, actions, values = x_next, r_next, p_next, v_next

        return states, activity, actions, values
    
    def get_action(self, state):
        _,_,actions,_ = self.forward(state)
        return actions
    


    def parameters(self):
        for params in super(RNN, self).parameters():
            if params.requires_grad:
                yield params