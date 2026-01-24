import torch
import torch.nn as nn

from o2s.config import Config
from o2s.task import Task

from typing import Callable, Tuple, List

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


class ConstrainedFeedForwardLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, requires_grad: bool, device: torch.device):
        super().__init__()

        self.in_features, self.out_features, self.requires_grad, self.device = in_features, out_features, requires_grad, device

        self.register_buffer('vecs', torch.empty((out_features, in_features), device=device))
        nn.init.orthogonal_(self.vecs)

        self.norms = nn.Parameter(data=torch.normal(mean=1, std=0.01, size=(in_features,), device=device), requires_grad=requires_grad).to(device)

        self.bias = nn.Parameter(0.1 + 0.01*torch.randn(out_features), requires_grad=requires_grad).to(device)


    @property
    def weight(self):
        W = self.vecs
        W = W / torch.norm(W, dim=0, keepdim=True) * self.norms.reshape(1, -1).repeat(self.out_features, 1)
        return W

    
    def forward(self, x):
        return x @ self.weight.T + (self.bias if self.bias is not None else 0)
    
    def __repr__(self):
        return f'ConstrainedFeedForwardLayer(in_features={self.in_features}, out_features={self.out_features})'
    

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


    @property
    def weight(self):
        W_rec = torch.zeros((self.n_neurons, self.n_neurons), device=self.device, dtype=self.M.dtype)
        for r in range(self.rank):
            m = self.M[r].unsqueeze(1)
            n = self.N[r].unsqueeze(0)
            W_rec += m @ n
        W_rec *= (1 / self.n_neurons)
        return W_rec

    
    def forward(self, x):
        return x @ self.weight.T

    def __repr__(self):
        return f'LowRankRecurrentLayer(rank={self.rank})'



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
    def __init__(self, task: Task, activation_func: Callable[[torch.Tensor], torch.Tensor] = None, **kwargs):
        config = task.config

        # Initialise properties from config/kwargs (see config.py)
        self.n_neurons = kwargs.get('n_neurons', config.n_neurons)
        self.n_inputs = kwargs.get('n_inputs', config.n_inputs)
        self.n_outputs = kwargs.get('n_outputs', config.n_outputs)
        self.activation_func_name = kwargs.get('activation_func_name', config.activation_func_name)
        self.dt = kwargs.get('dt', config.dt)
        self.tau = kwargs.get('tau', config.tau)
        self.hidden_g = kwargs.get('hidden_g', config.hidden_g)

        self.learn_x_0 = kwargs.get('learn_x_0', config.learn_x_0)
        self.learn_W_in = kwargs.get('learn_W_in', config.learn_W_in)
        self.learn_W_in_norm = kwargs.get('learn_W_in_norm', config.learn_W_in_norm)
        self.learn_W_rec = kwargs.get('learn_W_rec', config.learn_W_rec)
        self.learn_W_out = kwargs.get('learn_W_out', config.learn_W_out)
        self.learn_W_out_norm = kwargs.get('learn_W_out_norm', config.learn_W_out_norm)

        if self.learn_W_in_norm:
            assert not self.learn_W_in
        if self.learn_W_out_norm:
            assert not self.learn_W_out

        self.state_noise_std = kwargs.get('state_noise_std', config.state_noise_std)
        self.rate_noise_std = kwargs.get('rate_noise_std', config.rate_noise_std)
        self.output_noise_std = kwargs.get('output_noise_std', config.output_noise_std)

        self.rank = kwargs.get('rank', config.rank)
        self.intermediate_output_dim = kwargs.get('intermediate_output_dim', config.intermediate_output_dim)
        self.linear = kwargs.get('linear', config.linear)

        self.solver = kwargs.get('solver', config.solver)

        self.device = kwargs.get('device', config.device)

        super(RNN, self).__init__()

        self.activation_func = get_activation_func_from_name(self.activation_func_name) if activation_func is None else activation_func

        # learn only input vector norms
        if self.learn_W_in_norm:
            self.W_in = ConstrainedFeedForwardLayer(self.n_inputs, self.n_neurons, self.learn_W_in_norm, self.device)
            nn.init.normal_(self.W_in.vecs, mean=0, std=self.hidden_g / np.sqrt(self.n_inputs))
        
        # learn full input vectors
        else:
            # W_in (Feed-forward input weights)
            self.W_in = nn.Linear(self.n_inputs, self.n_neurons, bias=True)
            nn.init.normal_(self.W_in.weight, mean=0, std=self.hidden_g / np.sqrt(self.n_inputs))
            self.W_in.weight.requires_grad = self.learn_W_in
        # W_in bias
        input_bias = 0.1 + 0.01*torch.randn(self.n_neurons)
        self.W_in.bias = torch.nn.Parameter(torch.squeeze(input_bias))
        self.W_in.bias.requires_grad = self.learn_W_in or self.learn_W_in_norm


        # Vanilla RNN
        if self.rank <= 0:

            # W_rec (recurrent weights)
            self.W_rec = nn.Linear(self.n_neurons, self.n_neurons, bias=False, device=self.device)
            nn.init.normal_(self.W_rec.weight, mean=0, std=self.hidden_g / np.sqrt(self.n_neurons)) 
            # Prohibit self-connections
            self.W_rec.weight = nn.Parameter(
                (1 - torch.eye(self.n_neurons, device=self.device)) * self.W_rec.weight.data, requires_grad=self.learn_W_rec
            )

        # Low-Rank RNN
        else: 

            self.W_rec = LowRankRecurrentLayer(
                rank=self.rank, n_neurons=config.n_neurons, requires_grad=self.learn_W_rec, device=self.device
            )

            self.W_in.bias = None

        if self.intermediate_output_dim > 0:

            self.W_out_inter = nn.Linear(self.n_neurons, self.intermediate_output_dim, bias=True, device=self.device)
            W_out_in_features = self.intermediate_output_dim

        else:

            self.W_out_inter = None
            W_out_in_features = self.n_neurons

        # learn only output vector norms
        if self.learn_W_out_norm:

            self.W_out = ConstrainedFeedForwardLayer(W_out_in_features, self.n_outputs, self.learn_W_out_norm, self.device)
        
        # learn full output vectors
        else:
            self.W_out = nn.Linear(W_out_in_features, self.n_outputs, bias=True)
            self.W_out.weight.requires_grad = self.learn_W_out
        output_bias = 0.1 + 0.01*torch.randn(self.n_outputs)
        self.W_out.bias = torch.nn.Parameter(torch.squeeze(output_bias))
        self.W_out.bias.requires_grad = self.learn_W_out or self.learn_W_out_norm

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
        select_t (default=None) :
            list of indices to select from the output tensors (i.e., only return outputs at these timesteps)
    
    Returns
        Tuple(torch.Tensor, torch.Tensor, torch.Tensor) :
            states :
                tensor of shape [batch_size, n_timesteps, n_neurons] representing non-activated states of hidden units during batch
            activity :
                hidden unit activities, corresponding to states (i.e., activation_func(states))
            outputs :
                tensor of shape [batch_size, n_timesteps, n_outputs] corresponding to network output during batch
    '''
    def forward(self, u: torch.Tensor, 
                x_0: torch.Tensor = None, 
                noise: Tuple[torch.Tensor, torch.Tensor, torch.Tensor] = None, 
                batch_first: bool = True, select_t: List[int] = None, 
                repeat_input: Tuple[int, int]=None,
                offload:bool=False, collapse:bool=False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        # If dimension of input is 2, assume is one trial (batch_size=1) of [n_timesteps, n_inputs]
        if repeat_input is not None:
            n_timesteps, init_duration = repeat_input
            assert len(u.shape) == 2
            assert u.shape[1] == self.n_inputs
            n_trials = u.shape[0]

        else:
            if len(u.shape)==2:
                n_trials, n_timesteps = 1, u.shape[0]
                u = torch.unsqueeze(u, 0)
            elif batch_first:
                # Transpose shape of input tensor (to make iteration cleaner)
                n_trials, n_timesteps = u.shape[0], u.shape[1]
                u = u.transpose(0, 1)
            else:
                n_timesteps, n_trials = u.shape[0], u.shape[1]

        on_device = self.device
        off_device = 'cpu' if offload else self.device
        
        if repeat_input is not None:
            u = u.to(on_device)
        else:
            u = u.to(off_device)
        
        # Initialise network state to chosen starting point
        x_0 = self.x_0 if x_0 is None else x_0
        if len(x_0.shape)==1:
            x_0 = x_0.repeat((n_trials,1))
        else:
            assert x_0.shape[0]==n_trials

        # Generate noise if not supplied
        state_noise, rate_noise, output_noise = None, None, None
        if noise is not None:
            if noise[0] is not None:
                state_noise = noise[0].transpose(0, 1).to(off_device)
            if noise[1] is not None:
                rate_noise = noise[1].transpose(0, 1).to(off_device)
            if noise[2] is not None:
                output_noise = noise[2].transpose(0, 1).to(off_device)

        assert collapse==False if select_t is not None else True
        if select_t is None and not collapse:
            select_t = np.arange(n_timesteps)

        # Initialise lists to store tensors corresponding to state of net at each point in input sequence
        # X : net hidden unit states (where X[i] is net state after receiving ith step of input)
        # R : net hidden unit rates (where R[i] is net activity after receiving ith step of input)
        # Z : net output unit states (where Z[i] is net output after receiving ith step of input)
        X = [x_0.to(off_device)]
        R = [self.activation_func(x_0).to(off_device)]
        Z = []

        def F(x, r, u, noise):
            if self.linear:
                x_step = -x + self.W_rec(x) + self.W_in(u)
            else:
                x_step = -x + self.W_rec(r) + self.W_in(u)
            if noise is not None:
                x_step += noise
            return (1/self.tau) * x_step

        for t in range(n_timesteps):
            x_t, r_t = X.pop(), R.pop()

            u_t = u[t] if repeat_input is None else (u if t < init_duration else torch.zeros_like(u))
            x_t, r_t, u_t = x_t.to(on_device), r_t.to(on_device), u_t.to(on_device)

            if state_noise is None:
                state_noise_t = None if self.state_noise_std==0 else torch.normal(mean=0, std=self.state_noise_std, size=(n_trials, self.n_neurons), device=on_device)
            else:
                state_noise_t = state_noise[t].to(on_device)
            if rate_noise is None:
                rate_noise_t = None if self.rate_noise_std==0 else torch.normal(mean=0, std=self.rate_noise_std, size=(n_trials, self.n_neurons), device=on_device)
            else:
                rate_noise_t = rate_noise[t].to(on_device)
            if output_noise is None:
                output_noise_t = None if self.output_noise_std==0 else torch.normal(mean=0, std=self.output_noise_std, size=(n_trials, self.n_outputs), device=on_device)
            else:
                output_noise_t = output_noise[t].to(on_device)

            # Continuous-Time RNN Update Funcion:
            if self.solver == 'euler':
                x_next = x_t + self.dt * F(x_t, r_t, u_t, state_noise_t)
                
            elif self.solver == 'rk4':
                x_next = x_t

                k1 = F(x_t, r_t, u_t, state_noise_t)
                x_next += (self.dt/6) * k1

                k2 = F(x_t + 0.5 * self.dt * k1, r_t, u_t, state_noise_t)
                x_next += (self.dt/3) * k2
                del k1

                k3 = F(x_t + 0.5 * self.dt * k2, r_t, u_t, state_noise_t)
                x_next += (self.dt/3) * k3
                del k2

                k4 = F(x_t + self.dt * k3, r_t, u_t, state_noise_t)
                x_next += (self.dt/6) * k4
                del k3, k4

            else:
                raise ValueError(f'Unsupported solver: {self.solver}')
            
            r_next = self.activation_func(x_next)
            if rate_noise_t is not None:
                r_next += rate_noise_t

            if self.W_out_inter is not None:
                z_next = self.W_out(self.activation_func(self.W_out_inter(x_next if self.linear else r_next)))
            else:
                z_next = self.W_out(x_next if self.linear else r_next)
            if output_noise_t is not None:
                z_next += output_noise_t


            if collapse:
                R.append(torch.mean(r_next.to(off_device)**2))
                Z.append(z_next.to(off_device))
            else:
                if t in select_t:
                    X.append(x_next.to(off_device))
                    R.append(r_next.to(off_device))
                    Z.append(z_next.to(off_device))
                
                if t >= max(select_t):
                    break

            X.append(x_next)
            R.append(r_next)

        if collapse:
            states, activity, outputs = None, torch.mean(torch.stack(R[1:-1], dim=0)), torch.stack(Z, dim=1)
        else:
            states, activity, outputs = torch.stack(X, dim=1), torch.stack(R, dim=1), torch.stack(Z, dim=1)

        return states, activity, outputs
    

    # Helper function for iterating over all tensors requiring gradient
    def parameters(self):
        for params in super(RNN, self).parameters():
            if params.requires_grad:
                yield params

        