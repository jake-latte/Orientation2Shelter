from config import Config

from typing import Callable, Tuple, Any, Dict

import matplotlib
import matplotlib.pyplot as plt

import torch



############################################################################################################################################
################################################################## TASK ####################################################################
############################################################################################################################################

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# Task superclass capturing common structure of data generation and presentation to 'RNN'                                                    #
# See 'Tasks' directory for instances                                                                                                      #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

# Global registry of tasks 
# Keys are task names, values are global task objecs (see any file in Tasks directory)
task_register = {}

class Task:
    '''
    __init__
    Create Task with specified parameters and (possibly) add to register
    ---------------------------------------------------------------------------------------------
    Receives
        name : 
            Name of task by which it will be known in the registry
        task_specific_params :
            Parameters to be used in the task (as opposed to general params of config.py)
        create_data_func :
            Function for generating data for task (receives task config object, input, target, and mask tensor; 
            returns filled input, target, mask tensors, and a dictionary of task variables associated )
        input_map :
            Mapping of task variable name to index in input data tensors
        target_map :
            Mapping of task variable name to index in target data tensors
        loss_func (optional) :
            Loss function (receives Task, 'RNN', batch objects; returns loss and output tensors) to use with task
            If not specified, a default regularised MSE is used (see below)
        test_func (optional) :
            Testing function (receives Task, 'RNN', batch objects) to generate figures
            If not specified, a default is used (see test_funcs.py)
        test_func_args (optional) :
            Addional arguments to be supplied to the testing function
        init_func (optional) :
            Function called at init for config-dependent setup
        register (default=True) :
            Flag indicating whether or not Task instance should be added to task_register
            In general, should be True for global Task objects and False for local copies
    '''
    def __init__(self, 
                 name: str,
                 task_specific_params: dict, 
                 create_data_func: Callable[[Config, torch.Tensor, torch.Tensor], 
                                            Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]],
                 input_map: dict, 
                 target_map:dict, 
                 loss_func: Callable[['Task', 'RNN', dict], Tuple[torch.Tensor, torch.Tensor]] = None, 
                 test_func: Callable[['Task', 'RNN', dict], Dict[str, matplotlib.figure.Figure]] = None, 
                 test_func_args:dict = {}, 
                 init_func: Callable[['Task'], Any] = None, 
                 register: bool = True):
        
        self.name = name
        self.task_specific_params = task_specific_params
        self.create_data_func = create_data_func
        self.input_map = input_map
        self.target_map = target_map

        # Generate config object which includes specified parameters
        self.config = Config(task=name, n_inputs=len(input_map), n_outputs=len(target_map), **task_specific_params)

        # Save default/supplied functions
        self.init_func = init_func
        if init_func is not None:
            init_func(self)

        if loss_func is not None:
            self.loss_func = loss_func
        else:
            self.loss_func = default_loss_func

        if test_func is not None:
            self.test_func = test_func
        else:
            from test_funcs import test_general
            self.test_func = test_general
        self.test_func_args = test_func_args

        # Register if desired
        if register:
            self.register()

    # Wrapper function for calling loss function
    def get_loss(self, net: 'RNN', batch: dict):
        return self.loss_func(task=self, net=net, batch=batch)

    # Add this object to global task register
    def register(self):
        task_register[self.name] = self

    # Create a copy of this object (usually to make a local copy of a global (i.e. registered) task object)
    def copy(self, **kwargs) -> Any:
        task_args = dict(
            name=self.name, 
            task_specific_params=self.task_specific_params, 
            create_data_func=self.create_data_func, init_func=self.init_func, loss_func=self.loss_func, 
            test_func=self.test_func, test_func_args=self.test_func_args, 
            input_map=self.input_map, target_map=self.target_map,
            register=False
        )
        task_args.update(**{k:v for k,v in kwargs.items() if k in task_args})
        copy = Task(**task_args)
        copy.config.update(**{k:v for k,v in self.config.dict.items() if k not in task_args})
        copy.config.update(**{k:v for k,v in kwargs.items() if k in copy.config.dict})
        copy.config.task = self.name
        return copy

    # Create an instance of a task from that saved in a checkpoint
    @classmethod
    def from_checkpoint(self, checkpoint: dict) -> Any:
        print(task_register.keys())
        task = task_register[checkpoint['config']['task']]
        copy = Task(task.name, task.task_specific_params, task.create_data_func, 
                    init_func=task.init_func, loss_func=task.loss_func, test_func=task.test_func, test_func_args=task.test_func_args, 
                    input_map=task.input_map, target_map=task.target_map)
        copy.config.update(**checkpoint['config'])
        return copy
    
    @classmethod
    def named(self, tname: str, **kwargs) -> Any:
        try:
            task = task_register[tname].copy(**kwargs)
            return task
        except KeyError:
            print(f'No task named {tname}')
            return None




# Default loss function
# MSE of output + L2 regularised rates + L2 regularised weights
def default_loss_func(task: Task, net: 'RNN', batch: dict) -> Tuple[torch.tensor, torch.tensor]:
    _, activity, outputs = net(batch['inputs'], noise=batch['noise'])

    # MSE of (masked) outputs
    loss_prediction = torch.sum(torch.square(outputs - batch['targets'])[batch['mask']]) / torch.sum(batch['mask']==1)

    # Rate Loss
    if task.config.rate_loss_type == 1:
        loss_activity = task.config.rate_lambda * torch.mean(torch.abs(activity))
    else:
        loss_activity = task.config.rate_lambda * torch.mean(torch.square(activity))

    # Weight Loss
    if task.config.weight_loss_type == 1:
        loss_weight = 0
        for weight, include in zip([net.W_rec.weight, net.W_in.weight, net.W_out.weight], [task.config.learn_W_rec, task.config.learn_W_in, task.config.learn_W_out]):
            if include:
                loss_weight += task.config.weight_lambda * torch.mean(torch.abs(weight))
    else:
        loss_weight = 0
        for weight, include in zip([net.W_rec.weight, net.W_in.weight, net.W_out.weight], [task.config.learn_W_rec, task.config.learn_W_in, task.config.learn_W_out]):
            if include:
                loss_weight += task.config.weight_lambda * torch.mean(torch.square(weight))

    loss = loss_prediction + loss_activity + loss_weight

    return loss, outputs

# def rate_l2_weight_l1_loss_func(task: Task, net: 'RNN', batch: dict) -> Tuple[torch.tensor, torch.tensor]:
    _, activity, outputs = net(batch['inputs'], noise=batch['noise'])

    # MSE of (masked) outputs
    loss_prediction = torch.sum(torch.square(outputs - batch['targets'])[batch['mask']]) / torch.sum(batch['mask']==1)

    # Rate L2
    loss_activity = task.config.rate_lambda * torch.mean(activity)

    # Weight L1
    loss_weight = task.config.weight_lambda * (torch.mean(torch.abs(net.W_rec.weight)) + 
                                               torch.mean(torch.abs(net.W_in.weight)) + 
                                               torch.mean(torch.abs(net.W_out.weight)))

    loss = loss_prediction + loss_activity + loss_weight

    return loss, outputs