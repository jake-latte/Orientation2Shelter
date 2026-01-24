from o2s.config import Config

from typing import Callable, Tuple, Any, Dict

import matplotlib
import matplotlib.pyplot as plt

import torch
import numpy as np



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
                 get_vars_func: Callable[[Config], dict],
                 create_data_func: Callable[[Config, torch.Tensor, torch.Tensor], 
                                            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
                 input_map: dict, 
                 target_map:dict, 
                 loss_func: Callable[['Task', 'RNN', dict], Tuple[torch.Tensor, torch.Tensor]] = None, 
                 test_func: Callable[['Task', 'RNN', dict], Dict[str, matplotlib.figure.Figure]] = None, 
                 test_func_args:dict = {}, 
                 init_func: Callable[['Task'], Any] = None, 
                 register: bool = True,
                 config: Config = None,
                 get_subtask_vars_funcs: Dict[str, Callable[[Config], dict]]=None) -> None:
        
        self.name = name
        self.task_specific_params = task_specific_params
        self.get_vars_func = get_vars_func
        self.create_data_func = create_data_func
        self.input_map = input_map
        self.target_map = target_map

        # Generate config object which includes specified parameters
        if config is None:
            self.config = Config(task=name, n_inputs=len(input_map), n_outputs=len(target_map), **task_specific_params)
        else:
            self.config = config

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
            from o2s.test import test_general
            self.test_func = test_general
        self.test_func_args = test_func_args

        self.get_subtask_vars_funcs = get_subtask_vars_funcs

        # Register if desired
        if register:
            self.register()

    # Wrapper function for calling loss function
    def get_loss(self, net: 'RNN', batch: dict):
        return self.loss_func(task=self, net=net, batch=batch)

    # Add this object to global task register
    def register(self):
        global task_register
        task_register[self.name] = self

    # Create a copy of this object (usually to make a local copy of a global (i.e. registered) task object)
    def copy(self, **kwargs) -> Any:
        copy_config = self.config.copy()
        copy_config.update(**{k:v for k,v in kwargs.items() if k in copy_config.dict})
        task_args = dict(
            name=self.name, 
            task_specific_params=self.task_specific_params,
            get_vars_func=self.get_vars_func, 
            create_data_func=self.create_data_func, init_func=self.init_func, loss_func=self.loss_func, 
            test_func=self.test_func, test_func_args=self.test_func_args, 
            input_map=self.input_map, target_map=self.target_map,
            register=False, config=copy_config, get_subtask_vars_funcs=self.get_subtask_vars_funcs
        )
        task_args.update(**{k:v for k,v in kwargs.items() if k in task_args})
        copy = Task(**task_args)
        return copy

    # Create an instance of a task from that saved in a checkpoint
    @classmethod
    def from_checkpoint(self, checkpoint: dict) -> Any:
        global task_register
        if not torch.cuda.is_available():
            checkpoint['config']['device'] = 'cpu'
        if ':' in checkpoint['config']['task']:
            task_name, subtask_name = checkpoint['config']['task'].split(':')
            task = task_register[task_name].get_subtask(subtask_name, **checkpoint['config'])
            return task
        else:
            task = task_register[checkpoint['config']['task']].copy(**checkpoint['config'])
        return task
    
    @classmethod
    def named(self, tname: str, **kwargs) -> Any:
        global task_register
        if tname in task_register:
            task = task_register[tname].copy(**kwargs)
            return task
        else:
            raise Exception(f'No task named {tname}')
        
    def get_subtask(self, subtask_name: str, **kwargs) -> Any:
        if subtask_name in self.get_subtask_vars_funcs:
            subtask_get_vars_func = self.get_subtask_vars_funcs[subtask_name]
            subtask_name = f'{self.name}:{subtask_name}'
            subtask = self.copy(get_vars_func=subtask_get_vars_func, **kwargs)
            return subtask
            if 'name' not in kwargs:
                subtask.name = subtask_name
        else:
            raise Exception(f'No subtask named {subtask_name}')




# Default loss function
# MSE of output + L2 regularised rates + L2 regularised weights
def default_loss_func(task: Task, net: 'RNN', batch: dict) -> Tuple[torch.tensor, torch.tensor]:
    _, activity, outputs = net(batch['inputs'], noise=batch['noise'], offload=task.config.conserve_vram, collapse=task.config.conserve_vram)

    # MSE of (masked) outputs
    loss_prediction = torch.sum(torch.square(outputs - batch['targets'])[batch['mask']]) / torch.sum(batch['mask']==1)

    # Rate Loss
    if task.config.conserve_vram:
        loss_activity = task.config.rate_lambda * activity
    else:
        if task.config.rate_loss_type == 1:
            loss_activity = task.config.rate_lambda * torch.sum(torch.abs(activity))
        else:
            loss_activity = task.config.rate_lambda * torch.mean(torch.square(activity))

    # Weight Loss
    if task.config.weight_loss_type == 1:
        loss_weight = 0
        for weight, include in zip([net.W_rec.weight, net.W_in.weight, net.W_out.weight], [task.config.learn_W_rec, task.config.learn_W_in, task.config.learn_W_out]):
            if include:
                loss_weight += task.config.weight_lambda * torch.sum(torch.abs(weight))
    else:
        loss_weight = 0
        for weight, include in zip([net.W_rec.weight, net.W_in.weight, net.W_out.weight], [task.config.learn_W_rec, task.config.learn_W_in, task.config.learn_W_out]):
            if include:
                loss_weight += task.config.weight_lambda * torch.mean(torch.square(weight))

    loss = loss_prediction + loss_activity + loss_weight

    return loss, outputs