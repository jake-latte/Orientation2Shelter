import torch
import sys
import o2s

class MULT(o2s.task.Task):
    task_name = "MULT"
    default_params = {
        'min_xy': -10,
        'max_xy': 10,
        'init_duration': 10,
        'loss_time': 50
    }
    input_map = {
        'x': 0,
        'y': 1
    }
    target_map = {
        'xy': 0
    }
    @staticmethod
    def get_vars(config):
        batch_size = config.batch_size
        
        x_vals = (config.max_xy-config.min_xy)*torch.rand((batch_size,)) + config.min_xy
        y_vals = (config.max_xy-config.min_xy)*torch.rand((batch_size,)) + config.min_xy
    
        return {'x': x_vals, 'y': y_vals}
    @staticmethod
    def create_data(config, vars, inputs, targets, mask):
        batch_size, n_timesteps = config.batch_size, config.n_timesteps
        init_duration = config.init_duration
    
        x_vals, y_vals = vars['x'], vars['y']
    
        inputs[:,:init_duration,MULT.input_map['x']] = x_vals.reshape((batch_size,1)).repeat((1,init_duration))
        inputs[:,:init_duration,MULT.input_map['y']] = y_vals.reshape((batch_size,1)).repeat((1,init_duration))
    
        targets[:,:,MULT.target_map['xy']] = (x_vals*y_vals).reshape((batch_size,1)).repeat((1,n_timesteps))
    
        mask[:,:,:] = False
        mask[:,config.loss_time:] = True
    
    
        return inputs, targets, mask
    def __init__(self, **kwargs):
        self.get_vars_func = self.get_vars
        self.get_subtask_vars_funcs = {}
        super().__init__(
            name=self.task_name,
            task_specific_params=self.default_params,
            get_vars_func=self.get_vars_func,
            create_data_func=self.create_data,
            input_map=self.input_map,
            target_map=self.target_map,
            **kwargs
        )

# default_params = {
#     'min_xy': -10,
#     'max_xy': 10,
#     'init_duration': 10,
#     'loss_time': 50
# }
#
# input_map = {
#     'x': 0,
#     'y': 1
# }
#
# target_map = {
#     'xy': 0
# }
#
# def get_vars(config):
#     batch_size = config.batch_size
#
#     x_vals = (config.max_xy-config.min_xy)*torch.rand((batch_size,)) + config.min_xy
#     y_vals = (config.max_xy-config.min_xy)*torch.rand((batch_size,)) + config.min_xy
#
#     return {'x': x_vals, 'y': y_vals}
#
# def create_data(config, vars, inputs, targets, mask):
#     batch_size, n_timesteps = config.batch_size, config.n_timesteps
#     init_duration = config.init_duration
#
#     x_vals, y_vals = vars['x'], vars['y']
#
#     inputs[:,:init_duration,input_map['x']] = x_vals.reshape((batch_size,1)).repeat((1,init_duration))
#     inputs[:,:init_duration,input_map['y']] = y_vals.reshape((batch_size,1)).repeat((1,init_duration))
#
#     targets[:,:,target_map['xy']] = (x_vals*y_vals).reshape((batch_size,1)).repeat((1,n_timesteps))
#
#     mask[:,:,:] = False
#     mask[:,config.loss_time:] = True
#
#
#     return inputs, targets, mask
#
#
# MULT_TASK = o2s.task.Task('MULT',
#                     task_specific_params=default_params, 
#                     get_vars_func=get_vars,
#                     create_data_func=create_data,
#                     input_map=input_map,
#                     target_map=target_map)
#
#
