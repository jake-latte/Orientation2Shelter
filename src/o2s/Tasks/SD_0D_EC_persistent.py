import torch
import numpy as np
import sys
from typing import Dict
import o2s
import o2s.Templates.vars_0D as template_0D
import o2s.Templates.vars_EC as template_EC

class SD_0D_EC_persistent(template_0D.Vars0D, template_EC.VarsEC):
    task_name = "SD-0D_EC_persistent"
    input_map = {
        'av': 0
    }
    target_map = {
        'sin_sd': 0,
        'cos_sd': 1,
    }
    default_params = {
        **template_0D.default_params,
        'n_position_place_cells': 0,
        'n_shelter_place_cells': 25,
        'n_head_direction_cells': 25,
        'head_direction_cell_scale': template_EC.default_params['head_direction_cell_scale'],
        'place_cell_scale': template_EC.default_params['place_cell_scale'],
        'input_noise_std': 0.1
    }
    get_vars = staticmethod(template_0D.get_vars)
    get_joint_vars = staticmethod(template_0D.get_joint_vars)
    get_hd_iso_vars = staticmethod(template_0D.get_hd_iso_vars)
    get_sd_iso_vars = staticmethod(template_0D.get_sd_iso_vars)
    get_av_vars = staticmethod(template_0D.get_av_vars)
    get_metric_vars = staticmethod(template_0D.get_metric_vars)
    test_func = o2s.test.test_gamut
    test_func_args = dict(tuning_vars_list=['HD', 'ego_SD', 'allo_SD', 'AV', 'HD'])
    init_func = template_EC.init_func
    

    @staticmethod
    def create_data(config, vars, inputs, targets, mask):
        init_duration, batch_size = config.init_duration, inputs.shape[0]

        def fill_persistent_head_direction_cell_inputs(config: o2s.config.Config, vars: Dict[str, torch.Tensor], inputs: torch.Tensor) -> torch.Tensor:
            batch_size, n_timesteps = inputs.shape[0], inputs.shape[1]
            init_duration, input_noise_std = config.init_duration, config.input_noise_std
        
            head_direction = vars['hd']
            head_direction_noise = torch.normal(mean=0, std=input_noise_std, size=(batch_size, n_timesteps))
            head_direction_noise[:,:init_duration] = 0
        
            inputs[:,:,-config.n_head_direction_cells:] = template_EC.HD_activity(
                Theta=head_direction + head_direction_noise, 
                mu_h=config.head_direction_cell_centers.to(head_direction.device), scale=config.head_direction_cell_scale) 
            
            return inputs
        
        def fill_persistent_place_cell_inputs(config: o2s.config.Config, vars: Dict[str, torch.Tensor], inputs: torch.Tensor) -> torch.Tensor:
            batch_size, n_timesteps = inputs.shape[0], inputs.shape[1]
            init_duration, input_noise_std = config.init_duration, config.input_noise_std
        
            sx = vars['sx'].reshape((batch_size,1)).repeat((1,n_timesteps))
            sy = vars['sy'].reshape((batch_size,1)).repeat((1,n_timesteps))
        
            if 'x' in vars and 'y' not in vars:
                x = torch.cos(vars['x'])
                y = torch.sin(vars['x'])
            elif 'x' in vars and 'y' in vars:
                x = vars['x']
                y = vars['y']
            else:
                x, y = None, None
                assert config.n_position_place_cells == 0

            x_noise = torch.normal(mean=0, std=input_noise_std, size=(batch_size, n_timesteps))
            x_noise[:,:init_duration] = 0
            y_noise = torch.normal(mean=0, std=input_noise_std, size=(batch_size, n_timesteps))
            y_noise[:,:init_duration] = 0
        
            if config.n_position_place_cells > 0:
                start_i = config.n_other_inputs
                end_i = config.n_other_inputs + config.n_position_place_cells
                inputs[:,:,start_i:end_i] = template_EC.PC_activity(X=x + x_noise, Y=y + y_noise, mu_c=config.position_place_cell_centers.to(x.device), sigma_c=config.place_cell_scale)
            if config.n_shelter_place_cells > 0:
                start_i = config.n_other_inputs + config.n_position_place_cells
                end_i = config.n_other_inputs + config.n_position_place_cells + config.n_shelter_place_cells
                inputs[:,:,start_i:end_i] = template_EC.PC_activity(X=sx + x_noise, Y=sy + y_noise, mu_c=config.shelter_place_cell_centers.to(sx.device), sigma_c=config.place_cell_scale)
            
            return inputs
    
        inputs = fill_persistent_head_direction_cell_inputs(config, vars, inputs)
        inputs = fill_persistent_place_cell_inputs(config, vars, inputs)
    
        mask[:,:init_duration] = False
    
        inputs[:,:,SD_0D_EC_persistent.input_map['av']] = vars['av']
    
        targets[:,:,SD_0D_EC_persistent.target_map['sin_sd']] = torch.sin(vars['sd'])
        targets[:,:,SD_0D_EC_persistent.target_map['cos_sd']] = torch.cos(vars['sd'])
    
        return inputs, targets, mask
    def __init__(self, **kwargs):
        self.get_vars_func = self.get_vars
        self.get_subtask_vars_funcs = {
            "joint": self.get_joint_vars,
            "hd_iso": self.get_hd_iso_vars,
            "sd_iso": self.get_sd_iso_vars,
            "av": self.get_av_vars,
            "metric": self.get_metric_vars,
        }
        super().__init__(
            name=self.task_name,
            task_specific_params=self.default_params,
            get_vars_func=self.get_vars_func,
            create_data_func=self.create_data,
            input_map=self.input_map,
            target_map=self.target_map,
            test_func=self.test_func,
            test_func_args=self.test_func_args,
            init_func=self.init_func,
            get_subtask_vars_funcs=self.get_subtask_vars_funcs,
            **kwargs
        )

# input_map = {
#     'av': 0
# }
#
# target_map = {
#     'sin_sd': 0,
#     'cos_sd': 1,
# }
#
# default_params = {
#     **template_0D.default_params,
#     'n_position_place_cells': 0,
#     'n_shelter_place_cells': 25,
#     'n_head_direction_cells': 25,
#     'head_direction_cell_scale': template_EC.default_params['head_direction_cell_scale'],
#     'place_cell_scale': template_EC.default_params['place_cell_scale'],
#     'intermediate_output_dim': 25
# }
#
# def fill_persistent_head_direction_cell_inputs(config: o2s.config.Config, vars: Dict[str, torch.Tensor], inputs: torch.Tensor) -> torch.Tensor:
#     init_duration, batch_size = config.init_duration, inputs.shape[0]
#
#     head_direction = vars['hd']
#
#     inputs[:,:,-config.n_head_direction_cells:] = template_EC.HD_activity(
#         Theta=head_direction, 
#         mu_h=config.head_direction_cell_centers.to(head_direction.device), scale=config.head_direction_cell_scale) 
#
#     return inputs
#
# def fill_persistent_place_cell_inputs(config: o2s.config.Config, vars: Dict[str, torch.Tensor], inputs: torch.Tensor) -> torch.Tensor:
#     init_duration, batch_size, n_timesteps = config.init_duration, inputs.shape[0], inputs.shape[1]
#
#     sx = vars['sx'].reshape((batch_size,1)).repeat((1,n_timesteps))
#     sy = vars['sy'].reshape((batch_size,1)).repeat((1,n_timesteps))
#
#     if 'x' in vars and 'y' not in vars:
#         x = torch.cos(vars['x'])
#         y = torch.sin(vars['x'])
#     elif 'x' in vars and 'y' in vars:
#         x = vars['x']
#         y = vars['y']
#     else:
#         x, y = None, None
#         assert config.n_position_place_cells == 0
#
#     if config.n_position_place_cells > 0:
#         start_i = config.n_other_inputs
#         end_i = config.n_other_inputs + config.n_position_place_cells
#         inputs[:,:,start_i:end_i] = template_EC.PC_activity(X=x, Y=y, mu_c=config.position_place_cell_centers.to(x.device), sigma_c=config.place_cell_scale)
#     if config.n_shelter_place_cells > 0:
#         start_i = config.n_other_inputs + config.n_position_place_cells
#         end_i = config.n_other_inputs + config.n_position_place_cells + config.n_shelter_place_cells
#         inputs[:,:,start_i:end_i] = template_EC.PC_activity(X=sx, Y=sy, mu_c=config.shelter_place_cell_centers.to(sx.device), sigma_c=config.place_cell_scale)
#
#     return inputs
#
#
# def create_data(config, vars, inputs, targets, mask):
#     init_duration, batch_size = config.init_duration, inputs.shape[0]
#
#     inputs = fill_persistent_head_direction_cell_inputs(config, vars, inputs)
#     inputs = fill_persistent_place_cell_inputs(config, vars, inputs)
#
#     mask[:,:init_duration] = False
#
#     inputs[:,:,input_map['av']] = vars['av']
#
#     targets[:,:,target_map['sin_sd']] = torch.sin(vars['sd'])
#     targets[:,:,target_map['cos_sd']] = torch.cos(vars['sd'])
#
#     return inputs, targets, mask
#
#
#
# SD_0D_EC_persistent_TASK = o2s.task.Task('SD-0D_EC_persistent',
#                     task_specific_params=default_params,
#                     get_vars_func=template_0D.get_vars, 
#                     create_data_func=create_data,
#                     init_func=template_EC.init_func,
#                     input_map=input_map,
#                     target_map=target_map,
#                     test_func=o2s.test.test_gamut,
#                     test_func_args=dict(tuning_vars_list=['HD', 'ego_SD', 'allo_SD', 'AV', 'HD']),
#                     get_subtask_vars_funcs={'joint': template_0D.get_joint_vars,
#                                             'hd_iso': template_0D.get_hd_iso_vars,
#                                             'sd_iso': template_0D.get_sd_iso_vars,
#                                             'av': template_0D.get_av_vars,
#                                             'metric': template_0D.get_metric_vars})
#
#
