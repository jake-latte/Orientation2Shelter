import torch

import sys

from ratinabox.Environment import Environment
from ratinabox.Agent import Agent

from task import *
from build import *
from test_funcs import *


default_params = {
    **Tasks.allo0D.default_params,

    'batch_size': 250,
    'n_timesteps': 2000,
    'num_batch_repeats': 15,

    'test_n_timesteps': 2000,
    'test_batch_size': 1000
}


input_map = {
    'av': 0,
    'sin_hd_0': 1,
    'cos_hd_0': 2,
    'x_0': 3,
    'y_0': 4,
    'v': 5
}

target_map = {
    'sin_hd': 0,
    'cos_hd': 1,
    'x': 2,
    'y': 3
}

def create_data(config, inputs, targets, mask):
    # Create local copies of parameter properties (for brevity's sake)
    batch_size, n_timesteps = inputs.shape[0], inputs.shape[1]
    angle_0_duration = config.angle_0_duration

    angle = torch.zeros((batch_size, n_timesteps, 2))
    position = torch.zeros((batch_size, n_timesteps, 2))
    velocity = torch.zeros((batch_size, n_timesteps, 2))

    # Generate dataset
    for i in range(batch_size):
        box = Environment(params={
            'dimensionality': '2D',
            'boundary': [[0, 0], [1, 0], [1, 1], [0, 1]]})
        rat = Agent(box, params={
            'dt': config.dt / 1000,})
        
        box.add_wall([[0, 0], [1, 0]])
        box.add_wall([[1, 0], [1, 1]])
        box.add_wall([[1, 1], [0, 1]])
        box.add_wall([[0, 1], [0, 0]])

        for t in range(angle_0_duration, n_timesteps):
            rat.update()

        position[i,:angle_0_duration] = torch.tensor(rat.history['pos'][0])
        position[i,angle_0_duration:] = torch.tensor(rat.history['pos'])

        velocity[i,angle_0_duration:], = torch.gradient(torch.tensor(rat.history['pos']), dim=0)

        angle[i,:angle_0_duration] = torch.tensor(rat.history['head_direction'][0])
        angle[i,angle_0_duration:] = torch.tensor(rat.history['head_direction'])

    x_0 = position[:,:angle_0_duration,0]
    y_0 = position[:,:angle_0_duration,1]

    head_direction = torch.atan2(angle[:,:,1], angle[:,:,0])
    head_direction[head_direction<0] += 2*np.pi

    sin_theta_0 = angle[:,:angle_0_duration,1]
    cos_theta_0 = angle[:,:angle_0_duration,0]

    k = np.pi
    angular_velocity = torch.cat((torch.zeros((batch_size,1)), torch.diff(head_direction, dim=1)), dim=1)
    angular_velocity = torch.where(angular_velocity>k, angular_velocity - 2*np.pi, angular_velocity)
    angular_velocity = torch.where(angular_velocity<-k, angular_velocity + 2*np.pi, angular_velocity)

    # Save input and target data streams
    inputs[:,:,input_map['av']] = angular_velocity
    inputs[:,:angle_0_duration,input_map['sin_hd_0']] = sin_theta_0
    inputs[:,:angle_0_duration,input_map['cos_hd_0']] = cos_theta_0
    inputs[:,:angle_0_duration,input_map['x_0']] = x_0
    inputs[:,:angle_0_duration,input_map['y_0']] = y_0
    inputs[:,angle_0_duration:,input_map['v']] = torch.sqrt(velocity[:,angle_0_duration:,0]**2 + velocity[:,angle_0_duration:,1]**2)

    targets[:,:,target_map['sin_hd']] = angle[:,:,1]
    targets[:,:,target_map['cos_hd']] = angle[:,:,0]
    targets[:,:,target_map['x']] = position[:,:,0]
    targets[:,:,target_map['y']] = position[:,:,1]

    return inputs, targets, mask

ALLO2D_TASK = Task('allo2D', 
                    n_inputs=6, n_outputs=2, 
                    task_specific_params=default_params, 
                    create_data_func=create_data,
                    test_func=test_allo,
                    input_map=input_map,
                    target_map=target_map)





