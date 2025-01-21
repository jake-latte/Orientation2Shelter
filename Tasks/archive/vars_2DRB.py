import torch

import sys

from ratinabox.Environment import Environment
from ratinabox.Agent import Agent

from task import *
from build import *
from test_funcs import *


default_params = {
    **Tasks.vars_0D.default_params,

    'batch_size': 250,
    'n_timesteps': 1000,
    'num_batch_repeats': 15,

    'test_n_timesteps': 1000,
    'test_batch_size': 1000
}


input_map = {
    'av': 0,
    'sin_hd_0': 1,
    'cos_hd_0': 2,
    'sx': 3,
    'sy': 4,
    'x_0': 5,
    'y_0': 6,
    'v': 7
}

def create_data(config):
    # Create local copies of parameter properties (for brevity's sake)
    batch_size, n_timesteps = config.batch_size, config.n_timesteps
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

    linear_velocity = torch.sqrt(velocity[:,angle_0_duration:,0]**2 + velocity[:,angle_0_duration:,1]**2)
                            
    shelter_x_0 = torch.rand(batch_size)
    shelter_y_0 = torch.rand(batch_size)
    shelter_wall_select_probs = torch.rand(batch_size)
    shelter_wall_select_horizontal = torch.where(shelter_wall_select_probs>0.5)[0]
    shelter_wall_select_vertical = torch.where(shelter_wall_select_probs<=0.5)[0]
    shelter_wall_select_directions = torch.randint(low=0, high=2, size=(batch_size,), dtype=shelter_x_0.dtype)

    shelter_x_0[shelter_wall_select_vertical] = shelter_wall_select_directions[shelter_wall_select_vertical]
    shelter_y_0[shelter_wall_select_horizontal] = shelter_wall_select_directions[shelter_wall_select_horizontal]

    shelter_x = torch.tile(shelter_x_0.reshape(batch_size,1), dims=(1,n_timesteps))
    shelter_y = torch.tile(shelter_y_0.reshape(batch_size,1), dims=(1,n_timesteps))

    d_x = shelter_x - position[:,:,0]
    d_y = shelter_y - position[:,:,1]
    dist = torch.sqrt(d_x**2 + d_y**2)
    pert = 10e-6 * torch.ones((batch_size,n_timesteps))
    dist[torch.where(dist==0)[0]] += (pert * np.random.choice([1, -1]))[torch.where(dist==0)[0]]
    allo_shelter_angle = torch.atan2(d_y, d_x)

    ego_angle = allo_shelter_angle - head_direction

    return {
        'av': angular_velocity,
        'hd': head_direction,
        'sd': ego_angle,
        'x': position[:,:,0],
        'y': position[:,:,1],
        'v': linear_velocity
    }


# Save input and target data streams
# inputs[:,:,input_map['av']] = angular_velocity
# inputs[:,:angle_0_duration,input_map['sin_hd_0']] = sin_theta_0
# inputs[:,:angle_0_duration,input_map['cos_hd_0']] = cos_theta_0
# inputs[:,:config.angle_0_duration,input_map['sx']] = shelter_x[:,:config.angle_0_duration]
# inputs[:,:config.angle_0_duration,input_map['sy']] = shelter_y[:,:config.angle_0_duration]
# inputs[:,:angle_0_duration,input_map['x_0']] = x_0
# inputs[:,:angle_0_duration,input_map['y_0']] = y_0
# inputs[:,angle_0_duration:,input_map['v']] = linear_velocity


