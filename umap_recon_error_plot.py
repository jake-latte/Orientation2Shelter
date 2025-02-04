import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from config import *
from data import *
from net import *
from task import *
from test_funcs import *

import Tasks
from Tasks.util import register_all_tasks

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import umap
import multiprocessing as mp
import pandas as pd



def fill_hd_sd_0d_data(config, inputs, targets, vars, mask):
    """
    Fills the input, target, and mask tensors with head direction (hd) and shelter direction (sd) data (which
    data is generated elsewhere)
    Args:
        config (object): Configuration object containing batch_size, n_timesteps, and init_duration.
        inputs (torch.Tensor): Input tensor to be filled.
        targets (torch.Tensor): Target tensor to be filled.
        vars (dict): Dictionary containing variables 'av', 'hd', 'sd', 'sx', and 'sy'.
        mask (torch.Tensor): Mask tensor to be updated.
    Returns:
        tuple: Updated inputs, targets, vars, and mask tensors.
    """

    batch_size, n_timesteps, init_duration = config.batch_size, config.n_timesteps, config.init_duration

    inputs[:,:,Tasks.vars_0D.input_map['av']] = vars['av']
    inputs[:,:init_duration,Tasks.vars_0D.input_map['sin_hd_0']] = torch.sin(vars['hd'][:,0]).reshape((batch_size,1)).repeat((1,init_duration))
    inputs[:,:init_duration,Tasks.vars_0D.input_map['cos_hd_0']] = torch.cos(vars['hd'][:,0]).reshape((batch_size,1)).repeat((1,init_duration))
    inputs[:,:init_duration,Tasks.vars_0D.input_map['sx']] = vars['sx'].repeat((1,init_duration))
    inputs[:,:init_duration,Tasks.vars_0D.input_map['sy']] = vars['sy'].repeat((1,init_duration))

    mask[:,:init_duration] = False

    targets[:,:,Tasks.HD_SD_0D.target_map['sin_hd']] = torch.sin(vars['hd'])
    targets[:,:,Tasks.HD_SD_0D.target_map['cos_hd']] = torch.cos(vars['hd'])
    targets[:,:,Tasks.HD_SD_0D.target_map['sin_sd']] = torch.sin(vars['sd'])
    targets[:,:,Tasks.HD_SD_0D.target_map['cos_sd']] = torch.cos(vars['sd'])

    return inputs, targets, vars, mask


def create_hd_sd_iso_data(config, inputs, targets, mask):
    batch_size, n_timesteps, init_duration = config.batch_size, config.n_timesteps, config.init_duration

    n_steps = int(np.sqrt(batch_size))
    assert batch_size == n_steps**2

    joint_range = torch.meshgrid(torch.linspace(0, 2*np.pi, n_steps), torch.linspace(0, 2*np.pi, n_steps))
    hd_range, sd_range = joint_range[0].flatten(), joint_range[1].flatten()

    head_direction = hd_range.reshape((batch_size, 1)).repeat((1, n_timesteps))
    ego_shelter_angle = sd_range.reshape((batch_size, 1)).repeat((1, n_timesteps))

    allo_shelter_angle = torch.remainder(head_direction + ego_shelter_angle, 2*np.pi)[:,0].reshape((batch_size,1))

    vars = {'av': torch.zeros((batch_size, n_timesteps)), 
            'hd': head_direction, 
            'sd': ego_shelter_angle,
            'sx': torch.cos(allo_shelter_angle), 
            'sy': torch.sin(allo_shelter_angle)}
    
    return fill_hd_sd_0d_data(config, inputs, targets, vars, mask)


def compute_umap_reconstruction_error(args):
    t, d, joint_batch_activity = args
    print(f'Computing UMAP reconstruction error for t={t}, d={d}')
    umapper = Pipeline([
        ("scaler", StandardScaler()),
        ("umap_reducer", umap.UMAP(
            n_components=d, 
            n_neighbors=15,
            random_state=0,
            verbose=False
        ))
    ])
    activity = joint_batch_activity[:, t].reshape((-1, 100))
    activity_down = umapper.fit_transform(activity)
    activity_up = umapper.inverse_transform(activity_down)
    mse = mean_squared_error(activity, activity_up)
    return t, d, mse




if __name__ == '__main__':
    register_all_tasks()

    checkpoint_path = 'trained-models/1737566092.4583075-task:HD_SD-0D-rate_lambda:0.01-weight_lambda:0.01/checkpoint-epochs:72000/net.pt'
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    task = Task.from_checkpoint(checkpoint=checkpoint)
    task.config.update(device='cpu')
    net = RNN(task)
    net.load_state_dict(checkpoint['net_state_dict'])

    joint_task = Task.named('HD_SD-0D', create_data_func=create_hd_sd_iso_data)
    joint_task.config.update(batch_size=10000, n_timesteps=100, state_noise_std=0)

    joint_batch = TaskDataset(joint_task).get_batch()
    joint_batch_activity = net(joint_batch['inputs'], noise=joint_batch['noise'])[1].detach().numpy()

    times = range(0, 100, 10)
    dimensions = range(2, 11)
    umap_results = np.full((len(times), len(dimensions)), fill_value=np.nan)


    with mp.Pool(processes=mp.cpu_count()) as pool:
        print('Beginning UMAP reconstruction error computation')
        results = pool.map(compute_umap_reconstruction_error, [(t, d, joint_batch_activity) for t in times for d in dimensions])

    for t, d, mse in results:
        i = times.index(t)
        j = dimensions.index(d)
        umap_results[i, j] = mse
        print(f'Completed t={t}, d={d} with mse {mse}')

    fig, ax = plt.subplots()
    cax = ax.imshow(umap_results, interpolation='nearest', cmap='viridis', extent=[2, 10, 0, 90])
    ax.set_title('UMAP Reconstruction Error')
    ax.set_xlabel('Dimensions')
    ax.set_ylabel('Timestep')
    ax.set_xticks(np.arange(len(dimensions)))
    ax.set_xticklabels(dimensions)
    ax.set_yticks(np.arange(len(times)))
    ax.set_yticklabels(times)
    fig.colorbar(cax, ax=ax, label='MSE')

    fig.savefig('umap_reconstruction_error.png')

    # Convert umap_results to a pandas DataFrame
    umap_df = pd.DataFrame(umap_results, index=times, columns=dimensions)

    # Save the DataFrame to a CSV file
    umap_df.to_csv('umap_reconstruction_error.csv')



