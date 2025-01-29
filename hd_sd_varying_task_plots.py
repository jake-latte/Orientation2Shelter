import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import sys
from sklearn.decomposition import PCA

from config import *
from data import *
from net import *
from task import *
from test_funcs import *

import Tasks
from Tasks.util import register_all_tasks
register_all_tasks()

checkpoint_path = sys.argv[1]
checkpoint = torch.load(checkpoint_path, map_location='cpu')
task = Task.from_checkpoint(checkpoint=checkpoint)
task.config.update(device='cpu')
net = RNN(task)
net.load_state_dict(checkpoint['net_state_dict'])


# HD- and SD-varying tasks

def fill_hd_sd_0d_data(config, inputs, targets, vars, mask):
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


def create_hd_varying_data(config, inputs, targets, mask):
    batch_size, n_timesteps, init_duration = config.batch_size, config.n_timesteps, config.init_duration

    head_direction = torch.linspace(0, 2*np.pi, batch_size).reshape((batch_size,1)).repeat((1,n_timesteps))

    ego_shelter_angle = torch.ones((batch_size,1)) * np.pi/2

    allo_shelter_angle = torch.remainder(head_direction + ego_shelter_angle, 2*np.pi)[:,0].reshape((batch_size,1))

    vars = {'av': torch.zeros((batch_size, n_timesteps)), 
            'hd': head_direction, 
            'sd': ego_shelter_angle,
            'sx': torch.cos(allo_shelter_angle), 
            'sy': torch.sin(allo_shelter_angle)}
    
    return fill_hd_sd_0d_data(config, inputs, targets, vars, mask) 


def create_sd_varying_data(config, inputs, targets, mask):
    batch_size, n_timesteps, init_duration = config.batch_size, config.n_timesteps, config.init_duration

    head_direction = torch.ones((batch_size,1)) * np.pi/2

    ego_shelter_angle = torch.linspace(0, 2*np.pi, batch_size).reshape((batch_size,1)).repeat((1,n_timesteps))

    allo_shelter_angle = torch.remainder(head_direction + ego_shelter_angle, 2*np.pi)[:,0].reshape((batch_size,1))

    vars = {'av': torch.zeros((batch_size, n_timesteps)), 
            'hd': head_direction, 
            'sd': ego_shelter_angle,
            'sx': torch.cos(allo_shelter_angle), 
            'sy': torch.sin(allo_shelter_angle)}
    
    return fill_hd_sd_0d_data(config, inputs, targets, vars, mask)


test_batch_size, test_n_timesteps = 10000, 101

hd_iso_task = Task.named('HD_SD-0D', create_data_func=create_hd_varying_data)
hd_iso_task.config.update(batch_size=test_batch_size, n_timesteps=test_n_timesteps, state_noise_std=0, init_duration=10)

sd_iso_task = Task.named('HD_SD-0D', create_data_func=create_sd_varying_data)
sd_iso_task.config.update(batch_size=test_batch_size, n_timesteps=test_n_timesteps, state_noise_std=0, init_duration=10)











# Dimensionality Plots

def get_iso_subspaces_at_time(eval_t, max_components=2, activity=None):

    if activity is None:
        hd_batch = TaskDataset(hd_iso_task).get_batch()
        sd_batch = TaskDataset(sd_iso_task).get_batch()

        hd_activity = net(hd_batch['inputs'], noise=hd_batch['noise'])[1].detach().numpy() 
        sd_activity = net(sd_batch['inputs'], noise=sd_batch['noise'])[1].detach().numpy()
    else:
        hd_batch, sd_batch = None, None
        hd_activity, sd_activity = activity 

    hd_pca = PCA(n_components=max_components)
    hd_pca_activity = hd_pca.fit_transform(hd_activity[:,eval_t].reshape((-1, net.n_neurons)))

    sd_pca = PCA(n_components=max_components)
    sd_pca_activity = sd_pca.fit_transform(sd_activity[:, eval_t].reshape((-1, net.n_neurons)))

    return hd_batch, sd_batch, hd_pca, sd_pca, hd_pca_activity, sd_pca_activity


def dimension_plot_at_time(eval_t):
    hd_batch, sd_batch, hd_pca, sd_pca, hd_pca_activity, sd_pca_activity = get_iso_subspaces_at_time(eval_t, max_components=10)

    
    hd_cmap = matplotlib.colormaps['Blues']
    sd_cmap = matplotlib.colormaps['Oranges']
    norm = matplotlib.colors.Normalize(vmin=0, vmax=2*np.pi)

    fig = plt.figure(figsize=(20,20))
    outer_gs = fig.add_gridspec(nrows=2, ncols=2)

    dim_subfig = fig.add_subfigure(outer_gs[0,:])
    dim_gs = dim_subfig.add_gridspec(ncols=2)
    dim_subfig.suptitle(f'Dimensionality of Model Activity at t={eval_t}', fontsize=20)

    hd_dim_ax = dim_subfig.add_subplot(dim_gs[0,0])

    x = 1 + np.arange(hd_pca.n_components_, dtype=int)
    hd_dim_ax.bar(x, hd_pca.explained_variance_ratio_, color=hd_cmap(0.8))

    hd_dim_ax.set_xlim([0.5, hd_pca.n_components_ + 1])
    hd_dim_ax.set_xticks(x)
    hd_dim_ax.set_xticklabels([f"PC {i+1}" for i in range(hd_pca.n_components_)])
    hd_dim_ax.set_xlabel("Principal Components")
    hd_dim_ax.set_ylabel("Explained Variance Ratio")
    hd_dim_ax.set_title('HD-Varying', fontsize=16)



    sd_dim_ax = dim_subfig.add_subplot(dim_gs[0,1])

    x = 1 + np.arange(sd_pca.n_components_, dtype=int)
    sd_dim_ax.bar(x, sd_pca.explained_variance_ratio_, color=sd_cmap(0.8))

    sd_dim_ax.set_xlim([0.5, sd_pca.n_components_ + 1])
    sd_dim_ax.set_xticks(x)
    sd_dim_ax.set_xticklabels([f"PC {i+1}" for i in range(sd_pca.n_components_)])
    sd_dim_ax.set_xlabel("Principal Components")
    sd_dim_ax.set_ylabel("Explained Variance Ratio")
    sd_dim_ax.set_title('SD-Varying', fontsize=16)



    ring_subfig = fig.add_subfigure(outer_gs[1,:])
    ring_gs = ring_subfig.add_gridspec(ncols=2)
    ring_subfig.suptitle('Activity in First Two Dimensions of PCA', fontsize=20)

    hd_ring_ax = ring_subfig.add_subplot(ring_gs[0,0])

    hd_ring_plot = hd_ring_ax.scatter(hd_pca_activity[:,0], hd_pca_activity[:,1], c=hd_batch['vars']['hd'][:,10], cmap=hd_cmap, norm=norm)

    hd_ring_ax.set_xlabel('PC 1')
    hd_ring_ax.set_xlabel('PC 2')
    hd_ring_ax.set_xticks([])
    hd_ring_ax.set_yticks([])
    hd_ring_ax.set_title('HD-Varying', fontsize=16)

    cbar = fig.colorbar(hd_ring_plot, orientation='horizontal')
    cbar.set_label('Head Direction (HD)')


    sd_ring_ax = ring_subfig.add_subplot(ring_gs[0,1])

    sd_ring_plot = sd_ring_ax.scatter(sd_pca_activity[:,0], sd_pca_activity[:,1], c=sd_batch['vars']['sd'][:,10], cmap=sd_cmap, norm=norm)

    sd_ring_ax.set_xlabel('PC 1')
    sd_ring_ax.set_xlabel('PC 2')
    sd_ring_ax.set_xticks([])
    sd_ring_ax.set_yticks([])
    sd_ring_ax.set_title('SD-Varying', fontsize=16)

    cbar = fig.colorbar(sd_ring_plot, orientation='horizontal')
    cbar.set_label('Head-Shelter Angle (SD)')


    return fig



# Subspace Similarity

def get_iso_vector_similarity_at_time(eval_t, max_components=2):

    # Extract input and output vectors
    in_vectors = net.W_in.weight.detach().numpy()
    out_vectors = net.W_out.weight.detach().T.numpy()

    # Get the principal components
    hd_pca, sd_pca = get_iso_subspaces_at_time(eval_t, max_components=max_components)[2:4]
    pc_vectors = np.concatenate((hd_pca.components_, sd_pca.components_), axis=0).T

    # Collate all vectors into matrices
    all_vectors = np.concatenate((pc_vectors, in_vectors, out_vectors), axis=1)

    # Rescale columns to be length 1
    pc_vectors = pc_vectors / np.linalg.norm(pc_vectors, axis=0, keepdims=True)
    all_vectors = all_vectors / np.linalg.norm(all_vectors, axis=0, keepdims=True)

    # Compute cosine similarity by matrix multiplication
    pc_sim = pc_vectors.T @ all_vectors

    return pc_sim

def get_norms(M_A, M_B):
    Q_A, R_A = np.linalg.qr(M_A, mode='reduced')
    Q_B, R_B = np.linalg.qr(M_B, mode='reduced')

    # 2. Form the r x r product Q_A^T Q_B
    Q_A_t_Q_B = Q_A.T @ Q_B

    # 3. Get singular values of Q_A^T Q_B
    svals = np.linalg.svd(Q_A_t_Q_B, full_matrices=False, compute_uv=False)


    # Spectral norm = largest singular value
    spectral_norm = svals.max()  # or svals[0], as they are returned in descending order

    # Nuclear norm = sum of singular values
    nuclear_norm = svals.sum()

    return spectral_norm, nuclear_norm

def get_iso_subspace_similarity_at_time(eval_t, max_components=2):
    spectral_sim = np.full((2*max_components, 2*max_components), fill_value=np.nan)
    nuclear_sim = np.full((2*max_components, 2*max_components), fill_value=np.nan)
    
    _, _, hd_pca, sd_pca, _, _ = get_iso_subspaces_at_time(eval_t, max_components=max_components)
    hd_subspace = hd_pca.components_.T
    sd_subspace = sd_pca.components_.T 

    in_subspace = net.W_in.weight.detach().numpy()
    out_subspace = net.W_out.weight.detach().T.numpy()

    subspaces = [hd_subspace, sd_subspace, in_subspace, out_subspace]

    for i in range(4):
        for j in range(i+1):
            spectral_norm, nuclear_norm = get_norms(subspaces[i], subspaces[j])
            spectral_sim[i,j] = spectral_norm
            nuclear_sim[i,j] = nuclear_norm

    return spectral_sim, nuclear_sim


def similarity_plot_at_time(eval_t):
    pc_sim = get_iso_vector_similarity_at_time(eval_t=eval_t)
    spectral_sim, nuclear_sim = get_iso_subspace_similarity_at_time(eval_t=eval_t)

    inverse_target_map = {}
    for k,v in hd_iso_task.target_map.items():
        inverse_target_map[v]=k
    inverse_input_map = {}
    for k,v in hd_iso_task.input_map.items():
        inverse_input_map[v]=k


    fig = plt.figure(figsize=(20,15))
    gs = fig.add_gridspec(3, 6, height_ratios=[1,1,0.05], hspace=1)


    sim_cmap = matplotlib.colormaps['cividis']
    sim_norm = matplotlib.colors.Normalize(vmin=-1, vmax=1)

    vector_sim_ax = fig.add_subplot(gs[0,:])

    im1 = vector_sim_ax.imshow(pc_sim, interpolation='nearest', cmap=sim_cmap, norm=sim_norm)
    vector_sim_ax.set_title('Similarity between Principal Components and Feed-Forward Vectors', pad=35, fontsize=16)
    vector_sim_ax.set_xlabel('Vectors')
    vector_sim_ax.set_xticks([i for i in range(4 + net.n_inputs + net.n_outputs)])
    vector_sim_ax.set_xticklabels(['HD PC 1', 'HD PC 2', 'SD PC 1', 'SD PC 2'] + [inverse_input_map[i] for i in range(net.n_inputs)] + [inverse_target_map[i] for i in range(net.n_outputs)], rotation=45, ha='right')
    vector_sim_ax.set_yticks([0,1,2,3])
    vector_sim_ax.set_yticklabels(['HD PC 1', 'HD PC 2', 'SD PC 1', 'SD PC 2'])

    for i in range(pc_sim.shape[0]):
        for j in range(pc_sim.shape[1]):
            vector_sim_ax.text(j, i, f"{pc_sim[i, j]:.2f}", ha="center", va="center", color="white" if im1.norm(pc_sim[i, j]) < 0.5 else "black")

    groups = ['HD PCs', 'SD PCs', 'Inputs', 'Outputs']
    group_sizes = np.array([2, 2, net.n_inputs, net.n_outputs])
    group_starts = np.cumsum(group_sizes)
    vector_sim_ax.vlines(group_starts[:-1]-0.5, ymin=-0.5, ymax=3.5, color='black', linewidth=3, linestyle='--')
    for i, group in enumerate(groups):
        vector_sim_ax.text(group_starts[i] - group_sizes[i]/2 - 0.5, -0.75, group, ha='center', va='center', fontsize=12)


    spectral_sim_ax = fig.add_subplot(gs[1,1:3])

    spectral_cmap = matplotlib.colormaps['viridis']
    spectral_norm = matplotlib.colors.Normalize(vmin=0, vmax=1)

    im2 = spectral_sim_ax.imshow(spectral_sim, interpolation='nearest', cmap=spectral_cmap, norm=spectral_norm)
    spectral_sim_ax.set_title('Similarity between Subspaces by\nSpectral Norm')
    spectral_sim_ax.set_xticks([0,1,2,3])
    spectral_sim_ax.set_xticklabels(['HD PCs', 'SD PCs', 'Inputs', 'Outputs'], rotation=45, ha='right')
    spectral_sim_ax.set_yticks([0,1,2,3])
    spectral_sim_ax.set_yticklabels(['HD PCs', 'SD PCs', 'Inputs', 'Outputs'])

    for i in range(spectral_sim.shape[0]):
        for j in range(spectral_sim.shape[1]):
            spectral_sim_ax.text(j, i, f"{spectral_sim[i, j]:.2f}", ha="center", va="center", color="white" if im2.norm(spectral_sim[i, j]) < 0.5 else "black")



    nuclear_sim_ax = fig.add_subplot(gs[1,3:5])

    nuclear_cmap = matplotlib.colormaps['magma']
    nuclear_norm = matplotlib.colors.Normalize(vmin=np.nanmin(nuclear_sim), vmax=np.nanmax(nuclear_sim))

    im3 = nuclear_sim_ax.imshow(nuclear_sim, interpolation='nearest', cmap=nuclear_cmap, norm=nuclear_norm)
    nuclear_sim_ax.set_title('Similarity between Subspaces by\nNuclear Norm')
    nuclear_sim_ax.set_xticks([0,1,2,3])
    nuclear_sim_ax.set_xticklabels(['HD PCs', 'SD PCs', 'Inputs', 'Outputs'], rotation=45, ha='right')
    nuclear_sim_ax.set_yticks([0,1,2,3])
    nuclear_sim_ax.set_yticklabels(['HD PCs', 'SD PCs', 'Inputs', 'Outputs'])

    for i in range(nuclear_sim.shape[0]):
        for j in range(nuclear_sim.shape[1]):
            nuclear_sim_ax.text(j, i, f"{nuclear_sim[i, j]:.2f}", ha="center", va="center", color="white" if im3.norm(nuclear_sim[i, j]) < 0.5 else "black")


    cax1 = fig.add_subplot(gs[2,0:2])
    cbar1 = fig.colorbar(im1, ax=vector_sim_ax, cax=cax1, orientation='horizontal', fraction=0.02, pad=0.1)
    cbar1.set_label('Cosine of Angle between Vectors')
    cbar1.set_ticks([sim_norm.vmin, sim_norm.vmax])
    cbar1.set_ticklabels([f"{sim_norm.vmin:.2f}", f"{sim_norm.vmax:.2f}"])

    cax2 = fig.add_subplot(gs[2,2:4])
    cbar2 = fig.colorbar(im2, ax=spectral_sim_ax, cax=cax2, orientation='horizontal', fraction=0.02, pad=0.1)
    cbar2.set_label("Maximum Cosine of Principal Angles Between Subspaces")
    cbar2.set_ticks([spectral_norm.vmin, spectral_norm.vmax])
    cbar2.set_ticklabels([f"{spectral_norm.vmin:.2f}", f"{spectral_norm.vmax:.2f}"])

    cax3 = fig.add_subplot(gs[2,4:6])
    cbar3 = fig.colorbar(im3, ax=nuclear_sim_ax, cax=cax3, orientation='horizontal', fraction=0.02, pad=0.1)
    cbar3.set_label("Sum of Cosines of Principal Angles Between Subspaces")
    cbar3.set_ticks([nuclear_norm.vmin, nuclear_norm.vmax])
    cbar3.set_ticklabels([f"{nuclear_norm.vmin:.2f}", f"{nuclear_norm.vmax:.2f}"])

    fig.suptitle(f'Similarity Between HD-Varying and SD-Varying\nSubspace Dimensions and Inputs and Outputs\nat time t={eval_t}', fontsize=20)
    fig.subplots_adjust(top=0.8)

    return fig




def get_iso_subspace_variance_across_time(max_dim=5):
    assert hd_iso_task.config.n_timesteps == sd_iso_task.config.n_timesteps
    n_timesteps = hd_iso_task.config.n_timesteps

    hd_dim_variance = np.full((max_dim, n_timesteps), fill_value=np.nan)
    sd_dim_variance = np.full((max_dim, n_timesteps), fill_value=np.nan)

    hd_batch = TaskDataset(hd_iso_task).get_batch()
    sd_batch = TaskDataset(sd_iso_task).get_batch()
    hd_activity = net(hd_batch['inputs'], noise=hd_batch['noise'])[1].detach().numpy() 
    sd_activity = net(sd_batch['inputs'], noise=sd_batch['noise'])[1].detach().numpy()

    for t in range(n_timesteps):
        hd_pca, sd_pca = get_iso_subspaces_at_time(t, max_components=max_dim, activity=(hd_activity, sd_activity))[2:4]

        hd_dim_variance[:,t] = np.cumsum(hd_pca.explained_variance_ratio_)
        sd_dim_variance[:,t] = np.cumsum(sd_pca.explained_variance_ratio_)

    return hd_dim_variance, sd_dim_variance

def get_iso_subspace_self_similarity_across_time(dim=2):
    assert hd_iso_task.config.n_timesteps == sd_iso_task.config.n_timesteps
    n_timesteps = hd_iso_task.config.n_timesteps

    hd_spectral_norms = np.full((n_timesteps, n_timesteps), fill_value=np.nan)
    sd_spectral_norms = np.full((n_timesteps, n_timesteps), fill_value=np.nan)

    hd_batch = TaskDataset(hd_iso_task).get_batch()
    sd_batch = TaskDataset(sd_iso_task).get_batch()
    hd_activity = net(hd_batch['inputs'], noise=hd_batch['noise'])[1].detach().numpy() 
    sd_activity = net(sd_batch['inputs'], noise=sd_batch['noise'])[1].detach().numpy()

    for i in range(n_timesteps):
        hd_pca_i, sd_pca_i = get_iso_subspaces_at_time(i, max_components=dim, activity=(hd_activity, sd_activity))[2:4]
        hd_subspace_i, sd_subspace_i = hd_pca_i.components_.T, sd_pca_i.components_.T

        for j in range(i+1):
            hd_pca_j, sd_pca_j = get_iso_subspaces_at_time(j, max_components=dim, activity=(hd_activity, sd_activity))[2:4]
            hd_subspace_j, sd_subspace_j = hd_pca_j.components_.T, sd_pca_j.components_.T

            hd_spectral_norm = get_norms(hd_subspace_i, hd_subspace_j)[0]
            sd_spectral_norm = get_norms(sd_subspace_i, sd_subspace_j)[0]

            hd_spectral_norms[i, j] = hd_spectral_norm
            sd_spectral_norms[i, j] = sd_spectral_norm

    return hd_spectral_norms, sd_spectral_norms

def get_iso_subspace_similarity_across_time(dim=2):
    assert hd_iso_task.config.n_timesteps == sd_iso_task.config.n_timesteps
    n_timesteps = hd_iso_task.config.n_timesteps

    hd_spectral_norms = np.full((n_timesteps, 3), fill_value=np.nan)
    sd_spectral_norms = np.full((n_timesteps, 3), fill_value=np.nan)

    hd_batch = TaskDataset(hd_iso_task).get_batch()
    sd_batch = TaskDataset(sd_iso_task).get_batch()
    hd_activity = net(hd_batch['inputs'], noise=hd_batch['noise'])[1].detach().numpy() 
    sd_activity = net(sd_batch['inputs'], noise=sd_batch['noise'])[1].detach().numpy()

    for t in range(n_timesteps):

        hd_pca, sd_pca = get_iso_subspaces_at_time(t, max_components=dim, activity=(hd_activity, sd_activity))[2:4]
        hd_t_subspace = hd_pca.components_.T
        sd_t_subspace = sd_pca.components_.T

        hd_vs_win = get_norms(hd_t_subspace, net.W_in.weight.detach().numpy())[0]
        hd_vs_wout = get_norms(hd_t_subspace, net.W_out.weight.detach().T.numpy())[0]
        hd_vs_sd = get_norms(hd_t_subspace, sd_t_subspace)[0]

        sd_vs_win = get_norms(sd_t_subspace, net.W_in.weight.detach().numpy())[0]
        sd_vs_wout = get_norms(sd_t_subspace, net.W_out.weight.detach().T.numpy())[0]
        sd_vs_hd = get_norms(sd_t_subspace, hd_t_subspace)[0]

        assert np.isclose(hd_vs_sd, sd_vs_hd), f"Spectral norm mismatch at timestep {t}: {hd_vs_sd} vs {sd_vs_hd}"

        hd_spectral_norms[t, 0] = hd_vs_win
        hd_spectral_norms[t, 1] = hd_vs_wout
        hd_spectral_norms[t, 2] = hd_vs_sd

        sd_spectral_norms[t, 0] = sd_vs_win
        sd_spectral_norms[t, 1] = sd_vs_wout
        sd_spectral_norms[t, 2] = sd_vs_hd

    return hd_spectral_norms, sd_spectral_norms


def similarity_plot_across_time():
    hd_dim_variance, sd_dim_variance = get_iso_subspace_variance_across_time()
    hd_self_spectral_norms, sd_self_spectral_norms = get_iso_subspace_self_similarity_across_time()
    hd_spectral_norms, sd_spectral_norms = get_iso_subspace_similarity_across_time()

    fig = plt.figure(figsize=(20, 25))
    gs = fig.add_gridspec(nrows=5, ncols=6, height_ratios=[1,0.05,1,0.05,1], wspace=0, hspace=0.5)

    ax1 = fig.add_subplot(gs[0,0:3])
    ax2 = fig.add_subplot(gs[0,3:6])

    dim_cmap = matplotlib.colormaps['Set3']

    x = np.arange(hd_iso_task.config.n_timesteps)
    for i in range(hd_dim_variance.shape[0]):
        ax1.plot(x, hd_dim_variance[i], label=f'{i+1} PC{"s" if i>0 else ""}', color=dim_cmap(i))
        ax1.annotate(f'{i+1}', xy=(hd_iso_task.config.n_timesteps, hd_dim_variance[i, -1]), color=dim_cmap(i)) 

        ax2.plot(x, sd_dim_variance[i], label=f'{i+1} PC{"s" if i>0 else ""}', color=dim_cmap(i))
        ax2.annotate(f'{i+1}', xy=(sd_iso_task.config.n_timesteps, sd_dim_variance[i, -1]), color=dim_cmap(i))

    ax1.vlines(x=hd_iso_task.config.init_duration, ymin=0, ymax=1, color='white', linestyle='--')
    ax2.vlines(x=sd_iso_task.config.init_duration, ymin=0, ymax=1, color='white', linestyle='--')

    ax1.set_xlabel('Timestep')
    ax2.set_xlabel('Timestep')
    ax1.set_xticks([])
    ax1.set_ylabel('Cumulative Explained Variance')
    ax2.set_yticks([])
    ax1.set_title('Cumulative Explained Variance of PCs over Time in HD-varying Task', fontsize=16)
    ax2.set_title('Cumulative Explained Variance of PCs over Time in SD-varying Task', fontsize=16)

    cax1 = fig.add_subplot(gs[1,2:4])
    solid_line = matplotlib.lines.Line2D([], [], color='white', linestyle='-', label='At each timestep, for given number of PCs')
    cax1.legend(handles=[solid_line], loc='upper center', ncol=2, bbox_to_anchor=(0.5, 4))
    cax1.set_axis_off()


    ax3 = fig.add_subplot(gs[2,0:3])
    ax4 = fig.add_subplot(gs[2,3:6])

    im1 = ax3.imshow(hd_self_spectral_norms, cmap='viridis', norm=matplotlib.colors.Normalize(vmin=0, vmax=1))
    ax3.hlines(y=hd_iso_task.config.init_duration, xmin=0, xmax=hd_iso_task.config.n_timesteps-1, color='white', linestyle='--')
    ax3.vlines(x=hd_iso_task.config.init_duration, ymin=0, ymax=hd_iso_task.config.n_timesteps-1, color='white', linestyle='--')

    ax3.set_title('Similarity of Top 2 PCs over Time in HD-varying Task', pad=20, fontsize=16)
    ax3.set_xlabel('Timestep')
    ax3.set_xticks([])
    ax3.set_ylabel('Timestep')
    ax3.set_yticks([])

    im2 = ax4.imshow(sd_self_spectral_norms, cmap='viridis', norm=matplotlib.colors.Normalize(vmin=0, vmax=1))
    ax4.hlines(y=sd_iso_task.config.init_duration, xmin=0, xmax=sd_iso_task.config.n_timesteps-1, color='white', linestyle='--')
    ax4.vlines(x=sd_iso_task.config.init_duration, ymin=0, ymax=sd_iso_task.config.n_timesteps-1, color='white', linestyle='--')

    ax4.set_title('Similarity of Top 2 PCs over Time in SD-varying Task', pad=20, fontsize=16)
    ax4.set_xlabel('Timestep')
    ax4.set_xticks([])
    ax4.set_ylabel('Timestep')
    ax4.set_yticks([])

    cax2 = fig.add_subplot(gs[3,1:5])
    cbar4 = fig.colorbar(im1, ax=[ax3,ax4], cax=cax2, orientation='horizontal')
    cbar4.set_label('Spectral Norm')


    comp_cmap = matplotlib.colormaps['Set3']

    ax5 = fig.add_subplot(gs[4,1:5])

    ax5.plot(x, hd_spectral_norms[:, 0], label='HD vs W_in', color=comp_cmap(0))
    ax5.plot(x, hd_spectral_norms[:, 1], label='HD vs W_out', color=comp_cmap(1))
    ax5.plot(x, hd_spectral_norms[:, 2], label='HD vs SD', color=comp_cmap(2))

    ax5.plot(x, sd_spectral_norms[:, 0], label='SD vs W_in', color=comp_cmap(3))
    ax5.plot(x, sd_spectral_norms[:, 1], label='SD vs W_out', color=comp_cmap(4))

    ax5.vlines(x=hd_iso_task.config.init_duration, ymin=0, ymax=1, color='white', linestyle='--')

    ax5.set_xlabel('Timestep')
    ax5.set_ylabel('Spectral Norm')
    ax5.set_title('Similarity of Subspaces Over Time', fontsize=16)
    ax5.legend(ncol=5, loc='lower right')

    fig.suptitle('Variation of Principal Components of HD- and SD-Varying Activity over Time', fontsize=20)
    fig.subplots_adjust(top=0.9)

    return fig



figs = {
    'dimension_t=0': dimension_plot_at_time(0),
    'dimension_t=10': dimension_plot_at_time(10),
    'dimension_t=100': dimension_plot_at_time(100),
    'similarity_t=0': similarity_plot_at_time(0),
    'similarity_t=10': similarity_plot_at_time(10),
    'similarity_t=100': similarity_plot_at_time(100),
    'similarity_all_t': similarity_plot_across_time()
}

for name, fig in figs.items():
    checkpoint_dir = '/'.join(checkpoint_path.split('/')[:-1])
    fig.savefig(f'{checkpoint_dir}/{name}.png')
    plt.close(fig)