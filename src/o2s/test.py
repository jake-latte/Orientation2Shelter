import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams["figure.facecolor"] = 'black'
matplotlib.rcParams["axes.facecolor"] = 'black'
matplotlib.rcParams["savefig.facecolor"] = 'black'
matplotlib.rcParams["text.color"] = 'white'
matplotlib.rcParams["axes.labelcolor"] = 'white'
matplotlib.rcParams["xtick.color"] = 'white'
matplotlib.rcParams["ytick.color"] = 'white'

# plt.rcParams['text.usetex'] = True

from typing import List, Dict, Tuple, Callable

from PIL import Image
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, MDS
import os

import o2s

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import umap
from scipy.optimize import curve_fit


import psutil
def print_memory_usage():
    """Prints current RAM (CPU memory) and VRAM (GPU memory) usage."""
    
    # Get RAM usage
    ram_used = psutil.virtual_memory().used / 1024**2  # Convert to MB
    ram_total = psutil.virtual_memory().total / 1024**2
    print(f"\n🖥️  RAM Usage: {ram_used:.2f} MB / {ram_total:.2f} MB ({psutil.virtual_memory().percent}%)")

    # Get VRAM usage for each GPU
    for i in range(torch.cuda.device_count()):
        vram_allocated = torch.cuda.memory_allocated(i) / 1024**2  # Used VRAM
        vram_reserved = torch.cuda.memory_reserved(i) / 1024**2  # Total reserved VRAM
        vram_total = torch.cuda.get_device_properties(i).total_memory / 1024**2  # GPU memory capacity
        
        print(f"🎮 GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"   - VRAM Allocated: {vram_allocated:.2f} MB")
        print(f"   - VRAM Reserved: {vram_reserved:.2f} MB")
        print(f"   - VRAM Total: {vram_total:.2f} MB")



############################################################################################################################################
################################################################## TESTING #################################################################
############################################################################################################################################

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# Testing functions for appraisal of tuning and geometry and performance of trained o2s.net.RNNs                                                   #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


def test_gamut(task=None, net=None, batch=None,
               checkpoint_path=None, 
               subtask_batch_size=400, subtask_n_timesteps=510,
               include_umap=True, umap_select_t=[0, 9, 249, 499],
               include_dimensionality=True, dimensionality_prop_explained=0.8, dimensionality_var_explained=0.8,
               include_metric=True, metric_select_tau=[0, 0.01, 1, 10, 25, 49.9], metric_n_samples=25, metric_alpha=0.01, metric_d_theta=1e-6, metric_order=3, metric_dtype=torch.float64, metric_n_instantiations=30, metric_norm_dphi=True,
               include_trajectories=True, trajectories_select_t=[(10,0,11), (20,0,20), (250,10,499)],
               include_stability=True, stability_n_timesteps=500, stability_total_n_timesteps=5000, stability_slow_mult=10,
               include_lesions=True, lesions_n_lesions=100,
               include_tuning=True, tuning_batch_size=400, tuning_vars_list=['HD', 'AV', 'ego_SD', 'allo_SD'],
               include_fourier=True,
               include_eigenspectra=True, eigenspectra_eval_t=250,
               save=True, savedir='figures', return_figures=False):
    assert int(subtask_batch_size**0.5)**2 == subtask_batch_size, "joint_subtask_batchsize must be a square number"
    assert max(umap_select_t) < subtask_n_timesteps, "umap_select_t must be less than joint_subtask_n_timesteps"
    assert stability_n_timesteps <= stability_total_n_timesteps, "stability_n_timesteps must be less than or equal to stability_total_n_timesteps"

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    task = o2s.task.Task.from_checkpoint(checkpoint=checkpoint)
    device = task.config.device
    if not isinstance(device, torch.device):
        device = torch.device(device)
    alpha = task.config.dt / task.config.tau
    task.config.update(device=device, repeat_input=None, tau=1, dt=alpha, rank=0 if task.config.rank is None else int(task.config.rank))
    torch.set_default_dtype(torch.float64 if task.config.precise else torch.float32)
    net = o2s.net.RNN(task)
    # net.load_state_dict(checkpoint['net_state_dict'])
    for key in checkpoint['net_state_dict']:
        net.state_dict()[key].copy_(checkpoint['net_state_dict'][key])
    assert net.W_rec.weight.dtype == (torch.float64 if task.config.precise else torch.float32)
    net.state_noise_std = net.rate_noise_std = net.output_noise_std = 0

    assert 'init_duration' in task.config.dict, "task must have an 'init_duration' parameter"
    if include_umap or include_stability or include_eigenspectra:
        assert 'joint' in task.get_subtask_vars_funcs
    if include_trajectories:
        assert 'joint' in task.get_subtask_vars_funcs and 'av' in task.get_subtask_vars_funcs
    if include_metric:
        assert 'metric' in task.get_subtask_vars_funcs

    def get_subtask_batch(name):
        subtask = task.get_subtask(name,
                                   batch_size=subtask_batch_size, n_timesteps=subtask_n_timesteps,
                                   state_noise_std=0, rate_noise_std=0, output_noise_std=0, device='cpu')
        subtask_batch_ = o2s.data.TaskDataset(subtask, include_noise=False).get_batch()
        if (subtask_batch_['inputs'][:,task.config.init_duration:]==0).all():
            subtask.config.update(repeat_input=(subtask_n_timesteps, task.config.init_duration))
            subtask_batch = {'inputs': subtask_batch_['inputs'][:,0].cpu(), 
                             'targets': subtask_batch_['targets'][:,0].cpu(),
                             'vars': {k: v[:,0].cpu() for k,v in subtask_batch_['vars'].items()}}
        else:
            subtask_batch = {'inputs': subtask_batch_['inputs'].cpu(), 
                             'targets': subtask_batch_['targets'].cpu(),
                             'vars': {k: v.cpu() for k,v in subtask_batch_['vars'].items()}}
        del subtask_batch_
        return subtask, subtask_batch
    
    subtasks = {name: get_subtask_batch(name) for name in task.get_subtask_vars_funcs.keys() if 'metric' not in name}
    
    batch_ = o2s.data.TaskDataset(task, include_noise=False).get_batch()
    batch = {'inputs': batch_['inputs'].cpu(), 'vars': {k: v.cpu() for k,v in batch_['vars'].items()}}
    del batch_
    subtasks['full'] = (task, batch)

    joint_task, joint_batch = subtasks['joint']
    joint_inputs = joint_batch['inputs'].to(device)
    joint_activity = net(joint_inputs, repeat_input=joint_task.config.repeat_input, offload=True)[0].detach().cpu()

    figures = {} if return_figures else None
    if save:
        checkpoint_dir = os.path.dirname(checkpoint_path)
        if not os.path.exists(os.path.join(checkpoint_dir, savedir)):
            os.makedirs(os.path.join(checkpoint_dir, savedir))

    if include_umap:
        try:
            fig = umap_project(task, umap_select_t, joint_activity, joint_batch['vars'], verbose=False)
            print('UMAP plot created')

            if save:
                fig.savefig(os.path.join(checkpoint_dir, savedir, f'umap.png'))
                plt.close(fig)
            if return_figures:
                figures['umap'] = fig
        except Exception as e:
            print(f' * Error creating UMAP plot: {e}')

        print_memory_usage()

    dimensionality_results = None
    if include_dimensionality:
        try:
            dimensionality_results = test_task_dimensionality(task, net, subtasks, prop_explained=dimensionality_prop_explained, var_explained=dimensionality_var_explained)
            fig = plot_task_dimensionality(task, dimensionality_results)
            print('Dimensionality plot created')

            if save:
                fig.savefig(os.path.join(checkpoint_dir, savedir, f'dimensionality.png'))
                plt.close(fig)
            if return_figures:
                figures['dimensionality'] = fig
        except Exception as e:
            print(f' * Error calculating dimensionality: {e}')

        print_memory_usage()

    if include_metric:
        try:
            t0 = int((task.config.init_duration*task.config.dt)/metric_alpha)
            metric_results = get_metrics(task, net, 
                                         select_tau=metric_select_tau, n_samples=metric_n_samples, alpha=metric_alpha, d_theta=metric_d_theta, order=metric_order, dtype=metric_dtype, n_input_times=t0, n_noise_instantiations=metric_n_instantiations, norm_dphi=metric_norm_dphi)
            fig = plot_metrics(metric_results)
            print('Metrics calculated')

            if save:
                fig.savefig(os.path.join(checkpoint_dir, savedir, f'metric.png'))
                print(f'Saved metric plot to {os.path.join(checkpoint_dir, savedir, f"metric.png")}')
                plt.close(fig)
            if return_figures:
                figures['metric'] = fig
        except Exception as e:
            print(f' * Error calculating metrics: {e}')

        print_memory_usage()

    if include_trajectories:
        try:
            fig = plot_joint_trajectories(task, net, joint_batch, joint_activity, T=trajectories_select_t)
            print('Joint trajectories plotted')

            if save:
                fig.savefig(os.path.join(checkpoint_dir, savedir, f'joint_trajectories.png'))
                plt.close(fig)
            if return_figures:
                figures['joint_trajectories'] = fig
        except Exception as e:
            print(f' * Error plotting joint trajectories: {e}')

        print_memory_usage()

    if include_trajectories:
        try:
            av_batch = subtasks['av'][1]
            av_inputs = av_batch['inputs'].to(device)
            av_activity = net(av_inputs, offload=True)[0].detach().cpu()
            av_dim = 4 if dimensionality_results is None else dimensionality_results['dim'][list(task.get_subtask_vars_funcs.keys()).index('av')] 
            fig = plot_av_trajectories(task, net, av_batch, av_activity, dim=av_dim)
            print('AV trajectories plotted')

            if save:
                fig.savefig(os.path.join(checkpoint_dir, savedir, f'av_trajectories.png'))
                plt.close(fig)
            if return_figures:
                figures['av_trajectories'] = fig
        except Exception as e:
            print(f' * Error plotting angular velocity trajectories: {e}')

        print_memory_usage()

    if include_stability:
        try:
            fig = plot_stability(task, net, subtask_batch_size, total_n_timesteps=stability_total_n_timesteps, n_timesteps=stability_n_timesteps, slow_mult=stability_slow_mult)
            print('Stability plotted')

            if save:
                fig.savefig(os.path.join(checkpoint_dir, savedir, f'stability.png'))
                plt.close(fig)
            if return_figures:
                figures['stability'] = fig
        except Exception as e:
            print(f' * Error plotting stability: {e}')

        print_memory_usage()

    if include_lesions:
        try:
            fig = plot_lesions(task, net, n_lesions=lesions_n_lesions)
            print('Lesions plotted')

            if save:
                fig.savefig(os.path.join(checkpoint_dir, savedir, f'lesions.png'))
                plt.close(fig)
            if return_figures:
                figures['lesions'] = fig
        except Exception as e:
            print(f' * Error plotting lesions: {e}')

        print_memory_usage()

    tuning_vars, tuning_dict = None, None
    if include_tuning:
        try:
            tuning_task = o2s.task.Task.from_checkpoint(checkpoint)
            tuning_task.config.update(batch_size=tuning_batch_size, device=device, rank=0 if tuning_task.config.rank is None else int(tuning_task.config.rank))
            tuning_net = o2s.net.RNN(tuning_task)
            tuning_net.load_state_dict(checkpoint['net_state_dict'])
            tuning_batch = o2s.data.TaskDataset(tuning_task, include_noise=False).get_batch()
            with torch.no_grad():
                tuning_inputs = tuning_batch['inputs'].to(device)
                activity = tuning_net(tuning_inputs, offload=True)[1].detach().cpu().numpy()
            inputs, targets, vars = tuning_batch['inputs'].cpu().numpy(), tuning_batch['targets'].cpu().numpy(), {k: v.cpu().numpy() for k,v in tuning_batch['vars'].items()}
            tuning_vars, tuning_dict = get_tuning_generalised(tuning_task, inputs, targets, vars, activity, tuning_vars_list)
            del inputs, targets, vars, activity
            figs = test_tuning(tuning_task, tuning_net, tuning_batch, tuning_vars_list, tuning_vars=tuning_vars, tuning_dict=tuning_dict)
            print('Tuning plots created')

            if save:
                for name, fig in figs.items():
                    fig.savefig(os.path.join(checkpoint_dir, savedir, f'{name}.png'))
                    plt.close(fig)
            if return_figures:
                figures.update(figs)

            del tuning_batch, tuning_net
        except Exception as e:
            print(f' * Error loading tuning task: {e}')

        print_memory_usage()

    if include_tuning:
        try:
            coefs = fit_tuning_curves(tuning_vars, tuning_dict)
            fig1 = plot_tuning_dist(tuning_vars, coefs)
            fig2 = plot_tuned_weights(task, net, coefs)
            print('Tuning curves fitted')

            if save:
                fig1.savefig(os.path.join(checkpoint_dir, savedir, f'tuning_dist.png'))
                fig2.savefig(os.path.join(checkpoint_dir, savedir, f'tuned_weights.png'))
                plt.close(fig1)
                plt.close(fig2)
            if return_figures:
                figures['tuning_dist'] = fig1
                figures['tuned_weights'] = fig2
        except Exception as e:
            print(f' * Error fitting tuning curves: {e}')

        print_memory_usage()

    if include_fourier:
        try:
            fig = plot_fourier_weights(task, net)
            print('Fourier weights plotted')

            if save:
                fig.savefig(os.path.join(checkpoint_dir, savedir, f'fourier_weights.png'))
                plt.close(fig)
            if return_figures:
                figures['fourier_weights'] = fig
        except Exception as e:
            print(f' * Error plotting Fourier weights: {e}')

        print_memory_usage()

    if include_eigenspectra:
        try:
            fig = plot_joint_eigenspectra(task, net, joint_task, joint_batch, eval_t=eigenspectra_eval_t)
            print('Joint eigenspectra plotted')

            if save:
                fig.savefig(os.path.join(checkpoint_dir, savedir, f'joint_eigenspectra.png'))
                plt.close(fig)
            if return_figures:
                figures['joint_eigenspectra'] = fig
        except Exception as e:
            print(f' * Error plotting joint eigenspectra: {e}')

        print_memory_usage()
    
    return figures




def umap_project(task, project_timesteps: List[int], activity: torch.Tensor, vars: Dict[str, torch.Tensor], 
                 n_neighbors: int = 25, n_components: int = 3, random_state: int = 0, verbose: bool = True) -> List[np.ndarray]:
    assert task.config.tau == 1
    assert vars['hd'].shape[0] == activity.shape[0]
    assert vars['hd'].shape == vars['sd'].shape
    if len(project_timesteps) < activity.shape[1]:
        activity = activity[:,project_timesteps]

    umap_activity = []
    for i, t in enumerate(project_timesteps):
        umapper = Pipeline([
            ("scaler", StandardScaler()),
            ("umap_reducer", umap.UMAP(
                n_components=n_components, 
                n_neighbors=n_neighbors,
                random_state=random_state,
                verbose=verbose
            ))
        ])
        umap_activity.append(
            umapper.fit_transform(activity[:,i].numpy().reshape((-1, task.config.n_neurons)))
        )

    fig = plt.figure(figsize=(min(5*len(project_timesteps),30), 15))
    gs = fig.add_gridspec(3, len(project_timesteps)+1, hspace=0.1, wspace=0.1, width_ratios=[1]*len(project_timesteps)+[0.1])

    hd = vars['hd']
    sd = vars['sd']
    ad = torch.remainder(hd + sd, 2*np.pi)
    if len(hd.shape)>1:
        hd, sd, ad = hd[:,0], sd[:,0], ad[:,0]

    hd_cmap = plt.get_cmap('Reds')
    sd_cmap = plt.get_cmap('Blues')
    ad_cmap = plt.get_cmap('Greens')
    norm = matplotlib.colors.Normalize(vmin=0, vmax=2*np.pi)

    for i, t in enumerate(project_timesteps):
        for j, (var, cmap, title) in enumerate(zip([hd, sd, ad], [hd_cmap, sd_cmap, ad_cmap], ['Head Direction', 'Head-Shelter Angle', 'Absolute Shelter Angle'])):
            ax = fig.add_subplot(gs[j, i], projection='3d')
            ax.scatter(umap_activity[i][:,0], umap_activity[i][:,1], umap_activity[i][:,2], c=var, cmap=cmap, norm=norm, s=10)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])

            if i==0 and j==0:
                ax.set_xlabel('UMAP 1', labelpad=-10)
                ax.set_ylabel('UMAP 2', labelpad=-10)
                ax.set_zlabel('UMAP 3', labelpad=-10)
            if i==len(project_timesteps)-1:
                cax = fig.add_subplot(gs[j, len(project_timesteps)])
                cbar = matplotlib.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm, orientation='vertical')
                cbar.set_label(title, fontsize=16, fontweight='bold')
            if j==0:
                ax.set_title(fr'${(t+1)*(task.config.dt/task.config.tau):.2f}\tau$')

    fig.suptitle('UMAP of o2s.net.RNN States at Different Timesteps (no angular velocity)', fontsize=22, fontweight='bold')

    return fig



def get_norms(M_A, M_B):
    # 1. Perform QR decomposition on the input matrices
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

def get_task_subspace_at_time(task, net, eval_t, max_components=4, activity=None):

    if activity is None:
        inputs = o2s.data.TaskDataset(task, include_noise=False).get_batch()['inputs']
        if task.config.repeat_input is not None:
            inputs = inputs[:,0]
        activity = net(inputs, select_t=[eval_t], repeat_input=task.config.repeat_input, offload=True)[0].cpu().detach().numpy()
    else:
        if len(activity.shape) > 2:
            activity = activity[:,eval_t,:]
        inputs = None

    pca = PCA(n_components=max_components)
    pca_activity = pca.fit_transform(activity.reshape((-1,net.n_neurons)))

    return inputs, pca, pca_activity

def get_task_subspace_variance_across_time(task, net, max_dim=10, activity=None):
    n_timesteps = task.config.n_timesteps if task.config.repeat_input is None else task.config.repeat_input[0]

    subspaces = np.full((n_timesteps, max_dim, net.n_neurons), fill_value=np.nan)
    dim_variance = np.full((max_dim, n_timesteps), fill_value=np.nan)

    if activity is None:
        inputs = o2s.data.TaskDataset(task, include_noise=False).get_batch()['inputs']
        if task.config.repeat_input is not None:
            inputs = inputs[:,0]
        activity = net(inputs, repeat_input=task.config.repeat_input, offload=True)[0].cpu().detach().numpy()

    for t in range(n_timesteps):
        pca = get_task_subspace_at_time(task, net, t, max_components=max_dim, activity=activity)[1]

        subspaces[t] = pca.components_
        dim_variance[:,t] = np.cumsum(pca.explained_variance_ratio_)

    return subspaces, dim_variance


def get_task_subspace_self_similarity_across_time(task, net, dim=4, activity=None, subspaces=None):
    n_timesteps = task.config.n_timesteps if task.config.repeat_input is None else task.config.repeat_input[0]

    spectral_norms = np.full((n_timesteps, n_timesteps), fill_value=np.nan)

    if subspaces is not None:
        subspaces = subspaces[:, :dim]
        assert activity is None

    if activity is None:
        inputs = o2s.data.TaskDataset(task, include_noise=False).get_batch()['inputs']
        if task.config.repeat_input is not None:
            inputs = inputs[:,0]
        activity = net(inputs, repeat_input=task.config.repeat_input, offload=True)[0].cpu().detach().numpy()
    else:
        assert subspaces is None

    for i in range(n_timesteps):
        if subspaces is None:
            pca_i = get_task_subspace_at_time(task, net, i, max_components=dim, activity=activity)[1]
            subspace_i = pca_i.components_.T
        else:
            subspace_i = subspaces[i].T

        for j in range(i+1):
            if subspaces is None:
                pca_j = get_task_subspace_at_time(task, net, j, max_components=dim, activity=activity)[1]
                subspace_j = pca_j.components_.T
            else:
                subspace_j = subspaces[j].T

            spectral_norm = get_norms(subspace_i, subspace_j)[0]

            spectral_norms[i, j] = spectral_norm

    return spectral_norms

def get_task_subspace_similarity_across_time(task, net, comp_subspaces, dim=4, activity=None, subspaces=None):
    n_timesteps = task.config.n_timesteps if task.config.repeat_input is None else task.config.repeat_input[0]

    for i, comp_subspace in enumerate(comp_subspaces):
        assert (len(comp_subspace.shape)==2 and (comp_subspace.shape[1]==net.n_neurons)) or (len(comp_subspace.shape)==3 and (comp_subspace.shape[2]==net.n_neurons))

    if subspaces is not None:
        subspaces = subspaces[:,:dim]
        assert activity is None

    if activity is None:
        inputs = o2s.data.TaskDataset(task, include_noise=False).get_batch()['inputs']
        if task.config.repeat_input is not None:
            inputs = inputs[:,0]
        activity = net(inputs, repeat_input=task.config.repeat_input, offload=True)[0].cpu().detach().numpy()
    else:
        assert subspaces is None

    spectral_norms = np.full((n_timesteps, len(comp_subspaces)), fill_value=np.nan)

    for t in range(n_timesteps):

        if subspaces is None:
            task_subspace = get_task_subspace_at_time(task, net, t, max_components=dim, activity=activity)[1].components_.T
        else:
            task_subspace = subspaces[t].T

        for i, comp_subspace in enumerate(comp_subspaces):

            if len(comp_subspace.shape)==3:
                if comp_subspace.shape[0] <= t: 
                    continue
                
                comp_subspace = comp_subspace[t]
            
            spectral_norms[t, i] = get_norms(task_subspace, comp_subspace.T)[0]
            

    return spectral_norms





def min_dim_explaining_variance(dim_variance: torch.Tensor, var_explained: float = 0.8, prop_explained: float = 0.8) -> int:
    for d in range(dim_variance.shape[0]):
        if (dim_variance[d,:] >= var_explained).mean() >= prop_explained:
            return d+1
    return dim_variance.shape[0]

def test_task_dimensionality(task, net, subtasks: Dict[str, Tuple[o2s.task.Task, Dict[str, torch.Tensor]]], 
                             prop_explained: float=0.8, var_explained: float=0.8, force_dims: List[int]=None) -> Dict[str, Tuple[int, int]]:
    if force_dims is not None:
        assert len(force_dims) == len(subtasks), "force_dims must have the same length as subtasks"

    results = {
        'name': [],
        'dim': [],
        'dim_variance': [],
        'self_spectral_norms': [],
        'comp_spectral_norms': [],
        'comp_subspaces_names': [],
        'comp_subspaces': []
    }

    with torch.no_grad():

        comp_subspaces_names = ['W_in', 'W_out']
        comp_subspaces = [net.W_in.weight.detach().cpu().numpy().T, net.W_out.weight.cpu().detach().numpy()]
        if task.config.intermediate_output_dim > 0:
            comp_subspaces[1] = net.W_out_inter.weight.cpu().detach().numpy()

        for subtask_name, (subtask, subtask_batch) in subtasks.items():

            print(f'Calculating subspace variance for {subtask_name} subtask')
            subtask_activity = net(subtask_batch['inputs'], repeat_input=subtask.config.repeat_input, offload=True)[0].cpu().detach().numpy()
            subtask_subspaces, subtask_dim_variance = get_task_subspace_variance_across_time(subtask, net, activity=subtask_activity)
            del subtask_activity
            if force_dims is not None:
                subtask_dim = force_dims.pop(0)
            else:
                subtask_dim = min_dim_explaining_variance(subtask_dim_variance, var_explained=var_explained, prop_explained=prop_explained)
            print(f'{subtask_name} subtask requires {subtask_dim} to explain {var_explained*100:.0f}% of variance {prop_explained*100:.0f}% of the time')
            print(f'Calculating subspace self-similarity for {subtask_name} subtask')
            subtask_self_spectral_norms = get_task_subspace_self_similarity_across_time(subtask, net, dim=subtask_dim, subspaces=subtask_subspaces)

            comp_subspaces_names.append(subtask_name)
            comp_subspaces.append(subtask_subspaces[:,:subtask_dim])

            results['name'].append(subtask_name)
            results['dim'].append(subtask_dim)
            results['dim_variance'].append(subtask_dim_variance)
            results['self_spectral_norms'].append(subtask_self_spectral_norms)
        
        results['comp_subspaces_names'] = comp_subspaces_names
        results['comp_subspaces'] = comp_subspaces

        for i, (subtask_name, (subtask, subtask_batch)) in enumerate(subtasks.items()):
            subtask_subspace = comp_subspaces[i+2]
            subtask_dim = subtask_subspace.shape[1]

            print(f'Calculating subspace similarity for {subtask_name} subtask')
            subtask_comp_spectral_norms = get_task_subspace_similarity_across_time(subtask, net, comp_subspaces, dim=subtask_dim, subspaces=subtask_subspace)

            results['comp_spectral_norms'].append(subtask_comp_spectral_norms)

    return results

def plot_task_dimensionality(task, dimensionality_results, legend_col_width=0.5, inset_self_sim=100, figwidth=45, figheight=15):
    n_timesteps = dimensionality_results['dim_variance'][0].shape[1]
    inset_self_sim = min(inset_self_sim,n_timesteps)

    names, dims, dim_variances, self_spectral_norms = dimensionality_results['name'], dimensionality_results['dim'], dimensionality_results['dim_variance'], dimensionality_results['self_spectral_norms']
    comp_subspaces_names, comp_spectral_norms = dimensionality_results['comp_subspaces_names'], dimensionality_results['comp_spectral_norms']

    if 'title' in dimensionality_results:
        titles = dimensionality_results['title']
    else:
        titles = names

    fig = plt.figure(figsize=(figwidth, figheight))
    gs = fig.add_gridspec(3, len(names)+1, width_ratios=[1]*len(names)+[legend_col_width], hspace=0.3, wspace=0.1)

    dim_cmap = plt.get_cmap('Dark2')
    first_ax = None
    sharex = []
    for i, (title, max_dim, dim_variance) in enumerate(zip(titles, dims, dim_variances)):
        ax = fig.add_subplot(gs[0, i], sharey=first_ax)
        if first_ax is None:
            first_ax = ax
        for dim in range(max_dim+1):
            ax.plot(dim_variance[dim-1], label=f'Dim {dim}', color=dim_cmap(dim), linestyle='--' if dim==max_dim else '-', linewidth=3, alpha=0.7)
        ax.axvline(x=task.config.init_duration, color='black', linestyle='--')
        ax.annotate(f'{title}', xy=(0.5, 1.5), xycoords='axes fraction', fontsize=24, fontweight='bold', ha='center')
        if i==0:
            ax.set_ylabel('Explained Variance', fontsize=20)
        ax.set_ylim([0,1.1])
        sharex.append(ax)

        if i==len(titles)//2:
            ax.set_title('Subspace Dimensionality', fontsize=24, pad=20, fontweight='bold')

    cax1 = fig.add_subplot(gs[0, -1])
    handles = [matplotlib.lines.Line2D([0], [0], color=dim_cmap(dim), label=f'Dim {dim+1}') for dim in range(max(dims)+1)]
    cax1.legend(handles=handles, title='Dimension', ncol=1, loc='center', fontsize=20, title_fontsize=24)
    cax1.axis('off')

    sim_cmap = plt.get_cmap('Spectral')
    first_ax = None
    for i, (title, self_similarity) in enumerate(zip(titles, self_spectral_norms)):
        ax = fig.add_subplot(gs[1, i], sharey=first_ax, sharex=sharex[i])
        if first_ax is None:
            first_ax = ax
        im = ax.imshow(self_similarity[:inset_self_sim][:,:inset_self_sim], cmap=sim_cmap)
        if inset_self_sim > task.config.init_duration:
            ax.axvline(x=task.config.init_duration, color='black', linestyle='--')
            ax.hlines(y=task.config.init_duration, xmin=0, xmax=task.config.init_duration, color='black', linestyle='--')
        if i==0:
            ax.set_ylabel('Time', fontsize=16)
            ax.set_yticks([0, task.config.init_duration] + [t for t in range(inset_self_sim, n_timesteps, 100)])
            ax.set_yticklabels([fr'${t*task.config.dt:.0f}\tau$' for t in [0, task.config.init_duration]] + [fr'${t*task.config.dt:.0f}\tau$' for t in range(inset_self_sim, n_timesteps, 100)])

        if i==len(titles)//2:
            ax.set_title('Subspace self-similarity across time', fontsize=24, pad=10, fontweight='bold')

    cax2 = fig.add_subplot(gs[1, -1])
    cax2.set_xlim([0,0.5])
    cbar = matplotlib.colorbar.ColorbarBase(cax2, cmap=sim_cmap, orientation='vertical')
    cbar.set_label('Spectral Norm', fontsize=20, labelpad=20)

    comp_cmap = plt.get_cmap('Dark2')
    first_ax = None
    for i, (title, comp_similarity) in enumerate(zip(titles, comp_spectral_norms)):
        ax = fig.add_subplot(gs[2, i], sharey=first_ax, sharex=sharex[i])
        if first_ax is None:
            first_ax = ax

        for j, comp_subspace_name in enumerate(comp_subspaces_names):
            if title==comp_subspace_name:
                continue
            ax.plot(comp_similarity[:,j], label=f'{comp_subspace_name}', color=comp_cmap(j), linewidth=3, alpha=0.7)
        ax.axvline(x=task.config.init_duration, color='black', linestyle='--')
        ax.set_xlabel('Time', fontsize=20)
        if i == 0:
            ax.set_ylabel('Spectral Norm', fontsize=20)
        ax.set_xticks([0, task.config.init_duration] + [t for t in range(inset_self_sim, n_timesteps, 100)])
        ax.set_xticklabels([fr'${t*task.config.dt:.0f}\tau$' for t in [0, task.config.init_duration]] + [fr'${t*task.config.dt:.0f}\tau$' for t in range(inset_self_sim, n_timesteps, 100)])
        ax.set_ylim([0,1.1])

        if i==len(titles)//2:
            ax.set_title('Subspace similarity to other subspaces', fontsize=24, pad=20, fontweight='bold')
    cax3 = fig.add_subplot(gs[2, -1])
    handles = [matplotlib.lines.Line2D([0], [0], color=comp_cmap(j), label=f'{comp_subspaces_names[j]}') for j in range(len(comp_subspaces_names))]
    cax3.legend(handles=handles, title='Comparison Subspace', ncol=1, loc='center', fontsize=20, title_fontsize=24)
    cax3.axis('off')

    return fig


import math

def coeff(n):
    P = np.linspace(-n, n, 2*n+1, dtype=np.int32)
    C = np.full((len(P),), np.nan)
    for i, p in enumerate(P):
        if p==0:
            continue
        c = ((-1)**(np.abs(p)+1) * math.factorial(n)**2) / (p * math.factorial(n-p) * math.factorial(n+p))
        C[i] = c

    return P, C

"""
dphi_dtheta
Compute the derivative of the o2s.net.RNN state over time with respect to the parameters theta_1 and theta_2

Parameters:
- theta_1_vals: torch.tensor, values of the first parameter theta_1 (1D tensor)
- theta_2_vals: torch.tensor, values of the second parameter theta_2 (1D tensor)
- net: Net, continuous-time o2s.net.RNN model to compute the state
- input_func: Callable[[torch.tensor, torch.tensor], torch.tensor], function to construct the input to the o2s.net.RNN given theta_1 and theta_2
- d_theta: float, small perturbation to theta_1/theta_1 for finite difference derivative
- n_timesteps, alpha, noise_std, n_input_times: parameters for the o2s.net.RNN model forward pass
- alpha: float, discrete time step size, alpha = tau/dt
- noise_std: float, standard deviation of Gaussian noise added to states at each timestep
- n_input_times: int, number of timesteps to present the input (otherwise input is zero)
- select_t: list of int, timesteps to keep in the output tensor (default is all)
- dtype: (pytorch) datatype to use for tensors

"""
def get_dphi_dtheta(n_samples: int, net: o2s.net.RNN, metric_task: o2s.task.Task, 
                    d_theta=1e-6, order=1, select_t=None, n_timesteps=500, n_input_times=10, dtype: torch.dtype = torch.float64, n_noise_instantiations=10):

    og_dtpye = net.W_rec.weight.dtype
    net = net.to(dtype)

    if select_t is None:
        select_t = list(range(n_timesteps))

    use_noise=True
    if n_noise_instantiations<1:
        n_noise_instantiations = 1
        use_noise = False

    dphi_dtheta_1 = torch.zeros((n_noise_instantiations, n_samples, n_samples, len(select_t), net.n_neurons), dtype=dtype)
    dphi_dtheta_2 = torch.zeros((n_noise_instantiations, n_samples, n_samples, len(select_t), net.n_neurons), dtype=dtype)
    P, C = coeff(order)
    all_dtheta_1_inputs = []
    all_dtheta_2_inputs = []
    for p, c in zip(P, C):
        if np.isnan(c):
            all_dtheta_1_inputs.append(None)
            all_dtheta_2_inputs.append(None)
            continue
        
        metric_task.config.update(dtheta_1=p*d_theta, dtheta_2=0)
        dtheta_1_inputs = o2s.data.TaskDataset(metric_task, include_noise=False).get_batch()['inputs'][:,0].to(dtype)
        all_dtheta_1_inputs.append(dtheta_1_inputs)

        metric_task.config.update(dtheta_1=0, dtheta_2=p*d_theta)
        dtheta_2_inputs = o2s.data.TaskDataset(metric_task, include_noise=False).get_batch()['inputs'][:,0].to(dtype)
        all_dtheta_2_inputs.append(dtheta_2_inputs)

    for i in range(n_noise_instantiations):

        if use_noise:
            trial_noise = torch.normal(mean=0, std=metric_task.config.state_noise_std, size=(n_timesteps, net.n_neurons), dtype=dtype)
            noise = trial_noise.repeat(n_samples**2, 1, 1)
        else:
            noise = None

        for j, (p, c) in enumerate(zip(P, C)):
            if np.isnan(c):
                continue

            dtheta_1_inputs = all_dtheta_1_inputs[j]
            phi_dtheta_1_theta_2 = net(dtheta_1_inputs, noise=(noise, None, None),
                                    repeat_input=(n_timesteps, n_input_times), select_t=select_t, offload=True)[0].cpu().detach().numpy()           
            phi_dtheta_1_theta_2 = phi_dtheta_1_theta_2.reshape(n_samples, n_samples, len(select_t), -1) 
            dphi_dtheta_1[i] += (c/d_theta) * phi_dtheta_1_theta_2

            dtheta_2_inputs = all_dtheta_2_inputs[j]
            phi_theta_1_dtheta_2 = net(dtheta_2_inputs, noise=(noise, None, None),
                                    repeat_input=(n_timesteps, n_input_times), select_t=select_t, offload=True)[0].cpu().detach().numpy() 
            phi_theta_1_dtheta_2 = phi_theta_1_dtheta_2.reshape(n_samples, n_samples, len(select_t), -1)  
            dphi_dtheta_2[i] += (c/d_theta) * phi_theta_1_dtheta_2

            print(f'Finished order {p}')

        print(f'Finished instantiation {i}')



    dphi_dtheta_1 = dphi_dtheta_1.permute(0, 3, 1, 2, 4).numpy()  
    dphi_dtheta_2 = dphi_dtheta_2.permute(0, 3, 1, 2, 4).numpy()   
    
    net.to(og_dtpye)                

    return dphi_dtheta_1, dphi_dtheta_2

"""
metric
Compute the metric tensor from the derivative of the o2s.net.RNN state over time with respect to the parameters theta_1 and theta_2

Parameters:
- dphi_dtheta_1: torch.tensor, derivative of the o2s.net.RNN state over time with respect to the first parameter theta_1
- dphi_dtheta_2: torch.tensor, derivative of the o2s.net.RNN state over time with respect to the second parameter theta_2
"""
def calculate_metric(dphi_dtheta_1, dphi_dtheta_2, take_mean=True, norm_dphi=True):
    assert dphi_dtheta_1.shape==dphi_dtheta_2.shape, "dphi_dtheta_1 and dphi_dtheta_2 must have the same shape"
    assert len(dphi_dtheta_1.shape)==5, "dphi_dtheta_1 and dphi_dtheta_2 must be 4D tensors (n_instantiations, n_timesteps, n_samples, n_samples, n_neurons)"
    n_instantiations, n_timesteps, n_samples, _, n_neurons = dphi_dtheta_1.shape

    if norm_dphi:
        dphi_dtheta_1 /= np.linalg.norm(dphi_dtheta_1, axis=4, keepdims=True)
        dphi_dtheta_2 /= np.linalg.norm(dphi_dtheta_2, axis=4, keepdims=True)

    tangent_basis = np.stack([dphi_dtheta_1, dphi_dtheta_2], axis=4)
    metric = np.matmul(tangent_basis, tangent_basis.transpose(0, 1, 2, 3, 5, 4)) 
    if take_mean:
        print(f'Taking mean with shape {metric.shape} | contains nan {np.isnan(metric).any()}')
        metric = np.mean(metric, axis=0)

        metric_expanded = np.full((n_timesteps, 2*n_samples, 2*n_samples), np.nan, dtype=tangent_basis.dtype) 
        
        metric_expanded[:, :n_samples, :n_samples] = metric[:, :, :, 0, 0]
        metric_expanded[:, n_samples:, :n_samples] = metric[:, :, :, 1, 0]
        metric_expanded[:, :n_samples, n_samples:] = metric[:, :, :, 0, 1]
        metric_expanded[:, n_samples:, n_samples:] = metric[:, :, :, 1, 1]

    else:
        metric_expanded = np.full((n_instantiations, n_timesteps, 2*n_samples, 2*n_samples), np.nan, dtype=tangent_basis.dtype)

        metric_expanded[:, :, :n_samples, :n_samples] = metric[:, :, :, :, 0, 0]
        metric_expanded[:, :, n_samples:, :n_samples] = metric[:, :, :, :, 1, 0]
        metric_expanded[:, :, :n_samples, n_samples:] = metric[:, :, :, :, 0, 1]
        metric_expanded[:, :, n_samples:, n_samples:] = metric[:, :, :, :, 1, 1]

    return metric_expanded

def get_metrics(task, net, select_tau=None, n_samples=25, alpha=0.01, dtype=torch.float64, d_theta=1e-6, order=3, n_input_times=100, n_noise_instantiations=100, take_mean=True, norm_dphi=True):
    assert net.tau == 1
    t_mult = 1 / alpha

    if select_tau is None:
        select_t = np.arange(task.config.n_timesteps * net.dt * t_mult).astype(int)
    else:
        select_t = (np.array(select_tau) * t_mult).astype(int)

    n_timesteps = max(select_t)+1
    net.dt = alpha
    net.to(dtype)

    hd_ad_task = task.get_subtask('metric')
    hd_ad_task.config.update(tau=1, dt=alpha, batch_size=n_samples**2, n_timesteps=1, init_duration=1,
                             theta_2_is_SD=False, dtheta_1=0, dtheta_2=0)

    hd_sd_task = task.get_subtask('metric')
    hd_sd_task.config.update(tau=1, dt=alpha, batch_size=n_samples**2, n_timesteps=1, init_duration=1,
                             theta_2_is_SD=True, dtheta_1=0, dtheta_2=0)

    metrics = []
    for i, metric_task in enumerate([hd_ad_task, hd_sd_task]):
        print(f'Computing metric for {["HD/AD", "HD/SD"][i]}')
        with torch.no_grad():
            dphi_dtheta_1, dphi_dtheta_2 = get_dphi_dtheta(n_samples, net, metric_task, 
                                                           n_timesteps=n_timesteps, d_theta=d_theta, select_t=select_t, order=order, n_input_times=n_input_times, dtype=dtype, n_noise_instantiations=n_noise_instantiations)
            metric = calculate_metric(dphi_dtheta_1, dphi_dtheta_2, take_mean=take_mean, norm_dphi=norm_dphi)
            metrics.append(metric)

    net.tau = task.config.tau
    net.dt = task.config.dt
    net.to(torch.float64 if task.config.precise else torch.float32)

    return {'HD-AD': metrics[0], 'HD-SD': metrics[1], 'params': {'select_t': select_t, 'd_theta': d_theta, 'order': order, 'n_input_times': n_input_times, 'alpha': alpha, 'dtype': dtype, 'n_noise_instantiations': n_noise_instantiations}}

def plot_metrics(metric_results, title=None):

    all_metrics = np.stack([metric_results['HD-AD'], metric_results['HD-SD']], axis=0)  
    select_t, d_theta, order, n_input_times, alpha, dtype, n_noise_instantiations = metric_results['params'].values()

    fig = plt.figure(figsize=(min(8*len(select_t), 30), min(5*all_metrics.shape[0], 30)))
    gs = fig.add_gridspec(len(all_metrics)+1, len(select_t), wspace=0.5, hspace=0.5, height_ratios=[1]*len(all_metrics)+[0.1])

    names = [r'$\theta_1=HD, \theta_2=ASA$', r'$\theta_1=HD, \theta_2=HSA$']
    for i, (metric,  name) in enumerate(zip(all_metrics, names)):
        for j, t in enumerate(select_t):
            vmax = np.max(np.abs(all_metrics[i,j]))
            if vmax < 1:
                norm = matplotlib.colors.Normalize(vmin=-vmax, vmax=vmax)
                ticks = [-vmax,0, vmax]
                labels = [f"{-vmax:.1E}", "0", f"{vmax:.1E}"]
            else:
                norm = matplotlib.colors.SymLogNorm(linthresh=1, vmin=-vmax, vmax=vmax)
                ticks = [-vmax, -1,0,1, vmax]
                labels = [f"{-vmax:.1E}", '-1','0','1', f"{vmax:.1E}"]

            metric_ = metric[j]

            ax = fig.add_subplot(gs[i, j])
            im = ax.imshow(metric_, cmap='seismic', norm=norm)
            ax.set_xticks([])
            ax.set_yticks([])

            if i==0:
                ax.set_title(fr"$t={(t+1)*alpha:.2f}\tau$", fontsize=20, fontweight='bold', pad=20)
            if j==0:
                ax.set_ylabel(fr"{name}", rotation=90, fontsize=22, labelpad=60, fontweight='bold')
                ax.annotate(r'$\frac{\partial x}{\partial \theta_1}$', xy=(0, 0.75), rotation=90, xycoords='axes fraction', fontsize=30, ha='right')
                ax.annotate(r'$\frac{\partial x}{\partial \theta_2}$', xy=(0, 0.25), rotation=90, xycoords='axes fraction', fontsize=30, ha='right')
                ax.annotate(r'$\frac{\partial x}{\partial \theta_1}$', xy=(0.25, 0), rotation=0, xycoords='axes fraction', fontsize=30, va='top', ha='center')
                ax.annotate(r'$\frac{\partial x}{\partial \theta_2}$', xy=(0.75, 0), rotation=0, xycoords='axes fraction', fontsize=30, va='top', ha='center')


            if i==len(all_metrics)-1:
                cax = fig.add_subplot(gs[i+1, j])
                cbar = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap='seismic'), cax=cax, orientation='horizontal')
                cax.set_xticks(ticks)
                cax.set_xticklabels(labels, rotation=45, ha='right', fontsize=16, fontweight='bold')

    if title is None:
        fig.suptitle(r"Mean metric" + f", using:\n" + fr"$\Delta t={alpha}\tau$ | $\Delta \theta_i = {d_theta}$ | Order {order} | Precision {dtype} | {n_noise_instantiations} instantiations ", fontsize=22, y=1.02)
    else:
        fig.suptitle(title, fontsize=22, fontweight='bold', y=1.02)

    return fig


def plot_joint_trajectories(task, net, joint_batch, joint_activity: torch.Tensor, T: List[Tuple[int, int, int]]):
    batch_size = joint_activity.shape[0]

    joint_batch['vars']['ad'] = torch.remainder(joint_batch['vars']['hd'] + joint_batch['vars']['sd'], 2*np.pi)
    joint_activity = np.concatenate((
        net.x_0.reshape((1, 1, -1)).repeat((batch_size, 1, 1)).cpu().detach().numpy(), joint_activity.numpy()
    ), axis=1)

    fig = plt.figure(figsize=(20, 15))
    gs = fig.add_gridspec(3, len(T)*2, height_ratios=[1,1,0.1])

    hd_cmap = plt.get_cmap('Reds')
    sd_cmap = plt.get_cmap('Blues')
    norm = matplotlib.colors.Normalize(vmin=0, vmax=2*np.pi)

    hd, sd = joint_batch['vars']['hd'], joint_batch['vars']['sd']
    ad = torch.remainder(hd + sd, 2*np.pi)
    if len(hd.shape)>1:
        hd, sd, ad = hd[:,0], sd[:,0], ad[:,0]

    sd_fixed = 1
    hd_varying = np.where(sd == sd[sd_fixed])[0]

    hd_fixed = np.where(ad == ad[sd_fixed])[0][0]
    sd_varying = np.where(hd == hd[hd_fixed])[0]

    for i, (eval_t, t_start, t_end) in enumerate(T):
        hd_varying_pca = PCA(n_components=2)
        hd_varying_pca.fit(joint_activity[hd_varying, eval_t].reshape((-1,net.n_neurons)))
        hd_varying_pca_activity = hd_varying_pca.transform(joint_activity[:, t_start:t_end].reshape((-1,net.n_neurons))).reshape((batch_size, -1, 2))

        sd_varying_pca = PCA(n_components=2)
        sd_varying_pca.fit(joint_activity[sd_varying, eval_t].reshape((-1,net.n_neurons)))
        sd_varying_pca_activity = sd_varying_pca.transform(joint_activity[:, t_start:t_end].reshape((-1,net.n_neurons))).reshape((batch_size, -1, 2))

        hd_ax = fig.add_subplot(gs[0, i*2:(i+1)*2])
        hd_ax.set_xticks([])
        hd_ax.set_yticks([])
        hd_ax.set_title(f'HD-Varying PC Slice\n' + fr'evaluated at $t={eval_t*net.dt:.2f}\tau$' + f'\n' + fr'showing ${t_start*net.dt:.2f}\tau <= t < {t_end*net.dt:.2f}\tau$', fontsize=16)
        for indices, linestyle, var, cmap in zip([hd_varying, sd_varying], ['-', '--'], [hd, sd], [hd_cmap, sd_cmap]):
            for j in indices:
                hd_ax.plot(hd_varying_pca_activity[j, :, 0], hd_varying_pca_activity[j, :, 1], color=cmap(norm(var[j])), linestyle=linestyle)
                hd_ax.scatter(hd_varying_pca_activity[j, 0, 0], hd_varying_pca_activity[j, 0, 1], s=100, color=cmap(norm(var[j])), marker='s')
                hd_ax.scatter(hd_varying_pca_activity[j, -1, 0], hd_varying_pca_activity[j, -1, 1], s=100, color=cmap(norm(var[j])), marker='^')

        sd_ax = fig.add_subplot(gs[1, i*2:(i+1)*2])
        sd_ax.set_xticks([])
        sd_ax.set_yticks([])
        sd_ax.set_title(f'HSA-Varying PC Slice\n' + fr'evaluated at $t={eval_t*net.dt:.2f}\tau$' + f'\n' + fr'showing ${t_start*net.dt:.2f}\tau <= t < {t_end*net.dt:.2f}\tau$', fontsize=16)
        for indices, linestyle, var, cmap in zip([hd_varying, sd_varying], ['--', '-'], [hd, sd], [hd_cmap, sd_cmap]):
            for j in indices:
                sd_ax.plot(sd_varying_pca_activity[j, :, 0], sd_varying_pca_activity[j, :, 1], color=cmap(norm(var[j])), linestyle=linestyle)
                sd_ax.scatter(sd_varying_pca_activity[j, 0, 0], sd_varying_pca_activity[j, 0, 1], s=100, color=cmap(norm(var[j])), marker='s')
                sd_ax.scatter(sd_varying_pca_activity[j, -1, 0], sd_varying_pca_activity[j, -1, 1], s=100, color=cmap(norm(var[j])), marker='^')


    hd_cax = fig.add_subplot(gs[-1, :len(T)])
    hd_cbar = matplotlib.colorbar.ColorbarBase(hd_cax, cmap=hd_cmap, norm=norm, orientation='horizontal')
    hd_cbar.set_label('Head Direction', fontsize=16)

    sd_cax = fig.add_subplot(gs[-1, len(T):])
    sd_cbar = matplotlib.colorbar.ColorbarBase(sd_cax, cmap=sd_cmap, norm=norm, orientation='horizontal')
    sd_cbar.set_label('Head-Shelter Angle', fontsize=16)

    fig.suptitle('Trajectories on HD-Varying and HSA-Varying PC Manifolds, for fixed ASA', fontsize=22, fontweight='bold')

    return fig

def plot_av_trajectories(task, net, av_batch, av_activity, dim=4, atol=1e-2):
    av_activity = av_activity.numpy()
    period_len = (av_activity.shape[1] - task.config.init_duration) // 5

    hd_start_1, sd_start_1 = 0, 0
    hd_start_2, sd_start_2 = np.pi, 0
    hd_start_3, sd_start_3 = 0, np.pi
    hd_start_4, sd_start_4 = np.pi, np.pi

    trial_1 = np.where(np.isclose(av_batch['vars']['hd'][:,0], hd_start_1) & np.isclose(av_batch['vars']['sd'][:,0], sd_start_1))[0][0]
    trial_2 = np.where(np.isclose(av_batch['vars']['hd'][:,0], hd_start_2) & np.isclose(av_batch['vars']['sd'][:,0], sd_start_2))[0][0]
    trial_3 = np.where(np.isclose(av_batch['vars']['hd'][:,0], hd_start_3) & np.isclose(av_batch['vars']['sd'][:,0], sd_start_3))[0][0]
    trial_4 = np.where(np.isclose(av_batch['vars']['hd'][:,0], hd_start_4) & np.isclose(av_batch['vars']['sd'][:,0], sd_start_4))[0][0]
    trials = [trial_1, trial_2, trial_3, trial_4]
    trial_cmap = plt.get_cmap('Set3')

    pca_start_t, pca_end_t = task.config.init_duration, task.config.init_duration+period_len
    pca = PCA(n_components=dim)
    pca.fit(av_activity[:, pca_start_t:pca_end_t].reshape((-1, net.n_neurons)))

    fig = plt.figure(figsize=(20, 15))
    gs = fig.add_gridspec(math.comb(dim, 2)+1, 5)

    for i in range(5):
        j = 0
        for pc_x in range(dim):
            for pc_y in range(pc_x+1,dim):
                ax = fig.add_subplot(gs[j, i])

                period_start_i = task.config.init_duration + period_len*i 
                period_end_i = period_start_i + period_len

                ref_points = pca.transform(av_activity[:, period_start_i+period_len//2].reshape((-1, net.n_neurons))).reshape((-1, dim))
                ax.scatter(ref_points[:,pc_x], ref_points[:,pc_y], color='white', alpha=0.5, s=5)

                trial_points = pca.transform(av_activity[trials, period_start_i:period_end_i].reshape((-1, net.n_neurons))).reshape((4, -1, dim))
                for k in range(len(trials)):
                    ax.scatter(trial_points[k, :, pc_x], trial_points[k, :, pc_y], s=25, color=trial_cmap(k), label=f'Trial {k}')
                    ax.scatter(trial_points[k, 0, pc_x], trial_points[k, 0, pc_y], s=100, color=trial_cmap(k), marker='s')
                    ax.scatter(trial_points[k, -1, pc_x], trial_points[k, -1, pc_y], s=100, color=trial_cmap(k), marker='^')

                if j==0:
                    titles = ['Stable', 'Clockwise', 'Stable', 'Counter-Clockwise', 'Stable']
                    ax.set_title(titles[i], fontsize=16, fontweight='bold', pad=20)
                if j==5:
                    ax.set_xlabel(f'PC {pc_x+1}')
                if i==0:
                    ax.set_ylabel(f'PC {pc_y+1}')
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_aspect('equal')

                j += 1

    cax = fig.add_subplot(gs[-1, :])
    trial_ends = [((hd_start_1, sd_start_1), (hd_start_1+np.pi, sd_start_1+np.pi)), ((hd_start_2, sd_start_2), (hd_start_2+np.pi, sd_start_2+np.pi)), ((hd_start_3, sd_start_3), (hd_start_3+np.pi, sd_start_3+np.pi)), ((hd_start_4, sd_start_4), (hd_start_4+np.pi, sd_start_4+np.pi))]
    labels = [r'$({:.2f}, {:.2f}) \leftrightarrow ({:.2f}, {:.2f})$'.format(*trial_ends[i][0], *trial_ends[i][1]) for i in range(4)]
    handles = [matplotlib.lines.Line2D([0], [0], color=trial_cmap(i), label=labels[i]) for i in range(4)]
    cax.legend(handles=handles, title='Trajectory', ncol=4, loc='center')
    cax.axis('off')

    fig.suptitle('Trajectories on PC manifold during rotation', fontsize=22, fontweight='bold')

    return fig


def plot_stability(task, net, joint_batch_size, total_n_timesteps=10000, n_timesteps=1000, slow_mult=10):
    assert int(total_n_timesteps/n_timesteps) == total_n_timesteps/n_timesteps, "total_n_timesteps must be divisible by n_timesteps"

    joint_task = task.get_subtask('joint', batch_size=joint_batch_size, n_timesteps=n_timesteps)
    joint_batch = o2s.data.TaskDataset(joint_task, include_noise=False).get_batch()

    with torch.no_grad():
        noisy_states = []
        silent_states = []
        slow_states = []

        noisy_output = []
        silent_output = []
        slow_output = []

        repeat_input = None
        ongoing_input = None
        if (joint_batch['inputs'][:,task.config.init_duration:]==0).all():
            joint_batch['inputs'] = joint_batch['inputs'][:,0]
            repeat_input = (n_timesteps, task.config.init_duration)
            ongoing_input = torch.zeros_like(joint_batch['inputs'])
        else:
            ongoing_input = joint_batch['inputs'][:,-1]

        for i in range(total_n_timesteps//n_timesteps):
            net.dt = task.config.dt
            
            if i==0:
                net.state_noise_std, net.rate_noise_std, net.output_noise_std = task.config.state_noise_std, task.config.rate_noise_std, task.config.output_noise_std
                noisy_states_, noisy_output_ = net(joint_batch['inputs'], repeat_input=repeat_input, offload=True)[0::2]
                noisy_states_, noisy_output_ = noisy_states_.detach().cpu(), noisy_output_.detach().cpu()
                net.state_noise_std = net.rate_noise_std = net.output_noise_std = 0
                silent_states_, silent_output_ =  net(joint_batch['inputs'], repeat_input=repeat_input, offload=True)[0::2]
                silent_states_, silent_output_ = silent_states_.detach().cpu(), silent_output_.detach().cpu()
            
            else:
                net.state_noise_std, net.rate_noise_std, net.output_noise_std = task.config.state_noise_std, task.config.rate_noise_std, task.config.output_noise_std
                noisy_states_, noisy_output_ = net(ongoing_input.to(net.device), x_0=noisy_states[-1][:,-1].to(net.device), repeat_input=(n_timesteps,n_timesteps), offload=True)[0::2]
                noisy_states_, noisy_output_ = noisy_states_.detach().cpu(), noisy_output_.detach().cpu()
                net.state_noise_std = net.rate_noise_std = net.output_noise_std = 0
                silent_states_, silent_output_ = net(ongoing_input.to(net.device), x_0=silent_states[-1][:,-1].to(net.device), repeat_input=(n_timesteps,n_timesteps), offload=True)[0::2]
                silent_states_, silent_output_ = silent_states_.detach().cpu(), silent_output_.detach().cpu()
            noisy_states.append(noisy_states_)
            silent_states.append(silent_states_)
            noisy_output.append(noisy_output_)
            silent_output.append(silent_output_)

            if repeat_input is not None:
                net.dt = task.config.dt/10
                net.state_noise_std = net.rate_noise_std = net.output_noise_std = 0
                for j in range(slow_mult):
                    if i==0 and j==0:
                        slow_states_, slow_output_ = net(joint_batch['inputs'], repeat_input=(n_timesteps, task.config.init_duration*10), offload=True)[0::2]
                        slow_states_, slow_output_ = slow_states_.detach().cpu(), slow_output_.detach().cpu()
                    else:
                        slow_states_, slow_output_ = net(ongoing_input.to(net.device), x_0=slow_states[-1][:,-1].to(net.device), repeat_input=(n_timesteps,1), offload=True)[0::2]
                        slow_states_, slow_output_ = slow_states_.detach().cpu(), slow_output_.detach().cpu()
                    slow_states.append(slow_states_)
                    slow_output.append(slow_output_)
            else:
                slow_states.append(torch.full((joint_batch['inputs'].shape[0], n_timesteps*slow_mult, net.n_neurons), np.nan))
                slow_output.append(torch.full((joint_batch['inputs'].shape[0], n_timesteps*slow_mult, task.config.n_outputs), np.nan))

        noisy_states = torch.cat(noisy_states, dim=1)
        silent_states = torch.cat(silent_states, dim=1)
        slow_states = torch.cat(slow_states, dim=1)
        print('Computed states')

        noisy_output = torch.cat(noisy_output, dim=1)
        silent_output = torch.cat(silent_output, dim=1)
        slow_output = torch.cat(slow_output, dim=1)

        target = torch.cat((
            joint_batch['targets'], joint_batch['targets'][:,-1].reshape((-1,1,net.n_outputs)).repeat((1,total_n_timesteps-n_timesteps,1))
        ), dim=1).cpu()

        noisy_error = torch.mean(torch.mean((noisy_output-target)**2, dim=2), dim=0)
        silent_error = torch.mean(torch.mean((silent_output-target)**2, dim=2), dim=0)
        if not torch.isnan(slow_output).any():
            slow_error = torch.mean(torch.mean((slow_output-target.repeat((1,slow_mult,1)))**2, dim=2), dim=0)
        else:
            slow_error = torch.full((total_n_timesteps*slow_mult,), np.nan)
        del noisy_output, silent_output, slow_output, target
        print('Computed errors')

        noisy_energy = np.full((total_n_timesteps,), np.nan)
        silent_energy = np.full((total_n_timesteps,), np.nan)
        slow_energy = np.full((total_n_timesteps*slow_mult,), np.nan)

        def F(x, u):
            x, u = x.to(net.device), u.to(net.device)
            return (-x + net.W_rec(net.activation_func(x)) + net.W_in(u)).cpu().detach().numpy()

        def q(x, u):
            q_ = (1/2) * np.linalg.norm(F(x, u), axis=1)**2
            return np.mean(q_, axis=0)

        # Compute energy and loss for each timestep
        for t in range(total_n_timesteps):
            if repeat_input is None:
                if t<n_timesteps:
                    silent_energy_ = q(silent_states[:,t], joint_batch['inputs'][:,t])
                    noisy_energy_ = q(noisy_states[:,t], joint_batch['inputs'][:,t])
                    silent_energy[t] = silent_energy_.item()
                    noisy_energy[t] = noisy_energy_.item()
                else:
                    silent_energy_ = q(silent_states[:,t], ongoing_input)
                    noisy_energy_ = q(noisy_states[:,t], ongoing_input)
                    silent_energy[t] = silent_energy_.item()
                    noisy_energy[t] = noisy_energy_.item()
            else:
                if t<task.config.init_duration:
                    silent_energy_ = q(silent_states[:,t], joint_batch['inputs'])
                    noisy_energy_ = q(noisy_states[:,t], joint_batch['inputs'])
                    silent_energy[t] = silent_energy_.item()
                    noisy_energy[t] = noisy_energy_.item()
                elif t<total_n_timesteps:
                    silent_energy_ = q(silent_states[:,t], ongoing_input)
                    noisy_energy_ = q(noisy_states[:,t], ongoing_input)
                    noisy_energy[t] = noisy_energy_.item()
                    silent_energy[t] = silent_energy_.item()
        
        if repeat_input is not None:
            for t in range(total_n_timesteps*slow_mult):
                if t<task.config.init_duration*slow_mult:
                    slow_energy_ = q(slow_states[:,t], joint_batch['inputs'])
                    slow_energy[t] = slow_energy_.item()
                else:
                    slow_energy_ = q(slow_states[:,t], ongoing_input)
                    slow_energy[t] = slow_energy_.item()

        print('Computed energy')

    fig = plt.figure(figsize=(20,20))
    gs = fig.add_gridspec(nrows=2, ncols=1, hspace=0.5)

    noisy_x = silent_x = np.linspace(0, total_n_timesteps*task.config.dt, total_n_timesteps)
    slow_x = np.linspace(0, total_n_timesteps*task.config.dt, total_n_timesteps*slow_mult)

    ax = fig.add_subplot(gs[0, 0])
    ax.plot(noisy_x, noisy_energy, label='Noisy Energy', color='red')
    ax.plot(silent_x, silent_energy, label='Silent Energy', color='blue')
    ax.plot(slow_x, slow_energy, label='Slow Energy', color='green')
    # ax.axvline(x=task.config.init_duration*net.dt, color='white', linestyle='--')
    ax.set_xlabel(r'Time ($\tau$)')
    ax.set_ylabel('Energy')
    ax.set_yscale('log')

    ax2 = ax.twinx()
    ax2.plot(noisy_x, noisy_error, label='Noisy Error', color='red', linestyle='--')
    ax2.plot(silent_x, silent_error, label='Silent Error', color='blue', linestyle='--')
    ax2.plot(slow_x, slow_error, label='Slow Error', color='green', linestyle='--')
    ax2.set_ylabel('MSE')
    ax2.set_yscale('log')

    handles = [matplotlib.lines.Line2D([0], [0], color='blue', label='Silent'),
                matplotlib.lines.Line2D([0], [0], color='red', label='Noisy'),
                matplotlib.lines.Line2D([0], [0], color='green', label='Slow'),
                matplotlib.lines.Line2D([0], [0], color='gray', linestyle='--', label='Error'),
                matplotlib.lines.Line2D([0], [0], color='gray', linestyle='-', label='Energy')]
    ax.legend(handles=handles, loc='upper right')
               


    unit_cmap = plt.get_cmap('plasma')
    variance = torch.var(silent_states, dim=1)
    max_variance_trial = torch.argmax(torch.sum(variance, dim=1)).item()

    ax = fig.add_subplot(gs[1, 0])
    for i in range(net.n_neurons):
        ax.plot(silent_x, silent_states[max_variance_trial, :, i], color=unit_cmap(i/net.n_neurons))
    ax.axvline(x=task.config.init_duration*net.dt, color='white', linestyle='--')
    ax.set_title('State of neurons in example trial', fontsize=16)
    ax.set_xlabel(r'Time ($\tau$)')
    ax.set_ylabel('State')

    fig.suptitle('Stability of network for long times', fontsize=22, fontweight='bold')

    return fig



def plot_lesions(task, net, n_lesions=100):
    fig = plt.figure(figsize=(20, 20))
    gs = fig.add_gridspec(1+net.n_inputs, 2)

    inverse_input_map = {v:k for k,v in task.input_map.items()}
    net_state_dict = net.state_dict().copy()

    weights = [net.W_rec.weight.detach().cpu().clone()] + [net.W_in.weight.detach().cpu()[:,i].clone() for i in range(net.n_inputs)]
    titles = ['Recurrent weights'] + [f'Input {i+1} ({inverse_input_map[i]})' for i in range(net.n_inputs)]
    batch = o2s.data.TaskDataset(task, include_noise=False).get_batch()
    inputs, targets = batch['inputs'], batch['targets'].cpu().numpy()
    del batch

    mse = np.full((len(weights), n_lesions), np.nan)
    n = np.full((len(weights), n_lesions), np.nan)

    cmap = plt.get_cmap('Set3')

    for i, weight in enumerate(weights):
        ax = fig.add_subplot(gs[i, 0])
        ax.hist(weight.flatten(), bins=50)#, color=cmap(i))
        if i==len(weights)-1:
            ax.set_xlabel('Weight')
        ax.set_ylabel(titles[i], rotation=90, labelpad=20, fontsize=16, fontweight='bold')
        print(f'Lesioning {titles[i]}')

        lesion_sizes = np.logspace(0, torch.log10(torch.max(torch.abs(weight))+1).item(), n_lesions)-1
        for j, lesion_size in enumerate(lesion_sizes):
            lesion = torch.abs(weight) <= lesion_size
            n_ = torch.sum(lesion).item()
            n[i,j] = n_

            weight_lesion = weight.clone()
            weight_lesion[lesion] = 0
            if i == 0:
                net.W_rec.weight = nn.Parameter(weight_lesion.to(net.device))
            else:
                W_in = net.W_in.weight.data.clone()
                W_in[:,i-1] = weight_lesion
                net.W_in.weight = nn.Parameter(W_in.to(net.device))

            with torch.no_grad():
                outputs = net(inputs, offload=True)[2].detach().cpu().numpy()
                mse_ = np.mean((outputs[:,task.config.init_duration:]-targets[:,task.config.init_duration:])**2)
                mse[i,j] = mse_

            net.load_state_dict(net_state_dict)

        ax = fig.add_subplot(gs[i, 1])
        ax.plot(n[i], mse[i], color=cmap(i))
        if i==len(weights)-1:
            ax.set_xlabel('Number of weights lesioned')
        ax.set_ylabel('MSE')

    fig.suptitle('Effect of lesioning weights on network performance', fontsize=22, fontweight='bold')

    return fig


def fit_tuning_curves(tuning_vars, tuning_dict):
    assert 'HD' in tuning_vars, "HD must be in tuning_vars"
    assert 'AV' in tuning_vars, "AV must be in tuning_vars"
    assert 'ego_SD' in tuning_vars, "ego_SD must be in tuning_vars"
    assert 'allo_SD' in tuning_vars, "allo_SD must be in tuning_vars"


    def fit_cosine(x: np.array, y: np.array):
        assert len(x)==len(y)

        p0 = [1, 0, 0]
        p, _ = curve_fit(lambda x, A, B, C: A*np.cos(np.pi*x/180 + B) + C, x, y, p0=p0, bounds=([0, -np.pi, 0], [1, np.pi, 1]))
        return p[0], p[1], p[2]

    def fit_linear(x: np.array, y: np.array):
        assert len(x)==len(y)

        p0 = [1, 0]
        p, _ = curve_fit(lambda x, A, B: A*x + B, x, y, p0=p0, bounds=([-np.inf, 0], [np.inf, 1]))
        return p[0], p[1], np.nan

    def fit_all(var, tuning_vars, tuning_dict, func):
        assert var in tuning_vars and f'{var}_tuning' in tuning_dict
        tuning_grid = tuning_dict[f'{var}_tuning'] 
        n_neurons = tuning_grid.shape[0]

        coefs = np.full((n_neurons, 3), np.nan)

        x = tuning_vars[var]['bins']
    
        for neuron in range(n_neurons):
            y = tuning_grid[neuron]
            A, B, C = func(x, y)
            coefs[neuron] = [A, B, C]

        return coefs

    hd_coefs = fit_all('HD', tuning_vars, tuning_dict, fit_cosine)
    sd_coefs = fit_all('ego_SD', tuning_vars, tuning_dict, fit_cosine)
    ad_coefs = fit_all('allo_SD', tuning_vars, tuning_dict, fit_cosine)
    av_coefs = fit_all('AV', tuning_vars, tuning_dict, fit_linear)

    return {'HD': hd_coefs, 'ego_SD': sd_coefs, 'allo_SD': ad_coefs, 'AV': av_coefs}


def plot_tuning_dist(tuning_vars, coefs):
    fig = plt.figure(figsize=(20, 20))
    gs = fig.add_gridspec(4, 4)

    cmap = plt.get_cmap('Set3')

    titles = [tuning_vars[var]['title'] for var in coefs]
    coefs = list(coefs.values())

    left_ax = [None for i in range(4)]
    top_ax = [None for i in range(4)]
    for i in range(4):
        for j in range(4):
            if j>i:
                continue
            elif i==j:
                ax = fig.add_subplot(gs[i, j])

                ax.hist(coefs[i][:,0], bins=30, color=cmap(i))
            else:
                ax = fig.add_subplot(gs[i, j], sharex=top_ax[j], sharey=left_ax[i])

                ax.scatter(coefs[j][:,0], coefs[i][:,0], color='white', alpha=0.5)
                ax.axvline(x=0, color='white', linestyle='--')
                ax.axhline(y=0, color='white', linestyle='--')

                if top_ax[j] is None:
                    top_ax[j] = ax
                if left_ax[i] is None:
                    left_ax[i] = ax

            if i==3:
                ax.set_xlabel(titles[j], fontsize=16, fontweight='bold')
            
            if j==0:
                ax.set_ylabel(titles[i], fontsize=16, fontweight='bold')

    fig.suptitle('Distribution of tuning curve coefficients', fontsize=22, fontweight='bold', y=0.9)

    return fig


def plot_tuned_weights(task, net, coefs):
    fig = plt.figure(figsize=(20, 15))
    gs = fig.add_gridspec(nrows=5, ncols=6, wspace=0.1, hspace=0.5, height_ratios=[1, 0.3, 0.3, 0.3, 0.1])

    tuned_cutoff = 0.0
    n_bins = 50
    W_rec = net.W_rec.weight.detach().cpu().numpy()

    hd_coefs, sd_coefs, ad_coefs, av_coefs = coefs['HD'], coefs['ego_SD'], coefs['allo_SD'], coefs['AV']

    for i, pre in enumerate(['hd', 'sd', 'ad']):

        row_axes = []
        for j, post in enumerate(['hd', 'sd', 'ad']):
            W_rec_tuned = None

            pre_phases = None

            pre_angle_tuning = None
            pre_av_tuning = None
            if pre=='hd':
                pre_phases = hd_coefs[hd_coefs[:, 0] > tuned_cutoff][:, 1]
                W_rec_tuned = W_rec[:,hd_coefs[:, 0] > tuned_cutoff]

                pre_angle_tuning = hd_coefs[hd_coefs[:, 0] > tuned_cutoff, 0]
                pre_av_tuning = av_coefs[hd_coefs[:, 0] > tuned_cutoff, 0]
            elif pre=='sd':
                pre_phases = sd_coefs[sd_coefs[:, 0] > tuned_cutoff][:, 1]
                W_rec_tuned = W_rec[:,sd_coefs[:, 0] > tuned_cutoff]

                pre_angle_tuning = sd_coefs[sd_coefs[:, 0] > tuned_cutoff, 0]
                pre_av_tuning = av_coefs[sd_coefs[:, 0] > tuned_cutoff, 0]
            elif pre=='ad':
                pre_phases = ad_coefs[ad_coefs[:, 0] > tuned_cutoff][:, 1]
                W_rec_tuned = W_rec[:,ad_coefs[:, 0] > tuned_cutoff]

                pre_angle_tuning = ad_coefs[ad_coefs[:, 0] > tuned_cutoff, 0]
                pre_av_tuning = av_coefs[ad_coefs[:, 0] > tuned_cutoff, 0]
            pre_phases[pre_phases<0] += 2*np.pi

            post_phases = None

            post_angle_tuning = None
            post_av_tuning = None
            if post=='hd':
                post_phases = hd_coefs[hd_coefs[:, 0] > tuned_cutoff][:, 1]
                W_rec_tuned = W_rec_tuned[hd_coefs[:, 0] > tuned_cutoff]

                post_angle_tuning = hd_coefs[hd_coefs[:, 0] > tuned_cutoff, 0]
                post_av_tuning = av_coefs[hd_coefs[:, 0] > tuned_cutoff, 0]
            elif post=='sd':
                post_phases = sd_coefs[sd_coefs[:, 0] > tuned_cutoff][:, 1]
                W_rec_tuned = W_rec_tuned[sd_coefs[:, 0] > tuned_cutoff]

                post_angle_tuning = sd_coefs[sd_coefs[:, 0] > tuned_cutoff, 0]
                post_av_tuning = av_coefs[sd_coefs[:, 0] > tuned_cutoff, 0]
            elif post=='ad':
                post_phases = ad_coefs[ad_coefs[:, 0] > tuned_cutoff][:, 1]
                W_rec_tuned = W_rec_tuned[ad_coefs[:, 0] > tuned_cutoff]

                post_angle_tuning = ad_coefs[ad_coefs[:, 0] > tuned_cutoff, 0]
                post_av_tuning = av_coefs[ad_coefs[:, 0] > tuned_cutoff, 0]
            post_phases[post_phases<0] += 2*np.pi
            
            pre_order, post_order = np.argsort(pre_phases), np.argsort(post_phases)
            pre_i, post_i = np.meshgrid(pre_order, post_order)
            pre_i, post_i = pre_i.flatten(), post_i.flatten()

            equal_mask = pre_i == post_i
            pre_i = pre_i[~equal_mask]
            post_i = post_i[~equal_mask]

            diff = np.remainder(post_phases[post_i] - pre_phases[pre_i], 2*np.pi)

            diff_bins = np.linspace(0, 2*np.pi, n_bins+2)[1:-1]
            diff_bins_i = np.digitize(diff, diff_bins)

            W_rec_tuned_pre_post = W_rec_tuned[post_i, pre_i]

            mean_tuning_scaled_weights = np.full((4, n_bins,), np.nan)
            std_tuning_scaled_weights = np.full((4, n_bins,), np.nan)
            n_synapses = np.full((n_bins,), np.nan)
            for k in range(n_bins):
                bin_mask = diff_bins_i == k
                n_synapses[k] = np.sum(bin_mask)

                for l, (var, order) in enumerate(zip([pre_angle_tuning, pre_av_tuning, post_angle_tuning, post_av_tuning], [pre_i, pre_i, post_i, post_i])):
                    var_tuning_scaled_weights = W_rec_tuned_pre_post * var[order]

                    mean_tuning_scaled_weights[l, k] = np.mean(var_tuning_scaled_weights[bin_mask])
                    std_tuning_scaled_weights[l, k] = np.std(var_tuning_scaled_weights[bin_mask])

            ax = None
            if pre==post:
                ax = fig.add_subplot(gs[0, 2*j:2*j+2])
            else:
                j_ = [
                    [None, 0, 1],
                    [0, None, 1],
                    [0, 1, None]
                ][i][j]
                ax = fig.add_subplot(gs[i+1, 3*j_:3*j_+3])

            for l, (label, color) in enumerate(zip(['Pre-Angle', 'Pre-AV', 'Post-Angle', 'Post-AV'], ['blue', 'green', 'red', 'purple'])):
                mean = mean_tuning_scaled_weights[l]
                std = std_tuning_scaled_weights[l]

                sem = std / np.sqrt(n_synapses)
                ax.fill_between(
                    diff_bins,
                    mean - sem,
                    mean + sem,
                    alpha=0.2,
                    color=color
                )

                ax.plot(diff_bins, mean, color=color, label=label)
                ax.hlines(0, xmin=0, xmax=2*np.pi, color='white', linestyle='--')
                ax.set_xlabel(f'{pre.upper()}-{post.upper()} Difference')

                if (pre==post and j==0) or (pre != post and j_==0):
                    ax.set_ylabel(f'Mean Scaled Weight')
                else:
                    ax.set_yticks([])

            row_axes.append(ax)

        y_min, y_max = np.min([ax.get_ylim()[0] for ax in row_axes]), np.max([ax.get_ylim()[1] for ax in row_axes])
        for ax in row_axes:
            ax.set_ylim(y_min, y_max)

    cax = fig.add_subplot(gs[-1, :])
    cax.axis('off')
    handles = [matplotlib.lines.Line2D([0], [0], color=color, label=label) for label, color in zip(['Pre-Synaptic Angle', 'Pre-Synaptic AV', 'Post-Synaptic Angle', 'Post-Synaptic Angle'], ['blue', 'green', 'red', 'purple'])]
    cax.legend(handles=handles, title='Average Synapse Strength Weighted by Tuning', ncol=4, loc='center')

    fig.suptitle('Synaptic strength weighted by tuning', fontsize=22, fontweight='bold', y=0.9)

    return fig


def plot_fourier_weights(task, net):
    torch.manual_seed(task.config.build_seed)
    # W_rec_0 = o2s.net.RNN(task).W_rec.weight.detach().cpu().numpy()
    W_rec = net.W_rec.weight.detach().cpu().numpy() # - W_rec_0
    W_rec_fft = np.fft.fft2(W_rec)

    W_rec_fft_magnitude = np.abs(W_rec_fft)

    fig = plt.figure(figsize=(10,10))
    gs = fig.add_gridspec(nrows=2, ncols=2, width_ratios=[1,0.5], height_ratios=[0.5,1])

    axs = gs.subplots()

    axs[1, 0].imshow(W_rec_fft_magnitude, 
                    cmap='viridis', 
                    extent=[0, net.W_rec.weight.shape[0], 0, net.W_rec.weight.shape[1]], 
                    norm=matplotlib.colors.Normalize(vmin=0, vmax=np.max(W_rec_fft_magnitude)))
    axs[1, 0].set_xlabel('X Frequency')
    axs[1, 0].set_ylabel('Y Frequency')

    y_cmap = plt.get_cmap('Blues')
    for i in range(net.n_neurons):
        axs[0, 0].plot(np.arange(net.W_rec.weight.shape[0]), W_rec_fft_magnitude[i,:], color=y_cmap(i/net.n_neurons))
    axs[0, 0].set_ylabel('Magnitude holding X Frequency constant')
    axs[0, 0].set_ylim([0, np.max(W_rec_fft_magnitude)])
    axs[0, 0].set_xticks([])

    x_cmap = plt.get_cmap('Reds')
    for i in range(net.n_neurons):
        axs[1, 1].plot(W_rec_fft_magnitude[:,i], np.arange(net.W_rec.weight.shape[1]), color=x_cmap(i/net.n_neurons))
    axs[1, 1].set_xlabel('Magnitude holding Y Frequency constant')
    axs[1, 1].set_xlim([0, np.max(W_rec_fft_magnitude)])
    axs[1, 1].set_yticks([])

    cax = gs[0,1].subgridspec(3, 1, hspace=1).subplots()

    cbar1 = fig.colorbar(matplotlib.cm.ScalarMappable(cmap='Blues'), cax=cax[0], orientation='horizontal')
    cbar1.set_label('Y Frequency')

    cbar2 = fig.colorbar(matplotlib.cm.ScalarMappable(cmap='Reds'), cax=cax[1], orientation='horizontal')
    cbar2.set_label('X Frequency')

    cbar3 = fig.colorbar(matplotlib.cm.ScalarMappable(cmap='viridis'), cax=cax[2], orientation='horizontal')
    cbar3.set_label('Magnitude at (x,y)')

    axs[0, 1].set_axis_off()

    cax[0].set_xticks([0, 1])
    cax[0].set_xticklabels(['0', f'{net.n_neurons}'])
    cax[0].set_yticks([])

    cax[1].set_xticks([0, 1])
    cax[1].set_xticklabels(['0', f'{net.n_neurons}'])
    cax[1].set_yticks([])

    cax[2].set_xticks([0, 1])
    cax[2].set_xticklabels(['0', f'{np.max(W_rec_fft_magnitude):.2f}'])
    cax[2].set_yticks([])

    fig.suptitle('Fourier Transform of Recurrent Weights', fontsize=22, fontweight='bold')

    return fig




def plot_joint_eigenspectra(task, net, joint_task, joint_batch, eval_t=250):
    n_samples = int(np.sqrt(joint_task.config.batch_size))

    with torch.no_grad():
        states = net(joint_batch['inputs'], repeat_input=joint_task.config.repeat_input, offload=True)[0].detach()

        def F(x, u):
            x, u = x.to(net.device), u.to(net.device)
            return (-x + net.W_rec(net.activation_func(x)) + net.W_in(u)).detach().cpu().numpy()

        def q(x, u):
            q_ = (1/2) * np.linalg.norm(F(x, u), axis=1)**2
            return q_

        Q = np.full((n_samples**2, joint_task.config.repeat_input[0]), fill_value=np.nan)
        for t in range(joint_task.config.repeat_input[1], joint_task.config.repeat_input[0]):
            Q[:,t] = q(states[:,t], torch.zeros((n_samples**2, net.n_inputs), device=states.device))


    def get_Jacobian_at(x):
        n = net.n_neurons
        W = net.W_rec.weight.detach().cpu().numpy()
        r_prime = np.where( x < 0 , np.zeros_like(x) , 1-(np.tanh(x)**2) )

        J = np.zeros_like(W)
        for i in range(n):
            for j in range(n):
                delta = 1 if i==j else 0
                J[i,j] = -delta + W[i,j]*r_prime[j]

        return J

    def get_eigenvalues_at(x):
        J = get_Jacobian_at(x)
        return np.linalg.eigvals(J)

    def get_stability_at(x):
        eigenvalues = get_eigenvalues_at(x)
        return np.all(np.abs(eigenvalues) < 1)
        
    fig = plt.figure(figsize=(20, 20))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 0.1])
    ax = np.array([[fig.add_subplot(gs[i, j]) for j in range(2)] for i in range(2)])

    hd_cmap = plt.get_cmap('Reds')
    sd_cmap = plt.get_cmap('Blues')
    norm = matplotlib.colors.Normalize(vmin=0, vmax=2*np.pi)

    for i in range(n_samples):
        mean_q_hd = np.nanmean(Q[i*n_samples:(i+1)*n_samples], axis=0)
        ax[0,0].plot(mean_q_hd, color=hd_cmap(norm(joint_batch['vars']['hd'][i*n_samples])))
        ax[0,0].set_yscale('log')
        ax[0,0].axvline(eval_t, color='white', linestyle='--')
        ax[0,0].set_title('Average network energy at angle', fontsize=16)

        mean_q_sd = np.nanmean(Q[i::n_samples], axis=0)
        ax[1,0].plot(mean_q_sd, color=sd_cmap(norm(joint_batch['vars']['sd'][i])))
        ax[1,0].set_yscale('log')
        ax[1,0].axvline(eval_t, color='white', linestyle='--')

        hd_states = states[i*n_samples:(i+1)*n_samples, eval_t].cpu().numpy()
        mean_state_hd = np.tile(np.mean(hd_states, axis=0).reshape((1,-1)), (n_samples, 1))
        med_state_hd_i = np.argmin(np.linalg.norm(hd_states - mean_state_hd, axis=1), axis=0).item()
        med_state_hd = hd_states[med_state_hd_i]
        hd_eigs = get_eigenvalues_at(med_state_hd)
        hd_eigs = hd_eigs[hd_eigs.real >= 0]
        ax[0,1].scatter(hd_eigs.real, hd_eigs.imag, color=hd_cmap(norm(joint_batch['vars']['hd'][i*n_samples])))
        ax[0,1].axvline(0, color='white', linestyle='--', linewidth=0.5)
        ax[0,1].axhline(0, color='white', linestyle='--', linewidth=0.5)
        ax[0,1].set_ylim([-1,1])
        ax[0,1].set_title('Eigenspectrum of median state at angle', fontsize=16)


        sd_states = states[i::n_samples, eval_t].cpu().numpy()
        mean_state_sd = np.tile(np.mean(sd_states, axis=0).reshape((1,-1)), (n_samples, 1))
        med_state_sd_i = np.argmin(np.linalg.norm(sd_states - mean_state_sd, axis=1), axis=0).item()
        med_state_sd = sd_states[med_state_sd_i]
        sd_eigs = get_eigenvalues_at(med_state_sd)
        sd_eigs = sd_eigs[sd_eigs.real >= 0]
        ax[1,1].scatter(sd_eigs.real, sd_eigs.imag, color=sd_cmap(norm(joint_batch['vars']['sd'][i])))
        ax[1,1].axvline(0, color='white', linestyle='--', linewidth=0.5)
        ax[1,1].axhline(0, color='white', linestyle='--', linewidth=0.5)
        ax[1,1].set_ylim([-1, 1])

    cax1 = fig.add_subplot(gs[-1, 0])
    hd_cbar = matplotlib.colorbar.ColorbarBase(cax1, cmap=hd_cmap, norm=norm, orientation='horizontal')
    hd_cbar.set_label('Head Direction', fontsize=16)
    cax2 = fig.add_subplot(gs[-1, 1])
    sd_cbar = matplotlib.colorbar.ColorbarBase(cax2, cmap=sd_cmap, norm=norm, orientation='horizontal')
    sd_cbar.set_label('Head-Shelter Angle', fontsize=16)

    fig.suptitle(f'Eigenspectrum of network stabilised at different angles', fontsize=22, fontweight='bold')

    return fig



'''
test_tuning
Battery of tuning-related plots for path- and head-direction-integration tasks
---------------------------------------------------------------------------------------------
Receives
    task :
        Task on which net was trained
    net :
        Trained o2s.net.RNN to be test
    batch :
        Data batch for testing with
    tuning_vars_list :
        List of variable names to check neuron tuning to (must be compatible with test_tuning_generalised below)
    checkpoint_path :
        Path to checkpoint where plots should be saved

Returns
    dict :
        Values are matplotlib figure objects containing the plots generated
'''
def test_tuning(task: o2s.task.Task, net: o2s.net.RNN, batch: dict, tuning_vars_list: list, checkpoint_path:str = None, 
                tuning_vars=None, tuning_dict=None, **kwargs) -> Dict[str, matplotlib.figure.Figure]:
    assert (tuning_vars is None and tuning_dict is None) or (tuning_vars is not None and tuning_dict is not None)

    print('Testing model')

    figures = {}

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Training Loss ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
    if not kwargs.get('ignore_loss', False):

        # Only create loss plot if checkpoint is supplied (where losses are saved)
        if checkpoint_path is not None:

            try:
                # Retrieve losses
                checkpoint = torch.load(f'{checkpoint_path}', map_location=torch.device(task.config.device))
                test_losses = checkpoint['test_losses']
                train_losses = checkpoint['train_losses']

                figures['loss'] = loss_plot(task, test_losses, train_losses)

                print('\tGenerated loss plot')
            except Exception as e:
                print(f'\tLoss plot generation failed: {e}')


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Examples ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


    _, activity, outputs = net(batch['inputs'], offload=True)

    # Detach resulting tensors for use with numpy-based matplotlib
    inputs, targets, mask = batch['inputs'].detach().cpu().numpy(), batch['targets'].detach().cpu().numpy(), batch['mask'].detach().cpu().numpy()
    vars = {key: var.detach().cpu().numpy() for key,var in batch['vars'].items()}
    activity, outputs = activity.detach().cpu().numpy(), outputs.detach().cpu().numpy()

    if not kwargs.get('ignore_examples', False):

        try:
            # Generate fit examples plot
            figures['fit_examples'] = fit_examples_plot(task, targets, outputs, n_fit_examples=3)

            print('\tGenerated fit example plot')
        except Exception as e:
            print(f'\tFit example plot generation failed: {e}')
        
        # If 2D, generate some examples of path integration performance
        if '2D' in task.name:

            try:
                figures['path_integration'] = path_integration_plot(task, targets, outputs)

                print('\tGenerated path_integration plot')
            except Exception as e:
                print(f'\tPath integration plot generation failed: {e}')





# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Tuning ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

    if not kwargs.get('ignore_tuning', False):

        # Calculate neuron tunings to variables and variable pairs
        if tuning_vars is None and tuning_dict is None:
            tuning_vars, tuning_dict = get_tuning_generalised(task, inputs, targets, vars, activity, tuning_vars_list)

        angle_vars = ['HD', 'ego_SD', 'allo_SD']
        angle_colors = ['#B51700', '#18E7CF', '#EF5FA7']
        included_angle_vars = [var for var in tuning_vars_list if var in angle_vars]

        # Generate plots
        # Angular variables (stacked onto one plot because they share x-axis)
        if len(included_angle_vars) > 0:

            try:
                figures['angle_vars'] = univar_tuning_plot(task, tuning_vars, tuning_dict,
                                        list(zip(included_angle_vars, angle_colors[:len(included_angle_vars)])))

                print('\tGenerated angular variable tuning plot')
            except Exception as e:
                print(f'\tAngular variable tuning plot generation failed: {e}')

        # Angular velocity
        if 'AV' in tuning_vars_list:
            
            try:
                figures['AV_tuning'] = univar_tuning_plot(task, tuning_vars, tuning_dict,
                                       [('AV', 'white')])

                print('\tGenerated angular velocity tuning plot')
            except Exception as e:
                print(f'\tAngular velocity tuning plot generation failed: {e}')

        pos_vars = ['x', 'y']
        pos_colors = ['#D98324', '#246EB9']
        included_pos_vars = [var for var in tuning_vars_list if var in pos_vars]

        # Positional plots
        if len(included_pos_vars) > 0:
            
            try:
                figures['pos_vars'] = univar_tuning_plot(task, tuning_vars, tuning_dict,
                                        list(zip(included_pos_vars, pos_colors[:len(included_pos_vars)])))

                print('\tGenerated positional variable tuning plot')
            except Exception as e:
                print(f'\tPositional variable tuning plot generation failed: {e}')

        # AV to angular variable tuning plots
        if 'AV' in tuning_vars_list:
            for var in included_angle_vars:
                
                try:

                    figures[f'{var}_AV'] = bivar_tuning_plot(task, tuning_vars, tuning_dict, vars=(var, 'AV'))
                    print(f'\tGenerated {var}-AV tuning plot')
                except Exception as e:
                    print(f'\t{var}-AV tuning plot generation failed: {e}')

        # Angle to angle tuning plots
        for var_a_index, var_a in enumerate(included_angle_vars):
            for var_b_index, var_b in enumerate(included_angle_vars):
                if var_b_index <= var_a_index:
                    continue

                try:
                    figures[f'{var_a}_{var_b}'] = bivar_tuning_plot(task, tuning_vars, tuning_dict, vars=(var_a, var_b))
                    print(f'\tGenerated {var_a}-{var_b} tuning plot')
                except Exception as e:
                    print(f'\t{var_a}-{var_b} tuning plot generation failed: {e}')

        # x-y tuning plot
        if 'x' in tuning_vars_list and 'y' in tuning_vars_list:

            try:
                figures['x_y'] = bivar_tuning_plot(task, tuning_vars, tuning_dict, vars=('x', 'y'))
                print(f'\tGenerated x-y tuning plot')
            except Exception as e:
                print(f'\tx-y tuning plot generation failed: {e}')

    # Tuning manifold plots
    if not kwargs.get('ignore_manifold', False):

        for var in tuning_vars_list:

            try:
                figures[f'{var}_manifold'] = manifold_plot(task, activity, tuning_vars, tuning_dict, var)
                print(f'\tGenerated {var} manifold plot')
            except Exception as e:
                print(f'\t{var} manifold plot generation failed: {e}')

            




# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Save ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

    # If checkpoint path is supplied, save plots as .png's
    if checkpoint_path is not None:
        checkpoint_dir = '/'.join(checkpoint_path.split('/')[:-1])
        for name, fig in figures.items():
            fig.savefig(f'{checkpoint_dir}/{name}.png', transparent=False)

        print('\tSuccesfully saved plots.')

    return figures


'''
test_general
Battery of plots for general task
---------------------------------------------------------------------------------------------
Receives
    task :
        Task on which net was trained
    net :
        Trained o2s.net.RNN to be test
    batch :
        Data batch for testing with
    checkpoint_path :
        Path to checkpoint where plots should be saved

Returns
    dict :
        Values are matplotlib figure objects containing the plots generated
'''
def test_general(task: o2s.task.Task, net: o2s.net.RNN, batch: dict, checkpoint_path: str = None, **kwargs) -> Dict[str, matplotlib.figure.Figure]:

    print('Testing model')

    figures = {}

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Loss ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

    if not kwargs.get('ignore_loss', False):

        # Only create loss plot if checkpoint is supplied (where losses are saved)
        if checkpoint_path is not None:

            try:
                # Retrieve losses
                checkpoint = torch.load(f'{checkpoint_path}', map_location=torch.device(task.config.device))
                test_losses = checkpoint['test_losses']
                train_losses = checkpoint['train_losses']

                figures['loss'] = loss_plot(task, test_losses, train_losses)

                print('\tGenerated loss plot.')
            except Exception as e:
                print(f'\tLoss plot generation failed: {e}')




# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Examples ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


    _, activity, outputs = net(batch['inputs'], noise=batch['noise'])

    # Detach resulting tensors for use with numpy-based matplotlib
    inputs, targets, mask = batch['inputs'].detach().cpu().numpy(), batch['targets'].detach().cpu().numpy(), batch['mask'].detach().cpu().numpy()
    activity, outputs = activity.detach().cpu().numpy(), outputs.detach().cpu().numpy()

    time_mask = mask[0,:,0]
    inputs, targets = inputs[:,time_mask,:], targets[:,time_mask,:]
    activity, outputs = activity[:,time_mask,:], outputs[:,time_mask,:]

    if not kwargs.get('ignore_examples', False):

        try:
            # Generate fit examples plot
            figures['fit_examples'] = fit_examples_plot(task, targets, outputs)

            print('\tGenerated fit example plot')
        except Exception as e:
            print(f'\tFit example plot generation failed: {e}')




# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Snapshot ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

    # If checkpoint path is supplied, save plots as .png's
    if checkpoint_path is not None:
        checkpoint_dir = '/'.join(checkpoint_path.split('/')[:-1])
        for name, fig in figures.items():
            fig.savefig(f'{checkpoint_dir}/{name}.png', transparent=False)

        print('\tSuccesfully saved plots.')

        figure_pastes = {
            'loss': (3, 2, 0, 0),
            'fit_examples': (3, 2, 0, 2),
        }

        # Also save an image which contains all plots
        create_snapshot_image(task, figures, checkpoint_dir, figure_pastes, width=3, height=4, )

        print('\tGenerated checkpoint snapshot.\n')

    return figures











'''
get_tuning_generalised
Caclulate tuning of each neuron to task variables (both individual variables and pairs of variables)
---------------------------------------------------------------------------------------------
Receives
    task :
        Task on which net was trained
    net :
        Trained o2s.net.RNN to be test
    inputs :
        Testing data inputs (convered to numpy)
    targets :
        Testing data targets (converted to numpy)
    vars :
        Dictionary of task variables
    activity :
        Testing data network rates (converted to numpy)
    tuning_vars_list :
        List of variable names to tune for (names must be captured in if-tree of function)

Returns
    tuning_vars (dict) :
        keys are variable names (from tuning vars list) and values are numpy arrays defining bins use for tuning
    tuning_dict (dict) :
        keys are prefixed with either variable names (from tuning vars list) or <variable name>_to_<variable name>
        keys are suffixed with _tuning_bins, tuning_bins_size, or tuning
        values are arrays of size [n_neurons, n_bins] for univariate tuning, or [n_neurons, n_bins (var 1), n_bins (var 2)]
        each element in an array represens the sum of activity (tuning_bins), number of occurences (tuning_bins_size), or 
        average activity (tuning) for a given neuron, at a given value of the target variable (or confluence of the two target variables)
'''
def get_tuning_generalised(task: o2s.task.Task, inputs: np.ndarray, targets: np.ndarray, vars: dict, activity: np.ndarray, tuning_vars_list: List[str]) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    init_duration = task.config.init_duration
    tuning_vars = {}

    # Restrict tuning to the period after initial transients
    activity, inputs, targets = activity[:,init_duration:,:], inputs[:,init_duration:,:], targets[:,init_duration:,:]
    for key, var in vars.items():
        if key != 'sx' and key != 'sy':
            vars[key] = var[:,init_duration:]

    # Extract tuning parameters
    n_angle_bins = 360#task.config.n_angle_bins
    n_AV_bins = 100#task.config.n_AV_bins
    n_AV_std = 3#task.config.n_AV_std
    n_position_bins = 100
    n_neurons = task.config.n_neurons
    n_trials = activity.shape[0]       
    n_timesteps = activity.shape[1]


    if 'AV' in tuning_vars_list:

        AV = vars['av']

        std_AV = np.std(AV)
        min_AV = np.min(AV)
        max_AV = np.max(AV)
        min_AV_included = max(-n_AV_std * std_AV, min_AV)
        max_AV_included = min(n_AV_std * std_AV, max_AV)

        AV_bins = np.linspace(min_AV_included, max_AV_included, n_AV_bins+1)[:-1]

        tuning_vars['AV'] = dict(var=AV, bins=AV_bins, title='Angular Velocity')

    if 'HD' in tuning_vars_list:

        HD = vars['hd'] * 180/np.pi

        angle_bins = np.linspace(0, 360, n_angle_bins+1)[:-1]

        tuning_vars['HD'] = dict(var=HD, bins=angle_bins, title='Head Direction')

    if 'ego_SD' in tuning_vars_list:
        
        ego_SD = vars['sd'] * 180/np.pi

        angle_bins = np.linspace(0, 360, n_angle_bins+1)[:-1]

        tuning_vars['ego_SD'] = dict(var=ego_SD, bins=angle_bins, title='Head-Shelter Angle')
    
    if 'allo_SD' in tuning_vars_list:

        HD = vars['hd'] * 180/np.pi
        ego_SD = vars['sd'] * 180/np.pi
        allo_SD = np.remainder(HD + ego_SD, 360)

        angle_bins = np.linspace(0, 360, n_angle_bins+1)[:-1]

        tuning_vars['allo_SD'] = dict(var=allo_SD, bins=angle_bins, title='Absolute Shelter Angle')

    if 'x' in tuning_vars_list:

        X = vars['x']

        position_bins = np.linspace(task.config.min_xy, task.config.max_xy, n_position_bins)

        tuning_vars['x'] = dict(var=X, bins=position_bins, title='X Position')

    if 'y' in tuning_vars_list:

        Y = vars['y']

        position_bins = np.linspace(task.config.min_xy, task.config.max_xy, n_position_bins)

        tuning_vars['y'] = dict(var=Y, bins=position_bins, title='Y Position')
    
    tuning_dict = {}


    # Create bins
    for i, i_key in enumerate(tuning_vars.keys()):
        for j, j_key in enumerate((list(tuning_vars.keys())[i:])):
            i_var, i_bins = tuning_vars[i_key]['var'], tuning_vars[i_key]['bins']
            j_var, j_bins = tuning_vars[j_key]['var'], tuning_vars[j_key]['bins']

            # Var tuning
            if j == 0:
                bins = np.zeros((n_neurons, len(i_bins)))
                bin_size = np.zeros((n_neurons, len(i_bins)))

                tuning_dict[f'{i_key}_tuning_bins'] = bins
                tuning_dict[f'{i_key}_tuning_bin_size'] = bin_size

            # Var-to-var tuning
            else:
                bins = np.zeros((n_neurons, len(i_bins), len(j_bins)))
                bin_size = np.zeros((n_neurons, len(i_bins), len(j_bins)))

                tuning_dict[f'{i_key}_to_{j_key}_tuning_bins'] = bins
                tuning_dict[f'{i_key}_to_{j_key}_tuning_bin_size'] = bin_size

    print('Computing tuning curves')

    # Fill bins
    for i, i_key in enumerate(tuning_vars.keys()):
        for j, j_key in enumerate((list(tuning_vars.keys())[i:])):
            i_var, i_bins = tuning_vars[i_key]['var'], tuning_vars[i_key]['bins']
            j_var, j_bins = tuning_vars[j_key]['var'], tuning_vars[j_key]['bins']

            # Var tuning
            if j == 0:
                bins = tuning_dict[f'{i_key}_tuning_bins']
                bin_size = tuning_dict[f'{i_key}_tuning_bin_size']

                for neuron in range(n_neurons):
                    for trial in range(n_trials):
                        i_bin_indices = (np.digitize(i_var[trial], i_bins)-1)
                        bins[neuron][i_bin_indices] += activity[trial, :, neuron]
                        bin_size[neuron][i_bin_indices] += 1

            # Var-to-var tuning
            else:
                bins = tuning_dict[f'{i_key}_to_{j_key}_tuning_bins']
                bin_size = tuning_dict[f'{i_key}_to_{j_key}_tuning_bin_size']

                tuning_key = f'{i_key}_to_{j_key}_tuning'
                tuning_dict[tuning_key] = np.zeros_like(bins)

                for neuron in range(n_neurons):
                    for trial in range(n_trials):
                        i_bin_indices = (np.digitize(i_var[trial], i_bins)-1)
                        j_bin_indices = (np.digitize(j_var[trial], j_bins)-1)
                        bins[neuron][i_bin_indices, j_bin_indices] += activity[trial, :, neuron]
                        bin_size[neuron][i_bin_indices, j_bin_indices] += 1

            print(f'\tCompleted {i_key}' + (f' and {j_key}' if j>0 else ''))

    # Average bins
    for i, i_key in enumerate(tuning_vars.keys()):
        for j, j_key in enumerate((list(tuning_vars.keys())[i:])):
            i_var, i_bins = tuning_vars[i_key]['var'], tuning_vars[i_key]['bins']
            j_var, j_bins = tuning_vars[j_key]['var'], tuning_vars[j_key]['bins']

            # Var tuning
            if j == 0:
                bins = tuning_dict[f'{i_key}_tuning_bins']
                bin_size = tuning_dict[f'{i_key}_tuning_bin_size']

                tuning_key = f'{i_key}_tuning'

            # Var-to-var tuning
            else:
                bins = tuning_dict[f'{i_key}_to_{j_key}_tuning_bins']
                bin_size = tuning_dict[f'{i_key}_to_{j_key}_tuning_bin_size']

                tuning_key = f'{i_key}_to_{j_key}_tuning'
            
            tuning_dict[tuning_key] = np.divide(bins, bin_size, out=np.zeros_like(bins), where=bin_size!=0)

    return tuning_vars, tuning_dict

'''
univar_tuning_plot
Plots the tuning of all neurons in the network to one variable (can stack multiple such tunings on top of each other)
---------------------------------------------------------------------------------------------
Receives
    task :
        Task on which net was trained
    tuning_vars :
        From get_tuning_generalised
    tuning_dict :
        From get_tuning_generalised
    vars :
        List of tuples: (task variable name to plot, HEX string for colour in which to plot it)
    ordering (optional) :
        Order of neuron indices in which to plot
    bin_mask (optional) :
        Mask for x-axis (i.e. range of task variable); boolean of same size as corresponding tuning_vars array

Returns
    matplotlib.figure.Figure :
        Generated figure
'''
def univar_tuning_plot(task: o2s.task.Task, tuning_vars: List[str], tuning_dict: Dict[str, np.ndarray], vars: List[Tuple[str, str]], ordering: List[int] = None, bin_mask: np.ndarray = None, **kwargs) -> matplotlib.figure.Figure:

    def _plot_angle_tuning(ax, neuron):
        for var, col in vars:
            bins = tuning_vars[var]['bins']
            tuning = tuning_dict[f'{var}_tuning']
            if bin_mask is not None:
                bins = bins[bin_mask]
                tuning = tuning[:,bin_mask]
            ax.scatter(bins, tuning[neuron], c=col, label=tuning_vars[var]['title'], s=1, zorder=0)

        ax.set_ylim([0,1])

    # Use neuron_by_neuron_plot template to create plot
    return neuron_by_neuron_plot(task,
                                 plot_closure=_plot_angle_tuning,
                                 x_label='Variable',
                                 y_label='Activity', 
                                 ordering=ordering, **kwargs)

'''
bivar_tuning_plot
Equivalent to univar_tuning_plot but for bivariate tuning
---------------------------------------------------------------------------------------------
Receives
    task :
        Task on which net was trained
    tuning_vars :
        From get_tuning_generalised
    tuning_dict :
        From get_tuning_generalised
    vars :
        Tuple of names of two variables to plot
    ordering (optional) :
        Order of neuron indices in which to plot
    x_mask (optional) :
        Mask for x-axis (i.e. range of task variable on x); boolean of same size as corresponding tuning_vars array
    x_mask (optional) :
        Mask for x-axis (i.e. range of task variable on x); boolean of same size as corresponding tuning_vars array

Returns
    matplotlib.figure.Figure :
        Generated figure
'''
def bivar_tuning_plot(task: o2s.task.Task, tuning_vars: List[str], tuning_dict: Dict[str, np.ndarray], vars: Tuple[str, str], ordering: List[int] = None, x_mask: np.ndarray = None, y_mask: np.ndarray = None, **kwargs) -> matplotlib.figure.Figure:
    assert len(vars)==2

    margin = kwargs.get('test_fig_margin', task.config.test_fig_margin)

    x_var, y_var = vars
    x_bins = tuning_vars[x_var]['bins']
    y_bins = tuning_vars[y_var]['bins']

    try:
        tuning_grid = tuning_dict[f'{x_var}_to_{y_var}_tuning']
    except KeyError:
        tuning_grid = tuning_dict[f'{y_var}_to_{x_var}_tuning'].transpose((0, 2, 1))

    if x_mask is not None:
        x_bins = x_bins[x_mask]
        tuning_grid = tuning_grid[:,x_mask,:]
    if y_mask is not None:
        y_bins = y_bins[y_mask]
        tuning_grid = tuning_grid[:,:,y_mask]
    
    def _plot_HD_AV_tuning(ax, neuron):
        ax.imshow(tuning_grid[neuron].T, cmap='turbo', label='Activity', aspect='auto', extent=[x_bins[0], x_bins[-1], y_bins[0], y_bins[-1]])

    def _make_legend(fig, ax):
        im_artist = ax[0,0].images[0]
        cbar_ax = fig.add_axes([0.75, margin/2, 1 - 0.75 - margin, margin/4])
        fig.colorbar(im_artist, cax=cbar_ax, orientation='horizontal')

    # Use neuron_by_neuron_plot template to create plot
    return neuron_by_neuron_plot(task,
                                 plot_closure=_plot_HD_AV_tuning,
                                 x_label=tuning_vars[x_var]['title'],
                                 y_label=tuning_vars[y_var]['title'], 
                                 legend_closure = _make_legend, 
                                 ordering=ordering, **kwargs)

'''
manifold_tuning_plot
Plots univariate tuning to a given variable on the PCA manifold
---------------------------------------------------------------------------------------------
Receives
    task :
        Task on which net was trained
    activity :
        Activity of network on testing data (converted to numpy)
    tuning_vars :
        From get_tuning_generalised
    tuning_dict :
        From get_tuning_generalised
    colour_var_namr :
        Name of variable to plot on manifold
    var_threshold (optional; default=0.9) :
        Cumulative variance threshold for which to consider principal components
    max_dim (optional; default=10) :
        Maximum number of principal components to consider
    cmap (optional; default='plasma') :
        Name of matplotlib colormap to use

Returns
    matplotlib.figure.Figure :
        Generated figure
'''
def manifold_plot(task: o2s.task.Task, activity: np.ndarray, tuning_vars: List[str], tuning_dict: Dict[str, np.ndarray], colour_var_name: str, var_threshold: float = 0.9, max_dim: int = 10, cmap: str = 'plasma'):
    n_neurons = activity.shape[2]
    config = task.config

    # Perform PCA on the activity data
    pca = PCA(n_components=max_dim)
    pca.fit(activity[:, config.init_duration:].reshape((-1, n_neurons)))

    # Determine the number of dimensions needed to explain the variance
    n_dimensions, = np.where(np.cumsum(pca.explained_variance_ratio_) > var_threshold)
    if len(n_dimensions) == 0:
        n_dimensions = max_dim
    else:
        n_dimensions = n_dimensions[0] + 1

    # Prepare color normalization and bins
    colour_bins = tuning_vars[colour_var_name]['bins']
    cmap = plt.get_cmap(cmap)
    norm = matplotlib.colors.Normalize(vmin=colour_bins[0], vmax=colour_bins[-1])

    # Create the figure and subfigures
    fig = plt.figure(figsize=(config.test_fig_width, 2.5 * config.test_fig_height))
    outer_gs = fig.add_gridspec(2, 1, hspace=0, height_ratios=[1,3])
    subfigs = [fig.add_subfigure(outer_gs[i,0]) for i in range(2)]

    # Subfigure 1: PCA explained variance
    var_fig = subfigs[0]
    var_fig_ax = var_fig.subplots()
    var_fig_ax.bar(np.arange(1, max_dim + 1), pca.explained_variance_ratio_, color='white')
    var_fig_ax.bar(np.arange(1, n_dimensions + 1), pca.explained_variance_ratio_[:n_dimensions], color=cmap(0.8))
    var_fig_ax.vlines(n_dimensions + 0.5, ymin=0, ymax=1, linestyle=':', color='gray')

    max_var = np.ceil(10 * pca.explained_variance_ratio_[0]) / 10
    var_fig_ax.set_ylim([0, max_var])
    var_fig_ax.set_xticks(np.arange(1, max_dim + 1))
    var_fig_ax.set_yticks([0, max_var])
    var_fig_ax.set_title('Principal Components Explained Variance', fontsize=20, fontweight='bold')

    # Subfigure 2: Manifold activity
    manifold_fig = subfigs[1]
    gs = manifold_fig.add_gridspec(nrows=n_dimensions+1, ncols=n_dimensions-1, hspace=0.5,
                                   height_ratios=([1] + [0.5 for _ in range(n_dimensions-1)] + [0.1]))

    var_activity = tuning_dict[f'{colour_var_name}_tuning']
    var_activity = pca.transform(var_activity.T)

    for i, pc_y in enumerate(range(1, n_dimensions)):
        for j, pc_x in enumerate(range(n_dimensions - 1)):
            if pc_x >= pc_y:
                continue

            ax = manifold_fig.add_subplot(gs[i, j])
            ax.scatter(var_activity[:, pc_x], var_activity[:, pc_y], c=colour_bins, cmap=cmap, norm=norm)

            ax.set_xticks([])
            ax.set_yticks([])

            if i == n_dimensions - 2:
                ax.set_xlabel(f'PC {pc_x + 1}')
            if j == 0:
                ax.set_ylabel(f'PC {pc_y + 1}')

    # Add color scale
    scale_ax = manifold_fig.add_subplot(gs[-1, 1:-1])
    gradient = np.vstack((np.linspace(0, 1, len(colour_bins)), np.linspace(0, 1, len(colour_bins))))
    scale_ax.imshow(gradient, aspect='auto', cmap=cmap, extent=[colour_bins[0], colour_bins[-1], 0, 1])
    scale_ax.set_yticks([])
    scale_ax.set_xticks([colour_bins[0], colour_bins[-1]])
    scale_ax.set_title(tuning_vars[colour_var_name]['title'], fontsize=20, fontweight='bold')

    # Title for manifold activity subfigure
    manifold_fig.suptitle(f'Manifold Activity', fontsize=25, fontweight='bold')

    fig.subplots_adjust(top=0.92)

    return fig
'''
path_integration_plot
Create figure with some examples of path integration ability
---------------------------------------------------------------------------------------------
Receives
    task :
        Task on which net was trained
    net :
        Trained o2s.net.RNN to be test
    targets :
        Testing data targets (converted to numpy)
    outputs :
        Testing data outputs (converted to numpy)
    true_colour :
        HEX string for colour of true path
    pred_colour :
        HEX string for colour of predicted path

Returns
    matplotlib.figure.Figure :
        Generated figure
'''
def path_integration_plot(task: o2s.task.Task, targets: np.ndarray, outputs: np.ndarray, true_colour: str = '#7776BC', pred_colour: str = '#CDC7E5') -> matplotlib.figure.Figure:

    t0 = task.config.init_duration

    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(task.config.test_fig_width,task.config.test_fig_height),
                           sharex=True, sharey=True)
    
    fig.suptitle('Path Integration', fontsize=20)

    example_indices = np.random.permutation(targets.shape[0])[:4].reshape((2,2))
    handles = {}

    for i in range(2):
        for j in range(2):
            k = example_indices[i,j]

            ax[i,j].set_xticks([0, 1])
            ax[i,j].set_yticks([0, 1])

            handles['true'], = ax[i,j].plot(targets[k, t0:, task.target_map['x']], targets[k, t0:, task.target_map['y']], color=true_colour, label='True Trajectory')
            handles['true_start'] = ax[i,j].scatter(targets[k, t0, task.target_map['x']], targets[k, t0, task.target_map['y']], color=true_colour, label='True Start', marker='o', s=100)
            handles['true_end'] = ax[i,j].scatter(targets[k, -1, task.target_map['x']], targets[k, -1, task.target_map['y']], color=true_colour, label='True End', marker='x', s=100)

            handles['pred'], = ax[i,j].plot(outputs[k, t0:, task.target_map['x']], outputs[k, t0:, task.target_map['y']], color=pred_colour, label='Model Trajectory')
            handles['pred_start'] = ax[i,j].scatter(outputs[k, t0, task.target_map['x']], outputs[k, t0, task.target_map['y']], color=pred_colour, label='Model Start', marker='o', s=100)
            handles['pred_end'] = ax[i,j].scatter(outputs[k, -1, task.target_map['x']], outputs[k, -1, task.target_map['y']], color=pred_colour, label='Model End', marker='x', s=100)

    fig.legend(loc='lower center', handles=[
        handles['true_start'], handles['pred_start'],
        handles['true'], handles['pred'],
        handles['true_end'], handles['pred_end']
    ])

    return fig

'''
loss_plot
Plot training and testing loss
---------------------------------------------------------------------------------------------
Receives
    task :
        Task on which net was trained
    test_losses :
        List of test losses generated in the course of training
    train_losses :
        List of train losses generate in the course of training

Returns
    matplotlib.figure.Figure :
        Generated figure
'''
def loss_plot(task: o2s.task.Task, test_losses: List[float], train_losses: List[float], **kwargs) -> matplotlib.figure.Figure:
    width = kwargs.get('test_fig_width', task.config.test_fig_width)
    height = kwargs.get('test_fig_height', task.config.test_fig_height)

    test_x = np.linspace(0, len(train_losses), len(test_losses))
    train_x = np.linspace(0, len(train_losses), len(train_losses))

    # Plot loss function value against weight update number
    fig, ax = plt.subplots(figsize=(width, height))

    ax.plot(test_x, test_losses, c='red', label='Testing Losses', zorder=10)
    ax.plot(train_x, train_losses, c='blue', label='Training Losses', zorder=1)

    ax.set_xlabel('Weight Update')
    ax.set_ylabel('Loss')
    ax.legend()

    return fig

'''
fit_examples_plot
Creates a figure containing examples of how well the given model predicts its targets
---------------------------------------------------------------------------------------------
Receives
    config :
        Configuration object corresponding to model
    targets :
        numpy.ndarray of targets corresponding to outputs (targets of test dataset)
    outputs :
        numpy.ndarray of o2s.net.RNN outputs corresponding to targets (result of passing test dataset through
        net)
    kwargs :
        kwargs which can override config parameters

Returns
    matplotlib.figure.Figure :
        Figure containing plot

'''
def fit_examples_plot(task: o2s.task.Task, targets: np.ndarray, outputs: np.ndarray, **kwargs) -> matplotlib.figure.Figure:
    
    # Get relevant config parameters

    width = kwargs.get('test_fig_width', task.config.test_fig_width)
    height = kwargs.get('test_fig_height', task.config.test_fig_height)
    margin = kwargs.get('test_fig_margin', task.config.test_fig_margin)

    n_vars = targets.shape[2]

    # Create plot - one row for each example sequence from test dataset
    fig, ax = plt.subplots(nrows=n_vars, figsize=(width, height), sharex=True)

    if n_vars == 1:
        ax = np.array([ax])

    example_i = np.random.permutation(targets.shape[0])[0]
    targets = targets[example_i]
    outputs = outputs[example_i]

    cmap = plt.get_cmap('Set3')#matplotlib.colormaps['Set3']

    for var, var_i in task.target_map.items():
        ax[var_i].plot(targets[:,var_i], c=cmap(var_i), label=var)
        ax[var_i].plot(outputs[:,var_i], c=cmap(var_i), linestyle='-.', label=f'model {var}')

        ax[var_i].vlines(task.config.init_duration, ymin=-1, ymax=1, color='gray', linewidth=3)

        ax[var_i].set_title(var, fontsize=18)

    # Aesthetic settings
    ax[-1].set_xlabel('Timestep')
    ax[-1].set_xticks([0, task.config.init_duration, targets.shape[0]])

    plt.subplots_adjust(left=margin, right=1-margin, top=1-margin, bottom=margin)

    return fig

'''
neuron_by_neuron_plot
Template function for creating a plot with subplots for every neuron
---------------------------------------------------------------------------------------------
Receives
    task :
        Task to which plot pertains
    plot_closure :
        Function which plots data on given neuron's axis
        Assumes call signature plot_closure(ax, i) where ax is axis for neuron at index i
    x_label (optional) :
        String to label x-axis with (unlabelled if not supplied)
    y_label (optional) :
        String to label y-axis with (unlabelled if not supplied)
    legend_closure (optional) :
        Function for creating legend based on axes
        Assumes call signature legend_closure(fig, ax) where fig and ax are the figure and
        axes of the current plot
    ordering (optional) :
        Order of indices by which to plot neurons
    kwargs :
        kwargs which can override config parameters

Returns
    matplotlib.figure.Figure :
        Figure containing plot

'''
def neuron_by_neuron_plot(task, plot_closure: Callable[[matplotlib.axes.Axes], int], x_label: str = None, y_label: str = None, legend_closure: Callable[[matplotlib.figure.Figure], matplotlib.axes.Axes] = None, ordering: List[int] = None, **kwargs) -> matplotlib.figure.Figure:

    # Get relevant config parameters

    width = kwargs.get('test_fig_width', task.config.test_fig_width)
    height = width
    margin = kwargs.get('test_fig_margin', task.config.test_fig_margin)

    # Create fig with config.n_neurons subplots, with a square arrangement
    n_rows = int(np.ceil(np.sqrt(task.config.n_neurons)))
    fig, ax = plt.subplots(nrows=n_rows, ncols=n_rows, figsize=(width, height), sharex=True, sharey=True)

    # Generate ordering if not supplied
    if ordering is None:
        ordering = np.arange(task.config.n_neurons, dtype=np.int32)

    # Plot on each subplot, going left-to-right, top-to-bottom
    for i in range(n_rows):
        for j in range(n_rows):
            
            # As there may be more subplots than neurons, turn off unused subplots
            if i*n_rows + j >= task.config.n_neurons:
                ax[i,j].set_axis_off()
                continue

            # Get the neuron index of this subplot (as defined by ordering)
            neuron = ordering[i*n_rows + j]

            # Plot on the subplot
            plot_closure(ax[i,j], neuron)

            ax[i,j].annotate(
                f'{neuron}',
                xy=(0, 1), xycoords='axes fraction',
                xytext=(+0.5, -0.5), textcoords='offset fontsize',
                fontsize='medium', verticalalignment='top', fontfamily='serif',
                bbox=dict(facecolor='0.7', edgecolor='none', pad=3.0))

    # Aesthetic settings
    if x_label is not None:
        fig.text(0.5, margin/4, x_label, ha='center', fontsize=18)
    if y_label is not None:
        fig.text(margin/4, 0.5, y_label, va='center', rotation='vertical', fontsize=18)

    if legend_closure is not None:
        legend_closure(fig, ax)
    else:
        handles, labels = ax[0,0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower right', ncol=len(handles), markerscale=10)

    plt.subplots_adjust(left=margin, right=1-margin, top=1-margin, bottom=margin)

    return fig




















############################################################################################################################################
########################################################### LEGACY FUNCTIONS ###############################################################
############################################################################################################################################


'''
classification_plot
Creates a figure identifying the different classifications of o2s.net.RNN units based on tuning profiles
(per Cueve et al., 2020)
---------------------------------------------------------------------------------------------
Receives
    config :
        Configuration object corresponding to model
    tuning_dict :
        Result of get_tuning_data
    class_dict :
        Result of classify_neurons
    kwargs :
        kwargs which can override config parameters

Returns
    matplotlib.figure.Figure :
        Figure containing plot

'''
def classification_plot(task, tuning_dict, class_dict, **kwargs):

    # Get relevant config parameters

    max_dif_for_untuned = kwargs.get('max_dif_for_untuned', task.config.max_dif_for_untuned)
    max_slope_for_untuned = kwargs.get('max_slope_for_untuned', task.config.max_slope_for_untuned)
    max_slope_for_compass = kwargs.get('max_slope_for_compass', task.config.max_slope_for_compass)
    min_dif_for_compass = kwargs.get('min_dif_for_compass', task.config.min_dif_for_compass)

    differential_target_HD_to_activity = tuning_dict['differential_target_HD_to_activity']
    slope_AV_to_activity = tuning_dict['AV_to_activity_linear_model'][:,0]

    width = kwargs.get('test_fig_width', task.config.test_fig_width)
    height = kwargs.get('test_fig_height', task.config.test_fig_height)
    margin = kwargs.get('test_fig_margin', task.config.test_fig_margin)

    # Create figure (only one plot)
    fig, ax = plt.subplots(figsize=(width, height))

    # Colours matches to ordering of classes (compass, positive shifters, negative shifters, weakly tuned, untuned)
    colours = ['red','blue','green','yellow', 'grey']

    # Plot each neuron on axes: 
    # y is slope of linear fit of angular velocity tuning profile
    # x is maximum differential of head-direction tuning profile (i.e. max - min activation over range of head-directions)
    # colour is classification based on these values (see classify_neurons)
    for i, (colour, classification) in enumerate(zip(colours, class_dict['ordered_names'])):    
        class_indices = class_dict['ordered_strat'][i]
        if len(class_indices) > 0:
            ax.scatter(differential_target_HD_to_activity[class_indices], slope_AV_to_activity[class_indices], c=colour, label=classification)
        
    # Plot bounding box for classification as compass
    ax.plot( [min_dif_for_compass, min_dif_for_compass], [-max_slope_for_compass, max_slope_for_compass], linestyle='--', color='pink', linewidth=3)
    ax.plot( [min_dif_for_compass, np.max(differential_target_HD_to_activity)], [-max_slope_for_compass, -max_slope_for_compass], linestyle='--', color='pink', linewidth=3)
    ax.plot( [min_dif_for_compass, np.max(differential_target_HD_to_activity)], [max_slope_for_compass, max_slope_for_compass], linestyle='--', color='pink', linewidth=3)
    
    # Plot bounding box for classification as untuned
    ax.plot( [max_dif_for_untuned, max_dif_for_untuned], [-max_slope_for_untuned, max_slope_for_untuned], linestyle='--', color='lightgrey', linewidth=3)
    ax.plot( [max_dif_for_untuned, 0], [-max_slope_for_untuned, -max_slope_for_untuned], linestyle='--', color='lightgrey', linewidth=3)
    ax.plot( [max_dif_for_untuned, 0], [max_slope_for_untuned, max_slope_for_untuned], linestyle='--', color='lightgrey', linewidth=3)


    # Aesthetic settings
    fig.suptitle('Number of compass units: {} | Number of pos shift units: {} | Number of neg shift units: {} | \n Number of weakly tuned units: {} | Number of unresponsive units: {}'.format(
        len(class_dict['compass']), len(class_dict['pos_shift']), len(class_dict['neg_shift']), len(class_dict['weakly_tuned']), len(class_dict['untuned'])
    ), fontsize=8, y=0.1)

    fig.text(0.5, margin/4, 'Strength of Input Angle-to-Activity Tuning', ha='center')
    fig.text(margin/4, 0.5, 'Slope of Angular Velocity-to-Activity Tuning', va='center', rotation='vertical')

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower right', ncol=len(handles))

    plt.subplots_adjust(left=margin, right=1-margin, top=1-margin, bottom=margin)

    return fig






'''
HD_AV_tuning_plot
Creates a plot of HD-AV tuning profile of all neurons
---------------------------------------------------------------------------------------------
Receives
    config :
        Configuration object corresponding to model
    HD_to_AV_grid_masked :
        Output of get_tuning_data (either get_tuning_data(...)['target_HD_to_AV_grid_masked'] or
        get_tuning_data(...)['model_HD_to_AV_grid_masked])
    ordering (optional) :
        1D array of length config.n_neurons, which gives the indices of all neurons in the
        order they should be plotted (usually classify_neurons(...)['ordered_flat'])
        If not supplied, order is [0, 1, ..., config.n_neurons-1]
    kwargs :
        kwargs which can override config parameters

Returns
    matplotlib.figure.Figure :
        Figure containing plot

'''
def HD_AV_tuning_plot(task, HD_to_AV_grid_masked, ordering=None, **kwargs):

    # Get relevant config parameters

    margin = kwargs.get('test_fig_margin', task.config.test_fig_margin)
    
    # Define plot_closure to be used with neuron_by_neuron_plot
    # y is the range of angular velocities covered by config.n_AV_std, binned according to config.n_AV_bins
    # x is the range of head-directions, binned according to config.n_angle_bins
    # colour is the average activation of the neuron in that x-y bin across the test dataset
    def _plot_HD_AV_tuning(ax, neuron):
        ax.imshow(HD_to_AV_grid_masked[neuron].T, cmap='plasma', label='Activity', aspect='auto')

    # Define a legend_closure to be used with neuron_by_neuron_plot
    # Creates a horizontal colourbar in the corner
    def _make_legend(fig, ax):
        im_artist = ax[0,0].images[0]
        cbar_ax = fig.add_axes([0.75, margin/2, 1 - 0.75 - margin, margin/3])
        fig.colorbar(im_artist, cax=cbar_ax, orientation='horizontal')

    # Use neuron_by_neuron_plot template to create plot
    return neuron_by_neuron_plot(task,
                                 plot_closure=_plot_HD_AV_tuning,
                                 x_label='Head Direction',
                                 y_label='Angular Velocity', 
                                 legend_closure = _make_legend, 
                                 ordering=ordering, **kwargs)


'''
HD_tuning_plot
Creates a plot of HD tuning profile of all neurons (with option to overlay HD direction based 
on either target or predicted head-direction in same plots)
---------------------------------------------------------------------------------------------
Receives
    config :
        Configuration object corresponding to model
    angle_bins :
        Output of get_tuning_data (get_tuning_data(...)['angle_bins'])
    primary_HD_to_activity_grid :
        Output of get_tuning_data (either get_tuning_data(...)['target_HD_to_activity_grid'] or
        get_tuning_data(...)['model_HD_to_activity_grid'])
        To be plotted in red
    secondary_HD_to_activity_grid (optional) :
        Same as primary_HD_to_activity_grid, but to be overlayed in orange
    primary_label (optional) :
        Legend label to associate with primary tuning profile's red
    secondary_label (optional) :
        Legend label to associate with secondary tuning profile's orange
    ordering (optional) :
        1D array of length config.n_neurons, which gives the indices of all neurons in the
        order they should be plotted (usually classify_neurons(...)['ordered_flat'])
        If not supplied, order is [0, 1, ..., config.n_neurons-1]
    kwargs :
        kwargs which can override config parameters

Returns
    matplotlib.figure.Figure :
        Figure containing plot

'''
def HD_tuning_plot(task, tuning_dict, ordering=None, **kwargs):

    angle_bins = tuning_dict['angle_bins']
    primary_HD_to_activity_grid = tuning_dict['target_HD_to_activity_grid']
    secondary_HD_to_activity_grid = tuning_dict['model_HD_to_activity_grid']
    
    # Define plot_closure to be used with neuron_by_neuron_plot
    # x is the range of head-directions, binned according to config.n_angle_bins
    # y is the average activity of that neuron in that head-direction bin across the test dataset
    # colour is red/orange depending on whether activity is from the primary or secondary tuning
    # profile (usually profiles based on target and predicted head-direction, respectively)
    def _plot_HD_tuning(ax, neuron):
        ax.plot(angle_bins, primary_HD_to_activity_grid[neuron], c='red', label='Target')

        if secondary_HD_to_activity_grid is not None:
            ax.plot(angle_bins, secondary_HD_to_activity_grid[neuron], c='orange', label='Model')

    # Use neuron_by_neuron_plot template to create plot
    return neuron_by_neuron_plot(task,
                                 plot_closure=_plot_HD_tuning,
                                 x_label='Head Direction',
                                 y_label='Activity', 
                                 ordering=ordering, **kwargs)



'''
AV_tuning_plot
Creates a plot of AV tuning profile of all neurons (with option to plot linear fit of profile)
---------------------------------------------------------------------------------------------
Receives
    config :
        Configuration object corresponding to model
    AV_bins :
        Output of get_tuning_data (get_tuning_data(...)['AV_bins'])
    AV_to_activity_grid :
        Output of get_tuning_data (get_tuning_data(...)['AV_to_activity_grid]')
    AV_to_activity_linear_model (optional) :
        Output of get_tuning_data (get_tuning_data(...)['AV_to_activity_linear_model'])
        If not provided, no linear model will be plotted
    AV_bin_mask (optional) :
        Outout of get_tuning_data (get_tuning_data(...)['AV_bin_mask'])
        Assumed provided if AV_to_activity_linear_model is provided
    ordering (optional) :
        1D array of length config.n_neurons, which gives the indices of all neurons in the
        order they should be plotted (usually classify_neurons(...)['ordered_flat'])
        If not supplied, order is [0, 1, ..., config.n_neurons-1]
    kwargs :
        kwargs which can override config parameters

Returns
    matplotlib.figure.Figure :
        Figure containing plot

'''
def AV_tuning_plot(task, tuning_dict, ordering=None, **kwargs):

    AV_bins = tuning_dict['AV_bins']
    AV_bin_mask = tuning_dict['AV_bin_mask']
    AV_to_activity_grid = tuning_dict['AV_to_activity_grid']
    AV_to_activity_linear_model = tuning_dict['AV_to_activity_linear_model']

    # Define plot_closure to be used with neuron_by_neuron_plot
    # x is the range of angular velocities, binned according to config.n_AV_bins
    # y is the average activity of that neuron in that angular velocity bin across the test dataset
    def _plot_tuning_and_model(ax, neuron):
        # Plot activity of neuron across entire range of angular velocities
        ax.plot(AV_bins[AV_bin_mask], AV_to_activity_grid[neuron][AV_bin_mask], c='black', label='Activity')

        # If linear models are provided, plot the corresponding line in red
        # This line only spans a subset of the angular velocity range (bins specified by AV_bin_mask)
        if AV_to_activity_linear_model is not None:
            if not np.isnan(AV_to_activity_linear_model[neuron][0]):
                x = AV_bins[AV_bin_mask]
                y = x * AV_to_activity_linear_model[neuron][0] + AV_to_activity_linear_model[neuron][1]
                ax.plot(x, y, c='red', label='Linear Fit at Preferred Angle', linewidth=3)
        
        ax.set_ylim([0, 1])

    # Use neuron_by_neuron_plot template to create plot
    return neuron_by_neuron_plot(task,
                                 plot_closure=_plot_tuning_and_model,
                                 x_label='Angular Velocity',
                                 y_label='Activity', 
                                 ordering=ordering, **kwargs)






'''
connectivity_plot
Creates a plot with recurrent weight matrix broken into 9 segments corresponding to inter- and
intra-class connections
---------------------------------------------------------------------------------------------
Receives
    config :
        Configuration object corresponding to model
    net :
        o2s.net.RNN object
    class_dict :
        Output of classify_neurons
    kwargs :
        kwargs which can override config parameters

Returns
    matplotlib.figure.Figure :
        Figure containing plot

'''
def connectivity_plot(task, net, class_dict=None, **kwargs):
    
    # Get relevant config parameters

    width = kwargs.get('test_fig_width', task.config.test_fig_width)
    margin = kwargs.get('test_fig_margin', task.config.test_fig_margin)

    # Get net's recurrent weight matrix
    W_rec = net.W_rec.weight.detach().numpy()

    fig, ax = plt.subplots(figsize=(width, width))

    if class_dict is not None:
        ordering = np.concatenate(class_dict['ordered_strat'][:3])
        W_rec = W_rec[ordering][:,ordering]
    
    ax.imshow(W_rec, cmap='seismic', aspect='auto', vmax=np.max(np.abs(W_rec)), vmin=-np.max(np.abs(W_rec)))

    ax.set_xticks([])
    ax.set_yticks([])

    if class_dict is not None:
        n_compass, n_pos_shift, n_neg_shift = len(class_dict['compass']), len(class_dict['pos_shift']), len(class_dict['neg_shift'])
        breaks = [n_compass-0.5,
                n_compass+n_pos_shift-0.5]
        n_neurons = len(ordering)
        ax.vlines(x=breaks, ymin=-0.5, ymax=n_neurons-0.5, colors='k')
        ax.hlines(y=breaks, xmin=-0.5, xmax=n_neurons-0.5, colors='k')

        text_pos = [0 + n_compass/2,
                    breaks[0] + n_pos_shift/2,
                    breaks[1] + n_neg_shift/2]
        ax.text(x=text_pos[0], y=n_neurons+1, s='Compass', fontsize=15, fontweight='bold', horizontalalignment='center')
        ax.text(x=text_pos[1], y=n_neurons+1, s='Positive Shifters', fontsize=15, fontweight='bold', horizontalalignment='center')
        ax.text(x=text_pos[2], y=n_neurons+1, s='Negative Shifters', fontsize=15, fontweight='bold', horizontalalignment='center')

        ax.text(x=-2, y=text_pos[0], s='Compass', fontsize=15, fontweight='bold', verticalalignment='center', rotation='vertical')
        ax.text(x=-2, y=text_pos[1], s='Positive Shifters', fontsize=15, fontweight='bold', verticalalignment='center', rotation='vertical')
        ax.text(x=-2, y=text_pos[2], s='Negative Shifters', fontsize=15, fontweight='bold', verticalalignment='center', rotation='vertical')

    # Aesthetic settings
    im_artist = ax.images[0]
    cbar_ax = fig.add_axes([0.75, margin/4, 1 - 0.75 - margin, margin/4])
    fig.colorbar(im_artist, cax=cbar_ax, orientation='horizontal')

    plt.subplots_adjust(left=margin, right=1-margin, top=1-margin, bottom=margin)

    return fig


'''
lesion_plot
Creates a plot showing the effects of lesioning connections between different neuron classes
---------------------------------------------------------------------------------------------
Receives
    config :
        Configuration object corresponding to model
    net :
        o2s.net.RNN object
    class_dict :
        Output of classify_neurons
    kwargs :
        kwargs which can override config parameters

Returns
    matplotlib.figure.Figure :
        Figure containing plot

'''
def lesion_plot(task, net, class_dict, **kwargs):

    # Get relevant config parameters

    n_lesion_timesteps = kwargs.get('n_lesion_timesteps', task.config.n_lesion_timesteps)
    n_lesion_transient = kwargs.get('n_lesion_transient', task.config.n_lesion_transient)
    av_step_std = kwargs.get('av_step_std', task.config.av_step_std)

    width = kwargs.get('test_fig_width', task.config.test_fig_width)
    height = kwargs.get('test_fig_height', task.config.test_fig_height)*1.2
    margin = kwargs.get('test_fig_margin', task.config.test_fig_margin)

    # Lesion experiment is as follows:
    # Consider three types of input: constant zero angular velocity, constant counter-clockwise (positive) rotation, 
    #   and constant clockwise (negative) rotation
    # The o2s.net.RNN is subjected to these three inputs multiple times, with each time involving a different 'lesion'
    # Here, a lesion is the setting of all projections from one class of neurons onto the other (i.e. columns 
    #   in the weight matrix corresponding to the one, at rows corresponding to the other) to zero
    # The plot is then of both the predicted head-direction under this lesion, and the activity of the network


    # Define middle four-fifths of sequences for lesion
    lesion_start, lesion_end = n_lesion_transient + n_lesion_timesteps//5, n_lesion_transient + n_lesion_timesteps - (n_lesion_timesteps//5)

    # Define three types of input sequences
    zero_rotation = torch.zeros((n_lesion_transient+n_lesion_timesteps,))
    pos_rotation = torch.cat((
        zero_rotation[:lesion_start], torch.ones((lesion_end-lesion_start,)), zero_rotation[lesion_end:]
    )) * av_step_std * 2
    neg_rotation = pos_rotation.neg()

    # Initial head-direction is zero for all input sequences (so cosine is one, sine is zero)
    init_cos_angle = torch.cat((torch.ones((3, n_lesion_transient)), torch.zeros(3, n_lesion_timesteps)), axis=1)
    init_sin_angle = torch.zeros((3, n_lesion_transient + n_lesion_timesteps))
    input_av = torch.stack((zero_rotation, pos_rotation, neg_rotation))

    # Create one batch-like input tensor of all sequences
    inputs = torch.stack((input_av, init_sin_angle, init_cos_angle), axis=2).to(task.config.device)
    # Partition input into pre-, peri-, and post-lesion input
    pre_lesion_input, peri_lesion_input, post_lesion_input = inputs[:,:lesion_start,:], inputs[:,lesion_start:lesion_end,:], inputs[:,lesion_end:,:]

    # Define target as usual for these inputs
    target_angle = input_av.cumsum(axis=1)[:, n_lesion_transient:]
    target_sin, target_cos = torch.sin(target_angle), torch.cos(target_angle)
    target_angle = np.arctan2(target_sin, target_cos) * 180 / np.pi

    # Ensure no gradient tracking is being done for the lesino procedure
    with torch.no_grad():
        # Save recurrent weight matrix
        W_original = net.W_rec.weight

        # No-lesion control pass
        _,no_lesion_rates,no_lesion_outputs = net(inputs)


        # Define order and domain of lesions, and initialise lists to track results
        lesions = [
            class_dict['compass'], class_dict['pos_shift'], class_dict['neg_shift'], np.concatenate((class_dict['pos_shift'], class_dict['neg_shift']))
        ]
        all_lesion_rates, all_lesion_outputs = [no_lesion_rates], [no_lesion_outputs]

        # For each lesion...
        for lesion in lesions:
            # ...create a weight matrix according to the lesion domain...
            W_lesion = W_original.detach().clone()
            W_lesion[:,lesion] = 0

            # ...compute initial period without lesion...
            net.W_rec.weight = torch.nn.Parameter(W_original)
            _, pre_lesion_rates, pre_lesion_outputs = net(pre_lesion_input)

            # ...then a middle period with lesion...
            net.W_rec.weight = torch.nn.Parameter(W_lesion)
            _, peri_lesion_rates, peri_lesion_outputs = net(peri_lesion_input)

            # ...and a final period without lesion again
            net.W_rec.weight = torch.nn.Parameter(W_original)
            _, post_lesion_rates, post_lesion_outputs = net(post_lesion_input)

            # Concatenate each period into one tensor (for each of the net's rates and outputs) whose dim 1 length
            # is the number of timesteps in the whole trial (i.e. config.n_lesion_transient + config.n_lesion_timesteps)
            # and save
            rates = torch.cat((pre_lesion_rates, peri_lesion_rates, post_lesion_rates), dim=1)
            outputs = torch.cat((pre_lesion_outputs, peri_lesion_outputs, post_lesion_outputs), dim=1)
            all_lesion_rates.append(rates)
            all_lesion_outputs.append(outputs)
             
        # Copy recurrent weight matrix back into the net
        net.W_rec.weight = torch.nn.Parameter(W_original, requires_grad=True)

    
    # Create a plot for results of the experiment with 5 rows and 6 columns
    # Rows correspond to lesion domain: none, compass units, positive shifters, negative shifters, and all shifters
    # Columns are paired, and correspond to input type: constant zero angular velocity outputs/rates, constant counter-
    #   clockwise rotation outputs/rates, and constant clockwise rotation outputs/rates
    fig, ax = plt.subplots(nrows=5, ncols=6, figsize=(width, height), sharex=True)
    lesion_names = ['No Lesion', 'Compasses to All', 'Positive Shifterers to All', 'Negative Shifters to All', 'Both Shifters to All']
    
    for i, (lesion_rates, lesion_outputs) in enumerate(zip(all_lesion_rates, all_lesion_outputs)):
        for j in range(6):

            # On even columns, plot the target and predicted head-direction of the network
            if j%2 == 0:
                # Copy lesion outputs to cpu, and calculate the head-direction they predict
                lesion_outputs = lesion_outputs.detach().cpu()
                output_angle = np.arctan2(lesion_outputs[:, n_lesion_transient:, 0], lesion_outputs[:, n_lesion_transient:, 1]) * 180 / np.pi

                # Get input type for this column
                target_y, output_y = target_angle[j%3], output_angle[j%3]
                target_x, output_x = np.arange(len(target_y)), np.arange(len(output_y))

                # Plot head-direction over time (scatter plot used here to avoid ugliness that occurs when
                # head direction wraps beyond 180 degrees to the other side of the plot)
                ax[i,j].scatter(target_x, target_y, s=1, color='red', label='Input Angle')
                ax[i,j].scatter(output_x, output_y, s=1, color='orange', label='Output Angle')

                ax[i,j].set_ylim([-180, 180])

            # On odd columns, plot the network's activity over time
            else:
                # Copy the lesion activities to cpu
                lesion_rates = lesion_rates.detach().cpu()
                # Select the activities corresponding to this column's input type, and order by unit type
                ordered_rates = lesion_rates[j%3-1, n_lesion_transient:, class_dict['ordered_flat']]

                # Plot as matrix, where vertical rows are activities of an individual unit over time
                ax[i,j].imshow(ordered_rates.T, cmap='plasma', aspect='auto')

            # Label left edges of rows with lesion type (iput type for columns is evident from target head-direction)
            if j == 0:
                ax[i,j].set_ylabel(lesion_names[i])

            # For all axes, plot vertical lines corresponding to lesion
            ax[i,j].axvline(x=lesion_start-n_lesion_transient, c='gray')
            ax[i,j].axvline(x=lesion_end-n_lesion_transient, c='gray')

    # Aesthetic settings
    plt.subplots_adjust(left=margin, right=1-margin, top=1-margin, bottom=margin)

    return fig







'''
create_snapshot_image
Creates an image which contains all plots in one
---------------------------------------------------------------------------------------------
Receives
    config :
        Configuration object corresponding to model
    figs :
        Dict of figures created in the course of test
    checkpoint_dir_identity :
        String containing the path to the directory where the current checkpoint is (where the
        plots will be saved)

Returns
    None (image saved directly to checkpoint_dir_identity)

'''
def create_snapshot_image(config, figs, checkpoint_dir_identity, figure_pastes, width, height, dpi=500):

    output_image = Image.new("RGB", (width*dpi, height*dpi), (255,255,255))

    # Paste each plot in its defined location, at its defined size
    for name, (w, h, x, y) in figure_pastes.items():
        if name in figs:
            fig_image = Image.open(f'{checkpoint_dir_identity}/{name}.png')
            fig_image = fig_image.resize((w*dpi, h*dpi))
            output_image.paste(fig_image, (int(x*dpi), int(y*dpi)))

    # Save image to checkpoint directory
    output_image.save(f'{checkpoint_dir_identity}/snapshot.png')











'''
get_tuning_data
Calculates the tuning profiles of all neurons in the network
---------------------------------------------------------------------------------------------
Receives
    config :
        Configuration object corresponding to model
    inputs :
        numpy.ndarray of testing dataset inputs
    outputs :
        numpy.ndarray of network's outputs for the testing datset
    activity :
        numpy.ndarray of network's activity for the testing dataset 
    kwargs :
        kwargs which can override config parameters

Returns
    dict { str : numpy.ndarray } :
        'angle_bins' : 
            1D array of bins for head direction (of length config.n_angle_bins; 
            i.e., discretised range of head-direction)
        'AV_bins' : 
            1D array of bins for angular velocity (of length config.n_AV_bins;
            i.e. discretised range of angular velocity)
        'AV_bin_mask' : 
            1D boolean array which masks AV_bins, where a bin is kept if it is wihtin
            config.n_AV_std of 0
        'target_HD_to_AV_grid' :
            3D array of shape (config.n_neurons, config.n_angle_bins, config.n_AV_bins)
            Element at index [i, j, k] gives average activity of ith neuron across test
            dataset, when TRUE head-direction was in the jth bin of angle_bins, and 
            angular velocity was in the kth bin of AV_bins
        'model_HD_to_AV_grid' :
            3D array of shape (config.n_neurons, config.n_angle_bins, config.n_AV_bins)
            Element at index [i, j, k] gives average activity of ith neuron across test
            dataset, when PREDICTED head direction was in the jth bin of angle_bins, and 
            angular velocity was in the kth bin of AV_bins
        'target_HD_to_AV_grid_masked' :
            target_HD_to_AV_grid, with angular velocity dimension masked by AV_bin_mask
        'model_HD_to_AV_grid_masked' :
            model_HD_to_AV_grid, with angular velocity dimension masked by AV_bin_mask
        'target_HD_to_activity_grid' :
            2D array of shape (config.n_neurons, config.n_angle_bins)
            Element at index [i, j] gives average activity of the ith neuron across test
            dataset, when TRUE head-direction was in the jth bin of angle_bins
        'model_HD_to_activity_grid' :
            2D array of shape (config.n_neurons, config.n_angle_bins)
            Element at index [i, j] gives average activity of the ith neuron across test
            dataset, when PREDICTED head-direction was in the jth bin of angle_bins
        'AV_to_activity_grid' :
            2D array of shape (config.n_neurons, config.n_angle_bins)
            Element at index [i, j] gives average activity of the ith neuron across test
            dataset, when angular velocity was in the jth bin of AV_bins
        'AV_to_activity_linear_model' :
            2D array of shape (config.n_neurons, 2)
            Element at index [i, j] gives the coefficients of the linear regression of
            the ith neuron's AV tuning profile, with j=0 giving the slope and j=1 giving the
            intercept
        'target_preferred_angle' :
            1D array of length config.n_neurons
            Element at index [i] is the preferred TRUE head-direction of the ith neuron (i.e. 
            TRUE head-direction where absolute deviation from mean of average activity is maximised)
        'model_preferred_angle' :
            1D array of length config.n_neurons
            Element at index [i] is the preferred PREDICTED head-direction of the ith neuron 
            (i.e. PREDICTED head-direction where absolute deviation from mean average activity is 
            maximised)
        'differential_target_HD_to_activity' : 
            1D array of length config.n_neurons
            Element at index [i] is the difference between the maximum average activity of the
            ith neuron across the range of TRUE head directions, minus the minimum average activity
            across the range (i.e. the range of its HD-tuning profile)

'''
def get_tuning_data(task, inputs, targets, outputs, activity, **kwargs):

    # Get relevant config parameters

    n_angle_bins = kwargs.get('n_angle_bins', task.config.n_angle_bins)
    n_AV_bins = kwargs.get('n_AV_bins', task.config.n_AV_bins)
    n_AV_std = kwargs.get('n_AV_std', task.config.n_AV_std)
    n_neurons = task.config.n_neurons
    n_trials = activity.shape[0]        # (not from config, as may vary by training/testing)
    n_timesteps = activity.shape[1]

    # Calculate target and predicted head-direction (in range of 0-360 degrees)
    target_angle = np.arctan2(targets[:, :, 0], targets[:, :, 1]) * 180 / np.pi
    target_angle[np.where(target_angle < 0)] += 360

    model_angle = np.arctan2(outputs[:, :, 0], outputs[:, :, 1]) * 180 / np.pi
    model_angle[np.where(model_angle < 0)] += 360

    # Define head-direction bins simply across range 0-360 degrees
    angle_bins = np.linspace(0, 360, n_angle_bins+1)[:-1]

    # Get angular velocity
    AV = inputs[:, :, 0] * 180 / np.pi
    
    # Define angular velocity bins across range of obsereved angular velocity
    std_AV = np.std(AV)
    min_AV = np.min(AV)
    max_AV = np.max(AV)
    AV_bins = np.linspace(min_AV, max_AV, n_AV_bins+1)[:-1]
    # Create mask for angular velocity bins which captures config.n_AV_std standard deviations
    # of observed angular velocity either side of 0
    min_AV_included = max(-n_AV_std * std_AV, min_AV)
    max_AV_included = min(n_AV_std * std_AV, max_AV)
    AV_bin_mask = (min_AV_included <= AV_bins) & (AV_bins <= max_AV_included)
    
    # Initialise return arrays
    target_HD_to_AV_grid = np.zeros((n_neurons, len(angle_bins), len(AV_bins)))
    target_HD_to_AV_bin_size = np.zeros((n_neurons, len(angle_bins), len(AV_bins)))
    model_HD_to_AV_grid = np.zeros((n_neurons, len(angle_bins), len(AV_bins)))
    model_HD_to_AV_bin_size = np.zeros((n_neurons, len(angle_bins), len(AV_bins)))

    target_HD_to_activity_grid = np.zeros((n_neurons, len(angle_bins)))
    target_HD_to_activity_bin_size = np.zeros((n_neurons, len(angle_bins)))
    model_HD_to_activity_grid = np.zeros((n_neurons, len(angle_bins)))
    model_HD_to_activity_bin_size = np.zeros((n_neurons, len(angle_bins)))

    AV_to_activity_grid = np.zeros((n_neurons, len(AV_bins)))
    AV_to_activity_bin_size = np.zeros((n_neurons, len(AV_bins)))
    AV_to_activity_linear_model_coefficients = np.zeros((n_neurons, 2))

    target_preferred_angle = np.zeros((n_neurons,))
    model_preferred_angle = np.zeros((n_neurons,))

    for neuron in range(n_neurons):

        for trial in range(n_trials):
            
            # For each neuron in each sequence of the testing dataset
            # bin the trial's target and predicted head-directions, and angular velocity
            target_angle_bin_indices = np.digitize(target_angle[trial], angle_bins)-1
            model_angle_bin_indices = np.digitize(model_angle[trial], angle_bins)-1
            AV_bin_indices = np.digitize(AV[trial], AV_bins)-1

            target_HD_to_AV_grid[neuron][target_angle_bin_indices, AV_bin_indices] += activity[trial, :, neuron]
            # Keep a tally of number of trials contributing to each bin
            target_HD_to_AV_bin_size[neuron][target_angle_bin_indices, AV_bin_indices] += 1

            model_HD_to_AV_grid[neuron][model_angle_bin_indices, AV_bin_indices] += activity[trial, :, neuron]
            model_HD_to_AV_bin_size[neuron][model_angle_bin_indices, AV_bin_indices] += 1

            # Similarly for head-direction bins...
            target_HD_to_activity_grid[neuron][target_angle_bin_indices] += activity[trial, :, neuron]
            target_HD_to_activity_bin_size[neuron][target_angle_bin_indices] += 1
            model_HD_to_activity_grid[neuron][model_angle_bin_indices] += activity[trial, :, neuron]
            model_HD_to_activity_bin_size[neuron][model_angle_bin_indices] += 1
            # ...and angular-velocity bins
            AV_to_activity_grid[neuron][AV_bin_indices] += activity[trial, :, neuron]
            AV_to_activity_bin_size[neuron][AV_bin_indices] += 1

    # Find mean HD-AV activity for all neurons at once
    target_HD_to_AV_grid = np.divide(target_HD_to_AV_grid, target_HD_to_AV_bin_size, out=np.zeros_like(target_HD_to_AV_grid), where=target_HD_to_AV_bin_size!=0)
    model_HD_to_AV_grid = np.divide(model_HD_to_AV_grid, model_HD_to_AV_bin_size, out=np.zeros_like(model_HD_to_AV_grid), where=model_HD_to_AV_bin_size!=0)
    # Save copy of HD-AV activity under AV mask
    target_HD_to_AV_grid_masked = target_HD_to_AV_grid[:,:,AV_bin_mask]
    model_HD_to_AV_grid_masked = model_HD_to_AV_grid[:,:,AV_bin_mask]
    # Find mean HD activity for all neurons at once
    target_HD_to_activity_grid = np.divide(target_HD_to_activity_grid, target_HD_to_activity_bin_size, out=np.zeros_like(target_HD_to_activity_grid), where=target_HD_to_activity_bin_size!=0)
    model_HD_to_activity_grid = np.divide(model_HD_to_activity_grid, model_HD_to_activity_bin_size, out=np.zeros_like(model_HD_to_activity_grid), where=model_HD_to_activity_bin_size!=0)
    # Find mean AV activity for all neurons at once
    AV_to_activity_grid = np.divide(AV_to_activity_grid, AV_to_activity_bin_size, out=np.zeros_like(AV_to_activity_grid), where=AV_to_activity_bin_size != 0)

    # Find target preferred angle for each neuron as:
    # target_preferred_angle = angle_bins[np.argmax(                                                                       # maximum...
    #     np.abs(                                                                                                          # ...absolute...
    #         target_HD_to_activity_grid -                                                                                 # ...deviation....
    #         np.tile(np.mean(target_HD_to_activity_grid, axis=1).reshape((config.n_neurons,1)), (1,angle_bins.shape[0]))  # ...from the mean
    #         ), axis=1)]
    target_preferred_angle = angle_bins[np.argmax(target_HD_to_activity_grid, axis=1)]
    
    # model_preferred_angle = angle_bins[np.argmax(                                                             
    #     np.abs(                                                                                                       
    #         model_HD_to_activity_grid -                                                                                
    #         np.tile(np.mean(model_HD_to_activity_grid, axis=1).reshape((config.n_neurons,1)), (1,angle_bins.shape[0])) 
    #         ), axis=1)]
    model_preferred_angle = angle_bins[np.argmax(model_HD_to_activity_grid, axis=1)]
    
    for neuron in range(n_neurons):
        # Compute linear regression of neuron's activity onto angular velocity
        preferred_angle = int(model_preferred_angle[neuron])
        x = AV_bins
        y = model_HD_to_AV_grid[neuron][preferred_angle]
        # Restrict model to masked domain
        linear_model_mask = ~np.isnan(y) & AV_bin_mask
        x = x[linear_model_mask]
        y = y[linear_model_mask]

        # Calculate and save coefficients of model
        if len(x) > 1:
            AV_to_activity_linear_model_coefficients[neuron] = np.polyfit(x, y, 1)
        # Save NaNs if model fails
        else:
            AV_to_activity_linear_model_coefficients[neuron] = [np.nan, np.nan]

    # Find range of HD-tuning for each neuron
    differential_target_HD_to_activity = np.max(model_HD_to_activity_grid, axis=1) - np.min(model_HD_to_activity_grid, axis=1)

    return {
        'angle_bins': angle_bins,
        'AV_bins': AV_bins,
        'AV_bin_mask': AV_bin_mask,
        'target_HD_to_AV_grid': target_HD_to_AV_grid,
        'model_HD_to_AV_grid': model_HD_to_AV_grid,
        'target_HD_to_AV_grid_masked': target_HD_to_AV_grid_masked,
        'model_HD_to_AV_grid_masked': model_HD_to_AV_grid_masked,
        'target_HD_to_activity_grid': target_HD_to_activity_grid,
        'model_HD_to_activity_grid': model_HD_to_activity_grid,
        'AV_to_activity_grid': AV_to_activity_grid,
        'AV_to_activity_linear_model': AV_to_activity_linear_model_coefficients,
        'target_preferred_angle': target_preferred_angle,
        'model_preferred_angle': model_preferred_angle,
        'differential_target_HD_to_activity': differential_target_HD_to_activity
    }













'''
classify_neurons
Assigns network neurons to classes (compass, positive and negative shifters, weakly tuned,
and untuned) based on tuning profiles
---------------------------------------------------------------------------------------------
Receives
    config :
        Configuration object corresponding to model
    activity :
        numpy.ndarray of network's activity for the testing dataset 
    tuning :
        result of get_tuning_data
    kwargs :
        kwargs which can override config parameters

Returns
    dict { str : numpy.array | List } :
        'keep' :
            List of indices of neurons to keep for analysis (i.e. that are not untuned)
        'compass' :
            List of indices of compass neurons
        'weakly_tuned' :
            List of indices of weakly tuned neurons
        'pos_shift' :
            List of indices of positive shifters
        'neg_shift' :
            List of indices of negative shifters
        'untuned' :
            List of indices of untuned neurons
        'ordered_strat' :
            List of length 5, where each element is one of the above lists, in order
            used for analysis/plotting
        'ordered_flat' :
            Flattened version of ordered_strat, with length config.n_neurons
        'ordered_names' :
            Names of classes in order; corresponds to lists in ordered_strat


'''
def classify_neurons(task, activity, tuning_dict, **kwargs):

    # Get relevant parameters from config

    max_dif_for_untuned = kwargs.get('max_dif_for_untuned', task.config.max_dif_for_untuned)
    max_slope_for_untuned = kwargs.get('max_slope_for_untuned', task.config.max_slope_for_untuned)
    max_slope_for_compass = kwargs.get('max_slope_for_compass', task.config.max_slope_for_compass)
    min_dif_for_compass = kwargs.get('min_dif_for_compass', task.config.min_dif_for_compass)
    max_slope_for_weakly_tuned = kwargs.get('max_slope_for_weakly_tuned', task.config.max_slope_for_weakly_tuned)
    min_dif_for_weakly_tuned = kwargs.get('min_dif_for_weakly_tuned', task.config.min_dif_for_weakly_tuned)
    max_dif_for_weakly_tuned = kwargs.get('max_dif_for_weakly_tuned', task.config.max_dif_for_weakly_tuned)

    # Select relevant tuning information
    differential_target_HD_to_activity = tuning_dict['differential_target_HD_to_activity']
    slope_AV_to_activity = tuning_dict['AV_to_activity_linear_model'][:,0]
    preferred_angle = tuning_dict['target_preferred_angle']

    # Find neurons which satisfy conditions for 'untuned' classification:
    # Differential HD-tuning below a maximum
    has_untuned_min_dif = np.where(differential_target_HD_to_activity < max_dif_for_untuned)[0]
    # Absolute AV-tuning slope below a maximum
    has_untuned_max_slope = np.where(np.abs(slope_AV_to_activity) < max_slope_for_untuned)[0]
    
    # Define untuned neurons as those satisfying BOTH conditions
    untuned_neurons = set(has_untuned_min_dif.tolist()) & set(has_untuned_max_slope.tolist())
    untuned_neurons = np.array(list(untuned_neurons))

    # Make a list of neurons to keep for analysis (i.e., not untuned)
    keep_neurons = np.setdiff1d(np.arange(task.config.n_neurons), untuned_neurons)

    # Distinguish neurons with negative slope and positive slope
    neg_slope_neurons = np.where((slope_AV_to_activity < 0))[0]
    neg_slope_neurons = np.setdiff1d(neg_slope_neurons, untuned_neurons)

    pos_slope_neurons = np.where((slope_AV_to_activity > 0))
    pos_slope_neurons = np.setdiff1d(pos_slope_neurons, untuned_neurons)

    # Find neurons which satisfy conditions for 'compass' classification:
    # Differential HD-tuning above a minimum
    has_compass_min_dif = np.where((differential_target_HD_to_activity > min_dif_for_compass))[0]
    # Absolute AV-tuning slope below a maximum
    has_compass_max_slope = np.where((np.abs(slope_AV_to_activity) < max_slope_for_compass))[0]

    # Define compass neurons as those satisfying BOTH conditions...
    compass_neurons = set(has_compass_min_dif.tolist()) & set(has_compass_max_slope.tolist())
    compass_neurons = np.array(list(compass_neurons))
    # ...and which aren't considered untuned (in case of overlap in parameters)
    compass_neurons = np.setdiff1d(compass_neurons, untuned_neurons)

    # Find neurons which satisfy conditions for 'weakly tuned' classification:
    # Differential HD-tuning within a range
    has_weakly_tuned_min_dif = np.where((differential_target_HD_to_activity > min_dif_for_weakly_tuned))[0]
    has_weakly_tuned_max_dif = np.where((differential_target_HD_to_activity < max_dif_for_weakly_tuned))[0]
    # Absolute AV-tuning slope below a maximum
    has_weakly_tuned_max_slope = np.where((np.abs(slope_AV_to_activity) < max_slope_for_weakly_tuned))[0]

    # Define weakly tuned neurons as those satisfying ALL conditions...
    weakly_tuned_neurons = set(has_weakly_tuned_min_dif.tolist()) & set(has_weakly_tuned_max_dif.tolist()) & set(has_weakly_tuned_max_slope.tolist())
    weakly_tuned_neurons = np.array(list(weakly_tuned_neurons))
    # ..and not found by previous classifications
    weakly_tuned_neurons = np.setdiff1d(weakly_tuned_neurons, np.concatenate((untuned_neurons, compass_neurons)))

    # Define shifters by slope, from among those neurons still unclassified
    pos_shift_neurons = np.setdiff1d(pos_slope_neurons, np.concatenate([weakly_tuned_neurons, compass_neurons, untuned_neurons]))
    neg_shift_neurons = np.setdiff1d(neg_slope_neurons, np.concatenate([weakly_tuned_neurons, compass_neurons, untuned_neurons]))

    # Sort compass neurons by preferred angle
    if len(compass_neurons) > 0:
        group_sort = preferred_angle[compass_neurons]
        compass_neurons = compass_neurons[np.argsort(group_sort)]

    # Sort positive shifters by decreasing slope
    if len(pos_shift_neurons) > 0:
        # group_sort = slope_AV_to_activity[pos_shift_neurons]
        group_sort = preferred_angle[pos_shift_neurons]
        pos_shift_neurons = pos_shift_neurons[np.argsort(group_sort)]

    # Sort negative shifters by decreasing absolute slope
    if len(neg_shift_neurons) > 0:
        # group_sort = np.abs(slope_AV_to_activity[neg_shift_neurons])
        group_sort = preferred_angle[neg_shift_neurons]
        neg_shift_neurons = neg_shift_neurons[np.argsort(group_sort)]

    # Sort weakly tuned neurons by preferred angle
    if len(weakly_tuned_neurons) > 0:
        # group_sort = preferred_angle[weakly_tuned_neurons]
        group_sort = preferred_angle[weakly_tuned_neurons]
        weakly_tuned_neurons = weakly_tuned_neurons[np.argsort(group_sort)]

    # Sort untuned neurons by decreasing mean activity
    if len(untuned_neurons) > 0:
        group_sort = np.mean(activity.reshape((-1, task.config.n_neurons)), axis=0)[untuned_neurons]
        untuned_neurons = untuned_neurons[np.argsort(group_sort)[::-1]]

    # Defined 'stratified' ordering, where each element in list is a list of indices for a class
    ordered_strat = [
        compass_neurons, pos_shift_neurons, neg_shift_neurons, weakly_tuned_neurons, untuned_neurons]

    # Define 'flat' ordering, where all indices are in one list
    ordered_flat = np.int64(np.concatenate(ordered_strat))

    return {
        'keep': keep_neurons,
        'compass': compass_neurons,
        'weakly_tuned': weakly_tuned_neurons,
        'pos_shift': pos_shift_neurons,
        'neg_shift': neg_shift_neurons,
        'untuned': untuned_neurons,
        'ordered_strat': ordered_strat,
        'ordered_flat': ordered_flat,
        'ordered_names': ['Compass', 'Positive Shift', 'Negative Shift', 'Weakly Tuned', 'Untuned']
    }









def test_allo(task, net, batch, checkpoint_path=None, **kwargs):

    print('Testing model')

    figures = {}

    if not kwargs.get('ignore_loss', False):

        # Only create loss plot if checkpoint is supplied (where losses are saved)
        if checkpoint_path is not None:

            try:
                # Retrieve losses
                checkpoint = torch.load(f'{checkpoint_path}', map_location=torch.device(task.config.device))
                test_losses = checkpoint['test_losses']
                train_losses = checkpoint['train_losses']

                figures['loss'] = loss_plot(task, test_losses, train_losses)

                print('\tGenerated loss plot.')
            except Exception as e:
                print(f'\tLoss plot generation failed: {e}')




# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Examples ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


    _, activity, outputs = net(batch['inputs'], noise=batch['noise'])

    # Detach resulting tensors for use with numpy-based matplotlib
    inputs, targets, mask = batch['inputs'].detach().cpu().numpy(), batch['targets'].detach().cpu().numpy(), batch['mask'].detach().cpu().numpy()
    activity, outputs = activity.detach().cpu().numpy(), outputs.detach().cpu().numpy()

    time_mask = mask[0,:,0]
    inputs, targets = inputs[:,time_mask,:], targets[:,time_mask,:]
    activity, outputs = activity[:,time_mask,:], outputs[:,time_mask,:]

    if not kwargs.get('ignore_examples', False):

        try:
            # Generate fit examples plot
            figures['fit_examples'] = fit_examples_plot(task, targets, outputs)

            print('\tGenerated fit example plot')
        except Exception as e:
            print(f'\tFit example plot generation failed: {e}')
        





# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Tuning ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

    if not kwargs.get('ignore_tuning', False):

        # Calculate tuning profiles of all neurons, and classify based on those profiles
        tuning_dict = get_tuning_data(task, inputs, targets, outputs, activity, **kwargs)
        class_dict = classify_neurons(task, activity, tuning_dict)

        print('\tComputed tuning data.')

        try:
            # Generate classification plot
            figures['classifications'] = classification_plot(task, tuning_dict, class_dict)

            print('\t\tGenerated classification plot.')
        except Exception as e:
            print(f'\t\tClassification plot generation failed: {e}')

        try:
            # Generate head-direction tuning plot
            figures['HD_tuning'] = HD_tuning_plot(task, tuning_dict, ordering=class_dict['ordered_flat'])
            
            print('\t\tGenerated HD tuning plot.')
        except Exception as e:
            print(f'\t\tHD tuning plot generation failed: {e}')
        
        try:
            # Generate angular-velocity tuning plot
            figures['AV_tuning'] = AV_tuning_plot(task, tuning_dict, ordering=class_dict['ordered_flat'])
            
            print('\t\tGenerated AV tuning plot.')
        except Exception as e:
            print(f'\t\tAV tuning plot generation failed: {e}')
        
        try:
            # Generate two version of head-direction to angular-velocity tuning plots 
            # (based on either target or predicted head direction)
            figures['target_HD-AV_tuning'] = HD_AV_tuning_plot(task, tuning_dict['target_HD_to_AV_grid_masked'], ordering=class_dict['ordered_flat'])
            
            figures['model_HD-AV_tuning'] = HD_AV_tuning_plot(task, tuning_dict['model_HD_to_AV_grid_masked'], ordering=class_dict['ordered_flat'])
            
            print('\t\tGenerated HD-AV tuning plots.')
        except Exception as e:
            print(f'\t\tHD-AV tuning plot generation failed: {e}')
    
    


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Connectivity ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

    if not kwargs.get('ignore_connectivity', False) and not kwargs.get('ignore_tuning', False):

        try:
            # Generate connectivity plot
            figures['connectivity'] = connectivity_plot(task, net, class_dict)

            print('\tGenerated Connectivity plot.')
        except Exception as e:
            print(f'\tConnectivity plot generation failed: {e}')






# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Lesions ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


    if not kwargs.get('ignore_lesions', False):
        try:
            # Generate lesions plot
            figures['lesions'] = lesion_plot(task, net, class_dict)

            print('\tGenerated Lesion plot.')
        except Exception as e:
            print(f'\tLesions plot generation failed: {e}')



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Snapshot ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

    # If checkpoint path is supplied, save plots as .png's
    if checkpoint_path is not None:
        checkpoint_dir = '/'.join(checkpoint_path.split('/')[:-1])
        for name, fig in figures.items():
            fig.savefig(f'{checkpoint_dir}/{name}.png', transparent=False)

        print('\tSuccesfully saved plots.')

        figure_pastes = {
            'loss': (3, 2, 0, 0),
            'fit_examples': (3, 2, 0, 2),
            'classifications': (3, 2, 0, 4),
            'HD_tuning': (3, 3, 3, 0),
            'AV_tuning': (3, 3, 6, 0),
            'target_HD-AV_tuning': (3, 3, 3, 3),
            'model_HD-AV_tuning': (3, 3, 6, 3),
            'connectivity': (2, 2, 10.5, 0),
            'lesions': (4, 2, 9, 2)
        }

        # Also save an image which contains all plots
        create_snapshot_image(task, figures, checkpoint_dir, figure_pastes, width=13, height=6, )

        print('\tGenerated checkpoint snapshot.\n')

    return figures








if __name__ == '__main__':
    import sys
    import o2s.Tasks

    checkpoint_path = sys.argv[1]

    plt.clf()
    test_gamut(checkpoint_path,
               subtask_batch_size=50**2, subtask_n_timesteps=510,
               include_umap=False,
               include_dimensionality=True, dimensionality_prop_explained=1.0, dimensionality_var_explained=0.8,
               include_metric=False, metric_select_tau=[0, 0.2, 0.5, 0.8, 1, 1.5, 2, 2.5, 5, 7.5, 10, 15, 25, 49], metric_n_samples=25, metric_alpha=0.1, metric_d_theta=1e-6, metric_order=3, metric_dtype=torch.float64, metric_n_instantiations=0, metric_norm_dphi=False,
               include_trajectories=False, trajectories_select_t=[(10, 0, 10), (10, 0, 50), (50, 0, 500), (50, 50, 500)],
               include_stability=False, stability_n_timesteps=200, stability_total_n_timesteps=200, stability_slow_mult=10,
               include_lesions=False, lesions_n_lesions=25,
               include_tuning=False, tuning_batch_size=2500, #tuning_vars_list=['HD', 'ego_SD', 'allo_SD', 'AV', 'x','y'],
               include_fourier=False,
               include_eigenspectra=False)
