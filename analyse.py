import numpy as np
import pandas as pd
import sys
import time
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from typing import Dict

from scipy.optimize import minimize
from scipy.linalg import eig
import multiprocessing as mp
from torch.multiprocessing import cpu_count

import torch



from net import *
from data import *
from train import *
from config import *
from test_funcs import *
from task import *


def ReTanh(x):
    return np.maximum(np.zeros_like(x), np.tanh(x))


def minimise_from_x_0(x_0: np.ndarray, u: np.ndarray, t: int, vars: Dict[str, np.ndarray], W_rec: np.ndarray, W_in: np.ndarray, b: np.ndarray, 
                      checkpoint_dir: str, verbose: bool, i: int, queue: mp.Queue, queue_i: int):
    if queue is not None:
        assert queue_i is not None

    start_time = time.time()

    if verbose:
        print(f'\tMinimising starting point {i+1}, x_0={x_0[:5]}..., u={u}, t={t}, hd={vars["hd"]}, sd={vars["sd"]}')

    def _F(x):
        return (-x + W_rec@ReTanh(x) + W_in@u + b)

    def _q(x):
        return (1/2) * np.linalg.norm(_F(x))**2


    res = minimize(_q, x_0, method='Powell', tol=10e-9, options={'maxiter': 1000})

    if verbose:
        print('\tMinimisation {} {} with value {:.4E} (in {:.2f}s)'.format(
            i+1, 'succeeded' if res.success else 'failed', res.fun, time.time() - start_time))
        
    result = {
        'i': i,
        'q': res.fun,
        'state': res.x,
        'input': u,
        'time': t,
        'vars': vars
    }
    
    if checkpoint_dir is not None:
        torch.save(result, f'{checkpoint_dir}/analyse-temp/{time.time()}-{i}.pt')

    if queue is not None:
        queue.put(queue_i)

    return result



def find_fixed_points(task, net, checkpoint_path, num_x_0=100, keep_inputs=[], keep_times=[], verbose=False, num_processes=None):  

    checkpoint_dir = '/'.join(checkpoint_path.split('/')[:-1])
    if not os.path.exists(f'{checkpoint_dir}/analyse-temp'):
        os.mkdir(f'{checkpoint_dir}/analyse-temp')
        
    probe = TaskDataset(task).get_batch()

    states = net(probe['inputs'], noise=probe['noise'])[0].detach().numpy()
    inputs = probe['inputs'].detach().numpy()
    vars = {k: v.detach().numpy() for k,v in probe['vars'].items()}
    for k,v in vars.items():
        if len(v.shape) == 1:
            assert len(v) == task.config.batch_size, 'Unexpected shape for variable {}'.format(k)
            vars[k] = np.tile(v.reshape((task.config.batch_size,1)), (1, task.config.n_timesteps))

    # Remove unwanted inputs
    inputs[:,:,[i for i in range(inputs.shape[2]) if i not in keep_inputs]] = 0

    # Select time slices
    T = np.tile(np.arange(task.config.n_timesteps), (task.config.batch_size, 1))
    if len(keep_times) > 0:
        T = T[:,keep_times]

    states = states[np.arange(states.shape[0])[:, None], T]
    inputs = inputs[np.arange(inputs.shape[0])[:, None], T]
    vars = {k: v[np.arange(v.shape[0])[:, None], T] for k,v in vars.items()}

    # Flatten batch/time dimensions
    states = states.reshape(-1, task.config.n_neurons)
    inputs = inputs.reshape(-1, task.config.n_inputs)
    T = T.reshape(-1)
    vars = {k: v.reshape(-1) for k,v in vars.items()}

    # Select random starting points
    random_i = np.random.permutation(states.shape[0])[:num_x_0]
    X = states[random_i]
    U = inputs[random_i]
    T = T[random_i]
    vars = {i: {k: v[random_i[i]] for k,v in vars.items()} for i in range(num_x_0)}

    W_rec, W_in, b = net.W_rec.weight.detach().numpy(), net.W_in.weight.detach().numpy(), net.W_in.bias.detach().numpy()

    c = cpu_count() if num_processes is None else num_processes
    c = min(c, num_x_0)

    if verbose:
        print('Running on {} cores'.format(c))
    
    queue = mp.Queue(maxsize=c)
    processes = []
    for i in range(c):
        queue_i = i
        p = mp.Process(target=minimise_from_x_0, args=(X[i], U[i], T[i], vars[i], W_rec, W_in, b, checkpoint_dir, verbose, i, queue, queue_i))
        p.start()
        processes.append(p)

    for next_i in range(c, num_x_0):
        queue_i = queue.get()

        processes[queue_i].join()
        p = mp.Process(target=minimise_from_x_0, args=(X[next_i], U[next_i], T[next_i], vars[next_i], W_rec, W_in, b, checkpoint_dir, verbose, next_i, queue, queue_i))
        p.start()
        processes[queue_i] = p

    for p in processes:
        p.join()

    queue.close()
    queue.join_thread()



    recover_fixed_points_from_temp(checkpoint_path)

    # if len(os.listdir(f'{checkpoint_dir}/analyse-temp'))==0:
    #     os.rmdir(f'{checkpoint_dir}/analyse-temp')






    

def recover_fixed_points_from_temp(checkpoint_path):
    checkpoint_dir = '/'.join(checkpoint_path.split('/')[:-1])

    if 'fixed_points.pt' in os.listdir(checkpoint_dir):
        result = torch.load(f'{checkpoint_dir}/fixed_points.pt', map_location='cpu')
    else:
        result = {
            'i': [],
            'q': [],
            'state': [],
            'input': [],
            'time': [],
            'vars': {}
        }

    n_recovered = 0
    for tempfile in os.listdir(f'{checkpoint_dir}/analyse-temp'):
        if '.pt' not in tempfile:
            continue
        tempresult = torch.load(f'{checkpoint_dir}/analyse-temp/{tempfile}', map_location='cpu')
        for key in result.keys():
            if key=='vars':
                for vars_key in tempresult['vars'].keys():
                    if vars_key not in result['vars']:
                        result['vars'][vars_key] = []
                    result['vars'][vars_key].append(tempresult['vars'][vars_key])
            else:
                result[key].append(tempresult[key])
        n_recovered += 1
        os.remove(f'{checkpoint_dir}/analyse-temp/{tempfile}')

    print(f'Recovered {n_recovered} points')

    torch.save(result, f'{checkpoint_dir}/fixed_points.pt')



# def get_Jacobian_at(x, net):
#     n = net.n_neurons
#     W = net.W_rec.weight.detach().cpu().numpy()
#     r_prime = np.where( x < 0 , np.zeros_like(x) , 1-(np.tanh(x)**2) )

#     J = np.zeros_like(W)
#     for i in range(n):
#         for j in range(n):
#             delta = 1 if i==j else 0
#             J[i,j] = -delta + W[i,j]*r_prime[j]

#     return J

# def get_eigendecomposition_at(x, net):
#     n = net.n_neurons
#     W = net.W_rec.weight.detach().cpu().numpy()
#     c = net.W_in.bias.detach().cpu().numpy()

#     A = get_Jacobian_at(x, net)
#     b = -x + W@ReTanh(x) + c

#     D = np.empty((n+1, n+1))
#     D[:n,:n] = A
#     D[:n,n] = b
#     D[n,:n] = 0
#     D[n,n] = 1

#     eigenvalues, left_eigenvectors, right_eigenvectors = eig(D, left=True, right=True)

#     scale_components = right_eigenvectors[n,:]
#     for i, scale_component in enumerate(scale_components):
#         if scale_component == 0:
#             continue
#         eigenvalues[i] /= scale_component
#         left_eigenvectors[:,i] /= scale_component
#         right_eigenvectors[:,i] /= scale_component

#     eigenvalues,unique_eigvals_i = np.unique(eigenvalues, return_index=True)
#     left_eigenvectors = left_eigenvectors[:n][:,unique_eigvals_i]
#     right_eigenvectors = right_eigenvectors[:n][:,unique_eigvals_i]
    
#     return left_eigenvectors, eigenvalues, right_eigenvectors

# def attractor_plot(config, net, activity, checkpoint_path, **kwargs):
#     n_examples = kwargs.get('n_fit_examples', config.n_fit_examples)
#     slow_threshold = kwargs.get('slow_point_threshold', config.slow_point_threshold)
#     fixed_threshold = kwargs.get('fixed_point_threshold', config.fixed_point_threshold)

#     width = kwargs.get('test_fig_width', config.test_fig_width)
#     height = kwargs.get('test_fig_height', config.test_fig_height)
#     margin = kwargs.get('test_fig_margin', config.test_fig_margin)

#     use = kwargs.get('use', 'pca')
#     n_nonlinear_samples = kwargs.get('n_nonlinear_samples', 10000)
#     dims = kwargs.get('dims', [0,1,2])
#     marker_target = kwargs.get('marker_target', 'q')
#     angles = kwargs.get('angles', None)
#     eig_eps = kwargs.get('eig_eps', 10e-2)
#     eig_d = kwargs.get('eig_d', 0.5)

#     checkpoint_dir = '/'.join(checkpoint_path.split('/')[:-1])
#     Q = pd.read_csv(f'{checkpoint_dir}/slow_points.csv').to_numpy()

#     q, HD, SD = Q[:,0], Q[:,1], Q[:,2]
#     Q = net.activation_func(torch.tensor(Q[:,3:])).numpy()

#     slow_i = np.where((q > fixed_threshold) & (q <= slow_threshold))[0]
#     fixed_i = np.where((q <= fixed_threshold))[0]

#     reduction_samples = np.concatenate((Q, activity.reshape((-1, config.n_neurons))), axis=0)
#     pca = PCA(n_components=10)
#     pca.fit(reduction_samples)

#     Q_red = pca.transform(Q)
#     activity_red = pca.transform(activity.reshape((-1, config.n_neurons))).reshape((activity.shape[0], activity.shape[1], pca.n_components_))

#     example_trajectories_i = np.random.permutation(config.test_batch_size)[:n_examples]
#     trajectories = activity_red[example_trajectories_i]

#     if use != 'pca':
#         n_pca_components = np.where(pca.explained_variance_ratio_ < 0.025)[0][0] + 1

#         nonlinear_reduction_samples = activity_red[example_trajectories_i].reshape((-1, pca.n_components_))

#         n_Q_samples = min(Q.shape[0], n_nonlinear_samples - nonlinear_reduction_samples.shape[0])
#         Q_sample_i = np.random.permutation(Q.shape[0])[:n_Q_samples]
#         nonlinear_reduction_samples = np.append(nonlinear_reduction_samples, Q_red[Q_sample_i], axis=0)

#         if nonlinear_reduction_samples.shape[0] < n_nonlinear_samples:
#             n_supplement_samples = n_nonlinear_samples - nonlinear_reduction_samples.shape[0]
#             supplement_sample_i = np.random.permutation(activity_red.shape[0]*activity_red.shape[1])[:n_supplement_samples]
#             nonlinear_reduction_samples = np.append(nonlinear_reduction_samples, activity_red.reshape((-1, pca.n_components_))[supplement_sample_i], axis=0)


#         print(f'Using {nonlinear_reduction_samples.shape[0]} samples in {n_pca_components} PCA dimensions for reduction with {use.upper()}')

#         if use == 'tsne':
#             nl_reducer = TSNE(n_components=3, verbose=100, n_iter=500)
#         elif use == 'mds':
#             nl_reducer = MDS(n_components=3, verbose=100, n_jobs=-1)
        
#         reduced_samples = nl_reducer.fit_transform(nonlinear_reduction_samples[:,:n_pca_components])

#         Q_start_i = n_examples*activity.shape[1]
#         Q_red = reduced_samples[Q_start_i:Q_start_i+n_Q_samples]

#         trajectories = np.zeros((n_examples, activity.shape[1], 3))
#         for i in range(n_examples):
#             trajectories[i] = reduced_samples[i*activity.shape[1]:(i+1)*activity.shape[1]].reshape((activity.shape[1], 3))

    

#     fig = plt.figure(figsize=(width,width))

#     if angles is None:
#         angles = [[ (45, -45), (90, 0), (45, 45) ], 
#                 [ (0, -90), (0, 0), (0, 90) ], 
#                 [ (-45, -45), (-90, 0), (-45, 45) ]]
#         if use == 'pca':
#             angles[2][2] = 'PCA'
        
#     n_rows = len(angles)
#     n_cols = len(angles[0])
#     for i in range(n_rows):
#         for j in range(n_cols):
#             if angles[i][j] == 'PCA':
#                 ax = fig.add_subplot(n_rows, n_cols, (i*n_rows + j)+1)

#                 ax.plot(np.arange(1, pca.n_components_+1, 1), pca.explained_variance_ratio_)
            
#             else:

#                 angle = angles[i][j]
#                 ax = fig.add_subplot(3, 3, (i*n_rows + j)+1, projection='3d')

#                 ax.view_init(elev=angle[0], azim=angle[1])
#                 ax.set_xlabel(f'Dim {dims[0]}')
#                 ax.set_ylabel(f'Dim {dims[1]}')
#                 ax.set_zlabel(f'Dim {dims[2]}')
#                 ax.set_xticks([])
#                 ax.set_yticks([])
#                 ax.set_zticks([])

#                 handles = {}
#                 marker_list = None
#                 if marker_target=='q':
#                     marker_list = q
#                 elif marker_target=='HD':
#                     marker_list = HD
#                 elif marker_target=='SD':
#                     marker_list = SD
#                 if len(slow_i) > 0:
#                     slow_artist = ax.scatter(Q_red[slow_i][:,dims[0]], Q_red[slow_i][:,dims[1]], Q_red[slow_i][:,dims[2]], c=marker_list[slow_i], label='Slow', zorder=10, marker='o')
#                     handles['slow'] = slow_artist
#                 if len(fixed_i) > 0:
#                     fixed_artist = ax.scatter(Q_red[fixed_i][:,dims[0]], Q_red[fixed_i][:,dims[1]], Q_red[fixed_i][:,dims[2]], c=marker_list[fixed_i], label='Fixed', zorder=100, marker='v')
#                     handles['fixed'] = fixed_artist

#                 for k, trajectory in enumerate(trajectories):
#                     trajectory_artist, = ax.plot(trajectory[:,dims[0]], trajectory[:,dims[1]], trajectory[:,dims[2]], c='cyan', label='Example Trajectory', linewidth=0.5, zorder=1)
#                     handles['trajectory'] = trajectory_artist

#                 eig_d = 0.1
#                 for k in range(len(q)):
#                     if k not in slow_i and k not in fixed_i:
#                         continue
#                     L, E, R = get_eigendecomposition_at(Q[k], net)
#                     for e_i, eigenvalue in enumerate(E):
#                         # select stable, non-oscillatory modes
#                         if np.abs(np.imag(eigenvalue)) < eig_eps and np.abs(np.real(eigenvalue)) < eig_eps:
#                             pca_mode = pca.transform(R[:,e_i].reshape(1,net.n_neurons).real)
#                             pca_mode = pca_mode[0][dims]
#                             pca_mode *= eig_d / np.linalg.norm(pca_mode)
#                             pca_pos = [Q_red[k,d_i] for d_i in dims]

#                             int_artist, = ax.plot([pca_pos[0]-pca_mode[0], pca_pos[0]+pca_mode[0]], [pca_pos[1]-pca_mode[1], pca_pos[1]+pca_mode[1]], [pca_pos[2]-pca_mode[2], pca_pos[2]+pca_mode[2]], c='red', linewidth=1, zorder=1000, label='Integrative Eigenmode')
#                             handles['integrative'] = int_artist

#                         # if np.abs(np.imag(eigenvalue)) >= eig_eps and np.abs(np.real(eigenvalue)) < eig_eps:
#                         #     pca_mode = pca.transform(R[:,e_i].reshape(1,net.n_neurons).real)
#                         #     pca_mode = pca_mode[0][dims]
#                         #     pca_mode *= d / np.linalg.norm(pca_mode)
#                         #     pca_pos = [Q_red[k,d_i] for d_i in dims]

#                         #     osc_r = np.tile(np.exp(np.array(pca_mode)).reshape(3,1), reps=(1,100))
#                         #     osc_span = np.linspace(0, 2*np.pi, 100)
#                         #     osc_pos = np.tile(np.array(pca_pos).reshape((3,1)), reps=(1,100)) + osc_r*np.cos(osc_span)


#                         #     osc_artist, = ax.plot(osc_pos[0], osc_pos[1], osc_pos[2], c='blue', linewidth=1, zorder=1000)
 

#                 if i==0 and j==n_cols-1:
#                     ax.legend(handles=list(handles.values()))

#                     if 'slow' in handles or 'fixed' in handles:
#                         margin = 0.05
#                         cbar_ax = fig.add_axes([0.75, margin/2, 1 - 0.75 - margin, margin/3])
#                         cbar_handle = 'slow' if 'slow' in handles else 'fixed'
#                         fig.colorbar(handles[cbar_handle], cax=cbar_ax, orientation='horizontal')

#     # Aesthetic settings
#     plt.subplots_adjust(left=0, right=1, top=1, bottom=margin)

#     return fig

    


if __name__ == '__main__':
    from Tasks.util import register_all_tasks
    register_all_tasks()

    assert len(sys.argv) >= 2

    checkpoint_path = sys.argv[1]

    print(f'Analysing {checkpoint_path}')

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    task = Task.from_checkpoint(checkpoint)
    task.config.update(device='cuda' if torch.cuda.is_available() else 'cpu')

    if '-r' in sys.argv:
        recover_fixed_points_from_temp(checkpoint_path)

    else:

        num_x_0 = 100
        if '-n' in sys.argv:
            argi = sys.argv.index('-n')
            num_x_0 = int(sys.argv[argi+1])

        keep_inputs = []
        if '-u' in sys.argv:
            argi = sys.argv.index('-u')
            for input_name in sys.argv[argi+1:]:
                if input_name[0] == '-':
                    break
                keep_inputs.append(task.input_map[input_name])

        keep_times = []
        if '-t' in sys.argv:
            argi = sys.argv.index('-t')
            for keep_time in sys.argv[argi+1:]:
                if keep_time[0] == '-':
                    break
                elif ':' in keep_time:
                    start, end = keep_time.split(':')
                    keep_times.extend(list(range(int(start), int(end))))
                else:
                    keep_times.append(int(keep_time))

        task.config.update(batch_size=num_x_0, n_timesteps=task.config.n_timesteps if len(keep_times)==0 else max(max(keep_times)+1, task.config.init_duration+1))

        net = RNN(task)
        net.load_state_dict(checkpoint['net_state_dict'])

        find_fixed_points(task, net, checkpoint_path=checkpoint_path, num_x_0=num_x_0, keep_inputs=keep_inputs, keep_times=keep_times, verbose=True)
    

