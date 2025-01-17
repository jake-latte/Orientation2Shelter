import numpy as np
import pandas as pd
import sys
import time
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from scipy.optimize import minimize
from scipy.linalg import eig
import multiprocessing as mp
from torch.multiprocessing import spawn,cpu_count


from RNN import *
from Data import *
from Train import *
from Config import *
from Test import *

def ReTanh(x):
    return np.maximum(np.zeros_like(x), np.tanh(x))


def minimise_from_x_0(i, queue, queue_i, x_0, u, W_rec, W_in, b, checkpoint_dir, verbose):

    start_time = time.time()

    if verbose:
        print(f'\tMinimising starting point {i}')

    def _F(x):
        return (-x + W_rec@ReTanh(x) + W_in@u + b)

    def _q(x):
        return (1/2) * np.linalg.norm(_F(x))**2


    res = minimize(_q, x_0, method='Powell', options={'maxiter': 1000})

    if verbose:
        print('\tMinimisation {} complete with value {:.6f} (in {:.2f}s)'.format(i, res.fun, time.time() - start_time))

    with open(f'{checkpoint_dir}/analyse-temp/{i}.csv', 'w') as tempfile:
        output = f'{res.fun}'
        for x_i in res.x:
            output += f',{x_i}'
        tempfile.write(output)
        tempfile.close()

    queue.put(queue_i)



def find_slow_points(config, net, checkpoint_path, num_x_0=100, verbose=False):  

    dataset = config.dataset(config, for_training=False)
    probe = dataset.get_batch()
    states,_,outputs = net(probe['inputs'], noise=probe['noise'])

    inputs, targets = probe['inputs'].detach().cpu().numpy()[:,100:,:], probe['targets'].detach().cpu().numpy()[:,100:,:]
    inputs = np.zeros_like(inputs)
    states, outputs = states.detach().cpu().numpy()[:,100:,:], outputs.detach().cpu().numpy()[:,100:,:]
    W_rec, W_in, b = net.W_rec.weight.detach().cpu().numpy(), net.W_in.weight.detach().cpu().numpy(), net.W_in.bias.detach().cpu().numpy()

    X = states.reshape(-1, config.n_neurons)
    U = inputs.reshape(-1, config.n_inputs)

    random_i = np.random.permutation(states.shape[0]*states.shape[1])[:num_x_0]
    X, U = X[random_i], U[random_i]

    if verbose:
        print('\tRunning on {} cores'.format(cpu_count()))

    checkpoint_dir = '/'.join(checkpoint_path.split('/')[:-1])
    if not os.path.exists(f'{checkpoint_dir}/analyse-temp'):
        os.mkdir(f'{checkpoint_dir}/analyse-temp')

    c = cpu_count()
    queue = mp.Queue(maxsize=c)
    processes = []
    for i in range(c):
        queue_i = i
        p = mp.Process(target=minimise_from_x_0, args=(i, queue, queue_i, X[i], U[i], W_rec, W_in, b, checkpoint_dir, verbose))
        p.start()
        processes.append(p)

    for next_i in range(c, num_x_0):
        queue_i = queue.get()

        processes[queue_i].join()
        p = mp.Process(target=minimise_from_x_0, args=(next_i, queue, queue_i, X[next_i], U[next_i], W_rec, W_in, b, checkpoint_dir, verbose))
        p.start()
        processes[queue_i] = p

    for p in processes:
        p.join()


    recover_slow_points_from_temp(config, checkpoint_path)

    if len(os.listdir(f'{checkpoint_dir}/analyse-temp'))==0:
        os.rmdir(f'{checkpoint_dir}/analyse-temp')
    


def recover_slow_points_from_temp(config, checkpoint_path):
    checkpoint_dir = '/'.join(checkpoint_path.split('/')[:-1])

    df = pd.DataFrame({'q': [], **{f'x_{i}': [] for i in range(config.n_neurons)}})
    for tempfile in os.listdir(f'{checkpoint_dir}/analyse-temp'):
        if '.csv' not in tempfile:
            continue
        tempdf = pd.read_csv(f'{checkpoint_dir}/analyse-temp/{tempfile}', header=None, names=df.columns)
        df = pd.concat([df, tempdf], ignore_index=True)
        df.reset_index()
        os.remove(f'{checkpoint_dir}/analyse-temp/{tempfile}')

    print(f'Recovered {len(df)} points')

    if 'slow_points.csv' in os.listdir(checkpoint_dir):
        prev_df = pd.read_csv(f'{checkpoint_dir}/slow_points.csv')
        df = pd.concat([df, prev_df], ignore_index=True)
        df.reset_index()

    df.to_csv(f'{checkpoint_dir}/slow_points.csv', index=False)



def get_Jacobian_at(x, net):
    n = net.n_neurons
    W = net.W_rec.get_weight().detach().cpu().numpy()
    r_prime = np.where( x < 0 , np.zeros_like(x) , 1-(np.tanh(x)**2) )

    J = np.zeros_like(W)
    for i in range(n):
        for j in range(n):
            delta = 1 if i==j else 0
            J[i,j] = -delta + W[i,j]*r_prime[j]

    return J

def get_eigendecomposition_at(x, net):
    n = net.n_neurons
    W = net.W_rec.get_weight().detach().cpu().numpy()
    c = net.W_in.bias.detach().cpu().numpy()

    A = get_Jacobian_at(x, net)
    b = -x + W@ReTanh(x) + c

    D = np.empty((n+1, n+1))
    D[:n,:n] = A
    D[:n,n] = b
    D[n,:n] = 0
    D[n,n] = 1

    eigenvalues, left_eigenvectors, right_eigenvectors = eig(D, left=True, right=True)

    scale_components = right_eigenvectors[n,:]
    for i, scale_component in enumerate(scale_components):
        if scale_component == 0:
            continue
        eigenvalues[i] /= scale_component
        left_eigenvectors[:,i] /= scale_component
        right_eigenvectors[:,i] /= scale_component

    eigenvalues,unique_eigvals_i = np.unique(eigenvalues, return_index=True)
    left_eigenvectors = left_eigenvectors[:n][:,unique_eigvals_i]
    right_eigenvectors = right_eigenvectors[:n][:,unique_eigvals_i]
    
    return left_eigenvectors, eigenvalues, right_eigenvectors




def attractor_plot(config, net, activity, checkpoint_path, **kwargs):
    n_examples = kwargs.get('n_fit_examples', config.n_fit_examples)
    slow_threshold = kwargs.get('slow_point_threshold', config.slow_point_threshold)
    fixed_threshold = kwargs.get('fixed_point_threshold', config.fixed_point_threshold)

    width = kwargs.get('test_fig_width', config.test_fig_width)
    height = kwargs.get('test_fig_height', config.test_fig_height)
    margin = kwargs.get('test_fig_margin', config.test_fig_margin)

    use = kwargs.get('use', 'pca')
    n_nonlinear_samples = kwargs.get('n_nonlinear_samples', 10000)
    dims = kwargs.get('dims', [0,1,2])
    marker_target = kwargs.get('marker_target', 'q')
    angles = kwargs.get('angles', None)
    eig_eps = kwargs.get('eig_eps', 10e-2)
    eig_d = kwargs.get('eig_d', 0.5)

    checkpoint_dir = '/'.join(checkpoint_path.split('/')[:-1])
    Q = pd.read_csv(f'{checkpoint_dir}/slow_points.csv').to_numpy()

    q, HD, SD = Q[:,0], Q[:,1], Q[:,2]
    Q = net.activation_func(torch.tensor(Q[:,3:])).numpy()

    slow_i = np.where((q > fixed_threshold) & (q <= slow_threshold))[0]
    fixed_i = np.where((q <= fixed_threshold))[0]

    reduction_samples = np.concatenate((Q, activity.reshape((-1, config.n_neurons))), axis=0)
    pca = PCA(n_components=10)
    pca.fit(reduction_samples)

    Q_red = pca.transform(Q)
    activity_red = pca.transform(activity.reshape((-1, config.n_neurons))).reshape((activity.shape[0], activity.shape[1], pca.n_components_))

    example_trajectories_i = np.random.permutation(config.test_batch_size)[:n_examples]
    trajectories = activity_red[example_trajectories_i]

    if use != 'pca':
        n_pca_components = np.where(pca.explained_variance_ratio_ < 0.025)[0][0] + 1

        nonlinear_reduction_samples = activity_red[example_trajectories_i].reshape((-1, pca.n_components_))

        n_Q_samples = min(Q.shape[0], n_nonlinear_samples - nonlinear_reduction_samples.shape[0])
        Q_sample_i = np.random.permutation(Q.shape[0])[:n_Q_samples]
        nonlinear_reduction_samples = np.append(nonlinear_reduction_samples, Q_red[Q_sample_i], axis=0)

        if nonlinear_reduction_samples.shape[0] < n_nonlinear_samples:
            n_supplement_samples = n_nonlinear_samples - nonlinear_reduction_samples.shape[0]
            supplement_sample_i = np.random.permutation(activity_red.shape[0]*activity_red.shape[1])[:n_supplement_samples]
            nonlinear_reduction_samples = np.append(nonlinear_reduction_samples, activity_red.reshape((-1, pca.n_components_))[supplement_sample_i], axis=0)


        print(f'Using {nonlinear_reduction_samples.shape[0]} samples in {n_pca_components} PCA dimensions for reduction with {use.upper()}')

        if use == 'tsne':
            nl_reducer = TSNE(n_components=3, verbose=100, n_iter=500)
        elif use == 'mds':
            nl_reducer = MDS(n_components=3, verbose=100, n_jobs=-1)
        
        reduced_samples = nl_reducer.fit_transform(nonlinear_reduction_samples[:,:n_pca_components])

        Q_start_i = n_examples*activity.shape[1]
        Q_red = reduced_samples[Q_start_i:Q_start_i+n_Q_samples]

        trajectories = np.zeros((n_examples, activity.shape[1], 3))
        for i in range(n_examples):
            trajectories[i] = reduced_samples[i*activity.shape[1]:(i+1)*activity.shape[1]].reshape((activity.shape[1], 3))

    

    fig = plt.figure(figsize=(width,width))

    if angles is None:
        angles = [[ (45, -45), (90, 0), (45, 45) ], 
                [ (0, -90), (0, 0), (0, 90) ], 
                [ (-45, -45), (-90, 0), (-45, 45) ]]
        if use == 'pca':
            angles[2][2] = 'PCA'
        
    n_rows = len(angles)
    n_cols = len(angles[0])
    for i in range(n_rows):
        for j in range(n_cols):
            if angles[i][j] == 'PCA':
                ax = fig.add_subplot(n_rows, n_cols, (i*n_rows + j)+1)

                ax.plot(np.arange(1, pca.n_components_+1, 1), pca.explained_variance_ratio_)
            
            else:

                angle = angles[i][j]
                ax = fig.add_subplot(3, 3, (i*n_rows + j)+1, projection='3d')

                ax.view_init(elev=angle[0], azim=angle[1])
                ax.set_xlabel(f'Dim {dims[0]}')
                ax.set_ylabel(f'Dim {dims[1]}')
                ax.set_zlabel(f'Dim {dims[2]}')
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_zticks([])

                handles = {}
                marker_list = None
                if marker_target=='q':
                    marker_list = q
                elif marker_target=='HD':
                    marker_list = HD
                elif marker_target=='SD':
                    marker_list = SD
                if len(slow_i) > 0:
                    slow_artist = ax.scatter(Q_red[slow_i][:,dims[0]], Q_red[slow_i][:,dims[1]], Q_red[slow_i][:,dims[2]], c=marker_list[slow_i], label='Slow', zorder=10, marker='o')
                    handles['slow'] = slow_artist
                if len(fixed_i) > 0:
                    fixed_artist = ax.scatter(Q_red[fixed_i][:,dims[0]], Q_red[fixed_i][:,dims[1]], Q_red[fixed_i][:,dims[2]], c=marker_list[fixed_i], label='Fixed', zorder=100, marker='v')
                    handles['fixed'] = fixed_artist

                for k, trajectory in enumerate(trajectories):
                    trajectory_artist, = ax.plot(trajectory[:,dims[0]], trajectory[:,dims[1]], trajectory[:,dims[2]], c='cyan', label='Example Trajectory', linewidth=0.5, zorder=1)
                    handles['trajectory'] = trajectory_artist

                eig_d = 0.1
                for k in range(len(q)):
                    if k not in slow_i and k not in fixed_i:
                        continue
                    L, E, R = get_eigendecomposition_at(Q[k], net)
                    for e_i, eigenvalue in enumerate(E):
                        # select stable, non-oscillatory modes
                        if np.abs(np.imag(eigenvalue)) < eig_eps and np.abs(np.real(eigenvalue)) < eig_eps:
                            pca_mode = pca.transform(R[:,e_i].reshape(1,net.n_neurons).real)
                            pca_mode = pca_mode[0][dims]
                            pca_mode *= eig_d / np.linalg.norm(pca_mode)
                            pca_pos = [Q_red[k,d_i] for d_i in dims]

                            int_artist, = ax.plot([pca_pos[0]-pca_mode[0], pca_pos[0]+pca_mode[0]], [pca_pos[1]-pca_mode[1], pca_pos[1]+pca_mode[1]], [pca_pos[2]-pca_mode[2], pca_pos[2]+pca_mode[2]], c='red', linewidth=1, zorder=1000, label='Integrative Eigenmode')
                            handles['integrative'] = int_artist

                        # if np.abs(np.imag(eigenvalue)) >= eig_eps and np.abs(np.real(eigenvalue)) < eig_eps:
                        #     pca_mode = pca.transform(R[:,e_i].reshape(1,net.n_neurons).real)
                        #     pca_mode = pca_mode[0][dims]
                        #     pca_mode *= d / np.linalg.norm(pca_mode)
                        #     pca_pos = [Q_red[k,d_i] for d_i in dims]

                        #     osc_r = np.tile(np.exp(np.array(pca_mode)).reshape(3,1), reps=(1,100))
                        #     osc_span = np.linspace(0, 2*np.pi, 100)
                        #     osc_pos = np.tile(np.array(pca_pos).reshape((3,1)), reps=(1,100)) + osc_r*np.cos(osc_span)


                        #     osc_artist, = ax.plot(osc_pos[0], osc_pos[1], osc_pos[2], c='blue', linewidth=1, zorder=1000)
 

                if i==0 and j==n_cols-1:
                    ax.legend(handles=list(handles.values()))

                    if 'slow' in handles or 'fixed' in handles:
                        margin = 0.05
                        cbar_ax = fig.add_axes([0.75, margin/2, 1 - 0.75 - margin, margin/3])
                        cbar_handle = 'slow' if 'slow' in handles else 'fixed'
                        fig.colorbar(handles[cbar_handle], cax=cbar_ax, orientation='horizontal')

    # Aesthetic settings
    plt.subplots_adjust(left=0, right=1, top=1, bottom=margin)

    return fig



    


if __name__ == '__main__':

    assert len(sys.argv) >= 3

    argv = sys.argv.copy()
    savedir = 'reference-models' if len(argv) == 3 else argv.pop(1)
    build_dir = f'{savedir}/{argv[1]}'
    min_checkpoint, min_checkpoint_dir = None, None
    for dir in os.listdir(build_dir):
        try:
            if 'loss' in dir:
                checkpoint = float(dir.split(':')[1])
                if min_checkpoint is None or checkpoint < min_checkpoint:
                    min_checkpoint = checkpoint
                    min_checkpoint_dir = dir
        except:
            continue
    checkpoint_dir = f'{build_dir}/{min_checkpoint_dir}'
    checkpoint_file = None
    for file in os.listdir(checkpoint_dir):
        if '.pt' in file:
            checkpoint_file = file
            break
    checkpoint_file = f'{checkpoint_dir}/{checkpoint_file}'

    print(f'Analysing {checkpoint_file}')

    num_x_0 = int(argv[2])

    checkpoint = torch.load(checkpoint_file, map_location='cpu')
    
    config = Config(**checkpoint['config'])
    config.device = 'cpu'

    if os.path.isdir(f'{checkpoint_dir}/analyse-temp'):
        if len(os.listdir(f'{checkpoint_dir}/analyse-temp'))>0:
            print('Recovering instead from analyse-temp')
            recover_slow_points_from_temp(config, checkpoint_file)
        os.rmdir(f'{checkpoint_dir}/analyse-temp')
    
    else:
        net = RNN(config)
        net.load_state_dict(checkpoint['rnn_state_dict'])

        find_slow_points(config, net, checkpoint_path=checkpoint_file, num_x_0=num_x_0, verbose=True)

