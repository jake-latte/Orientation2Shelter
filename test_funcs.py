import torch

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

from typing import List, Dict

from PIL import Image
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, MDS

from net import *
from data import *
from config import *
from task import Task




############################################################################################################################################
################################################################## TESTING #################################################################
############################################################################################################################################

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# Testing functions for appraisal of tuning and geometry and performance of trained RNNs                                                   #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


'''
test_tuning
Battery of tuning-related plots for path- and head-direction-integration tasks
---------------------------------------------------------------------------------------------
Receives
    task :
        Task on which net was trained
    net :
        Trained RNN to be test
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
def test_tuning(task: Task, net: RNN, batch: dict, tuning_vars_list: list, checkpoint_path:str = None, **kwargs) -> Dict[str, matplotlib.figure.Figure]:

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


    _, activity, outputs = net(batch['inputs'], noise=batch['noise'])

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
        Trained RNN to be test
    batch :
        Data batch for testing with
    checkpoint_path :
        Path to checkpoint where plots should be saved

Returns
    dict :
        Values are matplotlib figure objects containing the plots generated
'''
def test_general(task: Task, net: RNN, batch: dict, checkpoint_path: str = None, **kwargs) -> Dict[str, matplotlib.figure.Figure]:

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
        Trained RNN to be test
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
def get_tuning_generalised(task: Task, inputs: np.ndarray, targets: np.ndarray, vars: dict, activity: np.ndarray, tuning_vars_list: List[str]) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
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

    if 'ego_SD' in tuning_vars_list and 'sin_sd' in task.target_map:
        
        ego_SD = vars['sd'] * 180/np.pi

        angle_bins = np.linspace(0, 360, n_angle_bins+1)[:-1]

        tuning_vars['ego_SD'] = dict(var=ego_SD, bins=angle_bins, title='Head-Shelter Angle')
    
    if 'allo_SD' in tuning_vars_list and 'sin_sd' in task.target_map:

        HD = vars['hd'] * 180/np.pi
        ego_SD = vars['sd'] * 180/np.pi
        allo_SD = np.remainder(HD + ego_SD, 360)

        angle_bins = np.linspace(0, 360, n_angle_bins+1)[:-1]

        tuning_vars['allo_SD'] = dict(var=allo_SD, bins=angle_bins, title='Absolute Shelter Angle')

    if 'x' in tuning_vars_list:

        X = vars['x']

        position_bins = np.linspace(-1, 1, n_position_bins)

        tuning_vars['x'] = dict(var=X, bins=position_bins, title='X Position')

    if 'y' in tuning_vars_list:

        Y = vars['y']

        position_bins = np.linspace(-1, 1, n_position_bins)

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

    print('\t\tComputing tuning curves')

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

            print(f'\t\t\tCompleted {i_key}' + (f' and {j_key}' if j>0 else ''))

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
def univar_tuning_plot(task: Task, tuning_vars: List[str], tuning_dict: Dict[str, np.ndarray], vars: List[Tuple[str, str]], ordering: List[int] = None, bin_mask: np.ndarray = None, **kwargs) -> matplotlib.figure.Figure:

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
def bivar_tuning_plot(task: Task, tuning_vars: List[str], tuning_dict: Dict[str, np.ndarray], vars: Tuple[str, str], ordering: List[int] = None, x_mask: np.ndarray = None, y_mask: np.ndarray = None, **kwargs) -> matplotlib.figure.Figure:
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
def manifold_plot(task: Task, activity: np.ndarray, tuning_vars: List[str], tuning_dict: Dict[str, np.ndarray], colour_var_name: str, var_threshold: float = 0.9, max_dim: int = 10, cmap: str = 'plasma'):
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
    subfigs = fig.subfigures(2, 1, hspace=0, height_ratios=[1, 3])

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
        Trained RNN to be test
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
def path_integration_plot(task: Task, targets: np.ndarray, outputs: np.ndarray, true_colour: str = '#7776BC', pred_colour: str = '#CDC7E5') -> matplotlib.figure.Figure:

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
def loss_plot(task: Task, test_losses: List[float], train_losses: List[float], **kwargs) -> matplotlib.figure.Figure:
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
        numpy.ndarray of RNN outputs corresponding to targets (result of passing test dataset through
        net)
    kwargs :
        kwargs which can override config parameters

Returns
    matplotlib.figure.Figure :
        Figure containing plot

'''
def fit_examples_plot(task: Task, targets: np.ndarray, outputs: np.ndarray, **kwargs) -> matplotlib.figure.Figure:
    
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
Creates a figure identifying the different classifications of RNN units based on tuning profiles
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
        RNN object
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
        RNN object
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
    # The RNN is subjected to these three inputs multiple times, with each time involving a different 'lesion'
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






