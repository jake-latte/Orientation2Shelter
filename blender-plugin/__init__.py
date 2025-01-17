# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTIBILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

bl_info = {
    "name": "Rnn Dynamical Analysis",
    "author": "Jake Laherty",
    "description": "",
    "blender": (2, 80, 0),
    "version": (0, 0, 1),
    "location": "",
    "warning": "",
    "category": "Generic",
}


import bpy
import sys
import torch
import os
import pandas as pd

sys.path.append('/Users/jakelaherty/Library/Mobile Documents/com~apple~CloudDocs/Uni/Gatsby/CuevaCampagner')
from task import *
from data import *
from net import *
from analyse import *

from scipy.linalg import eig

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, MDS

import threading as mt

import queue

import matplotlib as mpl


##################################################################################################################################
######################################################## PROPERTIES ##############################################################
##################################################################################################################################


reduction_techniques = [
    ('pca', 'PCA', 'Principal Component Analysis', 0),
    ('tsne', 'tSNE', 't-Distributed Stochastic Neighbour Embedding', 1),
    ('mds', 'MDS', 'Multi-Dimensional Scaling', 2),
    ('pca-tsne', 'PCA + tSNE', '', 3),
    ('pca-mds', 'PCA + MDS', '', 4)
]

tuning_targets = [
    ('q', 'Kinetic Energy of Network State', '', 0),
    ('hd', 'Head Direction', '', 1),
    ('sd', 'Shelter Direction', '', 2),
]


class RNN_Properties(bpy.types.PropertyGroup):
    checkpoint_selected: bpy.props.BoolProperty(name='Checkpoint Selected', default=False)
    checkpoint_file: bpy.props.StringProperty(subtype="FILE_PATH", name='Checkpoint File')
    checkpoint_dir: bpy.props.StringProperty(subtype="DIR_PATH", name='Checkpoint Directory')

    reduction_technique: bpy.props.EnumProperty(items=reduction_techniques, name='Dimension Reduction Technique', default=0)
    technique_used: bpy.props.StringProperty(name='Technique Used for Current Instance')
    n_pca_components: bpy.props.IntProperty(name='Number of PCA Components to Keep', default=10, min=3, soft_max=25)
    n_nl_reduction_samples: bpy.props.IntProperty(name='Number of Samples for Non-Linear Dimension Reduction', default=25, min=1)
    n_nl_components: bpy.props.IntProperty(name='Number of Components to Compute for Non-Linear Dimension Redcution', default=3, min=3)
    max_nl_iter: bpy.props.IntProperty(name='Maximum Number of Iterations to Use for Non-Linear Dimensions Reduction', default=500, min=100)
    tsne_perplexity: bpy.props.IntProperty(name='Perplexity Parameter for MDS', default=30, min=5)
    mds_n_init: bpy.props.IntProperty(name='Number of Initialisations to Try for MDS', default=4, min=1)

    init_id: bpy.props.IntProperty(name="Initialisation ID")
    is_initialising: bpy.props.BoolProperty(name='Is Initialising', default=False)
    init_progress: bpy.props.IntProperty(name="Initialisation Progress", default=0)
    initialised: bpy.props.BoolProperty(name='Analysis Initialised', default=False)

    n_operative_trajectories: bpy.props.IntProperty(name='Number of Trajectories', default=10, min=0, max=100)
    visualised_dimensions: bpy.props.IntVectorProperty(name='Dimensions to Visualise', default=[1,2,3])
    new_dimensions: bpy.props.IntVectorProperty(name='New Dimensions to Visualise', default=[1,2,3])

    n_search_inits: bpy.props.IntProperty(name='Number of Inits', default=100, min=1, soft_max=1000) 
    is_searching: bpy.props.BoolProperty(name="Is Currently Searching for Fixed Points", default=False)
    search_progress: bpy.props.IntProperty(name="Fixed Point Search Progress", default=0)

    log10_fixed_threshold: bpy.props.FloatProperty(name='log10 of Fixed Point q Threshold', default=-9, soft_min=-24, max=0)

    cluster_distance: bpy.props.FloatProperty(name='Max Intracluster Distance', default=0.5, min=0, soft_max=1)

    eig_eps: bpy.props.FloatProperty(name='Eigenvalue Epsilon', default=0.1, min=0, soft_max=0.5)
    n_oscillation_points: bpy.props.IntProperty(name='Number of Points for Oscilliation Visual', default=25, min=10, soft_max=100)
    is_decomposing: bpy.props.BoolProperty(name="Is Currently Eigendecomposing Fixed Points", default=False)
    decomp_progress: bpy.props.FloatProperty(name="Fixed Point Eigendecomposition Progress", default=0)

    use_linear_sim: bpy.props.BoolProperty(name="Use Linearised Simulation", default=False)
    sim_start_frame: bpy.props.IntProperty(name='Start', default=0, min=0)
    sim_length: bpy.props.IntProperty(name='Length', default=10, min=1)
    sim_target: bpy.props.PointerProperty(type=bpy.types.Object, name='Simulation Target Object')

    rotate_start_frame: bpy.props.IntProperty(name='Start', default=0, min=0)
    rotate_length: bpy.props.IntProperty(name='Length', default=10, min=1)
    is_rotating: bpy.props.BoolProperty(name="Is Currently Baking Rotation", default=False)
    rotate_progress: bpy.props.FloatProperty(name="Rotation Bake Progress", default=0)

    is_viewing_variance: bpy.props.BoolProperty(name="View Component Variance", default=False)

    tuning_target: bpy.props.EnumProperty(items=tuning_targets, name='Tuning Target', default=0)



##################################################################################################################################
######################################################### OPERATORS ##############################################################
##################################################################################################################################



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Select Checkpoint ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

class OBJECT_OT_rnn_filebrowser(bpy.types.Operator):
    bl_idname = "rnn.open_filebrowser"
    bl_label = "Select"

    filepath: bpy.props.StringProperty(subtype="FILE_PATH")

    @classmethod
    def poll(cls, context):
        try:
            props = context.scene.rnn_props
            return True
        except:
            return False

    def execute(self, context): 
        props = context.scene.rnn_props

        props.initialised = False

        props.checkpoint_file = self.filepath
        props.checkpoint_dir = '/'.join(self.filepath.split('/')[:-1])
        props.checkpoint_selected = True

        return {'FINISHED'}

    def invoke(self, context, event):

        self.filepath = '/Users/jakelaherty/Library/Mobile Documents/com~apple~CloudDocs/Uni/Gatsby/CuevaCampagner/trained-models/'
        context.window_manager.fileselect_add(self) 

        return {'RUNNING_MODAL'} 



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Initialise ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

class OBJECT_OT_rnn_initialise(bpy.types.Operator):
    bl_idname = "rnn.initialise"
    bl_label = "Initialise"

    @classmethod
    def poll(cls, context):
        try:
            props = context.scene.rnn_props
            return props.checkpoint_selected
        except:
            return False

    def execute(self, context):
        global init_process

        props = context.scene.rnn_props

        if not props.is_initialising:
            props.init_id = random.randint(0, 2**16 - 1)

            init_process = mt.Thread(target=initialise, args=(props.init_id,), daemon=True)
           
            props.is_initialising = True
            bpy.types.Scene.new_init_objects = {}
            props.init_progress = 0
            init_process.start()

            bpy.app.timers.register(init_timer_func)

            return {'RUNNING_MODAL'}

        else:

            init_process = None
            props.is_initialising = False

            return {'FINISHED'}


    
def initialise(thread_init_id):
    props = bpy.context.scene.rnn_props

    if props.is_initialising and props.init_id == thread_init_id:
        print('loading')
        load_result = load_checkpoint()
        load_result['id'] = thread_init_id
        init_queue.put(load_result)

    if props.is_initialising and props.init_id == thread_init_id:
        print('creating')
        create_result = create_data(load_result)
        create_result['id'] = thread_init_id
        init_queue.put(create_result)

    if props.is_initialising and props.init_id == thread_init_id:
        print('reducing')
        reduce_result = dimension_reduce(create_result)
        reduce_result['id'] = thread_init_id
        init_queue.put(reduce_result)
        
    print('done')

init_queue = queue.Queue()
init_process = None
def init_timer_func():
    try:
        props = bpy.context.scene.rnn_props

        while not init_queue.empty():

            obj_dict = init_queue.get()

            if obj_dict['id'] != props.init_id:
                continue

            for name, obj in obj_dict.items():
                bpy.types.Scene.new_init_objects[name] = obj

            props.init_progress += 1
                    
        if props.is_initialising:
            global init_process
            
            if props.init_progress == 3:

                for obj in bpy.data.objects:
                    bpy.data.objects.remove(obj, do_unlink=True)

                for collection in bpy.data.collections:
                    bpy.data.collections.remove(collection)
                
                new_init = bpy.context.scene.new_init_objects

                bpy.types.Scene.checkpoint = new_init['checkpoint']
                bpy.types.Scene.task = new_init['task']
                bpy.types.Scene.net = new_init['net']

                bpy.types.Scene.dataset = new_init['dataset']
                bpy.types.Scene.data = new_init['data']

                bpy.types.Scene.inputs = new_init['inputs']
                bpy.types.Scene.targets = new_init['targets']

                bpy.types.Scene.states = new_init['states']
                bpy.types.Scene.activity = new_init['activity']
                bpy.types.Scene.outputs = new_init['outputs']

                bpy.types.Scene.activity_reduced = new_init['activity_reduced']
                bpy.types.Scene.pca = new_init['pca']
                bpy.types.Scene.activity_pca = new_init['activity_pca']

                bpy.types.Scene.fp_states = None
                bpy.types.Scene.fp_states = None
                bpy.types.Scene.fp_activity = None
                bpy.types.Scene.fp_activity_reduced = None
                bpy.types.Scene.fp_test_input = None
                bpy.types.Scene.fp_true_input = None
                bpy.types.Scene.fp_output = None
                bpy.types.Scene.fp_target = None

                bpy.types.Scene.eig_indices = None
                bpy.types.Scene.eig_states = None
                bpy.types.Scene.eig_activity = None
                bpy.types.Scene.eig_activity_reduced = None

                bpy.types.Scene.fixed_i = None

                bpy.types.Scene.sim_targets = []
                bpy.types.Scene.sim_initial_states = []
                bpy.types.Scene.sim_initial_activity = []
                bpy.types.Scene.sim_initial_activity_reduced = []
                bpy.types.Scene.sim_keyframed_activity = []
                bpy.types.Scene.sim_keyframed_activity_reduced = []

                bpy.types.Scene.operative_trajectory_indices = None

                props.is_searching = False
                props.is_decomposing = False
                props.technique_used = props.reduction_technique
                # props.visualised_dimensions.max = new_init['activity_reduced'].shape[2]-1

                props.is_initialising = False
                props.initialised = True
                init_process.join()
                bpy.app.timers.unregister(init_timer_func)

        else:

            bpy.app.timers.unregister(init_timer_func)

            

    except Exception as e:
        print(e)

    return 0.1


def load_checkpoint():
    context = bpy.context

    checkpoint_file = context.scene.rnn_props.checkpoint_file
    checkpoint = torch.load(checkpoint_file, map_location='cpu')

    task = Task.from_checkpoint(checkpoint)
    task.config.update(device='cpu', test_batch_size=100, test_n_timesteps=500)

    net = RNN(task.config)
    net.load_state_dict(checkpoint['net_state_dict'])

    np.random.seed(task.config.build_seed)
    torch.manual_seed(task.config.build_seed)

    return {
        'checkpoint': checkpoint,
        'task': task,
        'net': net
    }


def create_data(new_init_objects):

    dataset = TaskDataset(new_init_objects['task'], for_training=False)
    data = dataset.get_batch()

    net = new_init_objects['net']
    states, activity, outputs = net(data['inputs'], noise=data['noise'])

    inputs, targets = data['inputs'].numpy(), data['targets'].numpy()
    states, activity, outputs = states.detach().numpy(), activity.detach().numpy(), outputs.detach().numpy()

    return {
        'dataset': dataset,
        'data': data,
        'inputs': inputs,
        'targets': targets,
        'states': states,
        'activity': activity,
        'outputs': outputs
    }


def dimension_reduce(new_init_objects):
    context = bpy.context

    props = context.scene.rnn_props
    n_pca_components = props.n_pca_components
    n_nl_reduction_samples = props.n_nl_reduction_samples
    n_nl_components = props.n_nl_components
    max_nl_iter = props.max_nl_iter
    tsne_perplexity, mds_n_init = props.tsne_perplexity, props.mds_n_init

    activity = new_init_objects['activity']
    n_trials, n_timesteps, n_neurons = activity.shape[0], activity.shape[1], activity.shape[2]

    result = {}
    if 'pca' in props.reduction_technique:

        pca = PCA(n_components=n_pca_components)

        activity = pca.fit_transform(activity.reshape(-1, n_neurons))
        activity = activity.reshape((n_trials, n_timesteps, n_pca_components))

        result['pca'] = pca
        result['activity_pca'] = activity
        result['activity_reduced'] = activity

    if 'pca' != props.reduction_technique:
        n_nl_reduction_samples = min(n_nl_reduction_samples, activity.shape[0]*activity.shape[1])
        if 'pca-' in props.reduction_technique:
            n_nl_components = min(n_nl_components, n_pca_components)

        sample_i = np.random.permutation(n_trials)[:n_nl_reduction_samples]
        activity = activity[sample_i]
        activity = activity.reshape((-1, activity.shape[2]))

        if 'tsne' in props.reduction_technique:
            technique = TSNE(n_components=n_nl_components, perplexity=tsne_perplexity, max_iter=max_nl_iter, verbose=1)
        elif 'mds' in props.reduction_technique:
            technique = MDS(n_components=n_nl_components, n_init=mds_n_init, max_iter=max_nl_iter, verbose=1, n_jobs=-1)
        
        activity = technique.fit_transform(activity)
        activity = activity.reshape((n_nl_reduction_samples, n_timesteps, n_nl_components))

        result['activity_reduced'] = activity
        result['inputs'] = context.scene.inputs[sample_i]
        result['targets'] = context.scene.inputs[sample_i]
        result['states'] = context.scene.inputs[sample_i]
        result['activity'] = context.scene.inputs[sample_i]
        result['output'] = context.scene.inputs[sample_i]
    
    return result

        

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Plot Trajectories ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
    
class OBJECT_OT_rnn_plot_operative_trajectories(bpy.types.Operator):
    bl_idname = "rnn.plot_operative_trajectories"
    bl_label = "Generate"

    @classmethod
    def poll(cls, context):
        try:
            props = context.scene.rnn_props
            return props.initialised
        except:
            return False

    def execute(self, context):
        props = context.scene.rnn_props
        n_trajectories = props.n_operative_trajectories

        activity_reduced = context.scene.activity_reduced
        n_trials, n_timesteps = activity_reduced.shape[0], activity_reduced.shape[1]
        n_trajectories = min(n_trajectories, n_trials)

        collection_name = 'Operative Trajectories'
        if bpy.data.collections.get(collection_name):
            trajectory_collection = bpy.data.collections.get(collection_name)
            for obj in trajectory_collection.objects:
                bpy.data.objects.remove(obj, do_unlink=True)
        else:
            trajectory_collection = bpy.data.collections.new(collection_name)
            bpy.context.scene.collection.children.link(trajectory_collection)

        example_trajectories_i = np.random.permutation(n_trials)[:n_trajectories]
        bpy.types.Scene.operative_trajectory_indices = example_trajectories_i
        trajectories = activity_reduced[example_trajectories_i]

        for i, traj in enumerate(trajectories):
            name = f'ot_{example_trajectories_i[i]}'
            vertices = []
            for t in range(n_timesteps):
                vertex_vis = get_vis_dims(traj[t])
                vertices.append((vertex_vis[0], vertex_vis[1], vertex_vis[2]))
            edges = [(t, t+1) for t in range(n_timesteps-1)]
            faces = []
            mesh = bpy.data.meshes.new(name)
            mesh.from_pydata(vertices, edges, faces)
            mesh.update()
            obj = bpy.data.objects.new(name, mesh)
            trajectory_collection.objects.link(obj) 
            

        return {'FINISHED'} 
    


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Show Fixed Points ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

class OBJECT_OT_show_fixed_points(bpy.types.Operator):
    bl_idname = "rnn.show_fixed_points"
    bl_label = "Show"


    @classmethod
    def poll(cls, context):
        try:
            props = context.scene.rnn_props
            has_fps = 'fixed_points.pt' in os.listdir(props.checkpoint_dir)
            return props.initialised and has_fps
        except:
            return False

    def execute(self, context):
        show_fixed_points()

        return {'FINISHED'} 


def show_fixed_points(clustered=False):
    context = bpy.context

    props = context.scene.rnn_props
    checkpoint_dir, checkpoint_file = props.checkpoint_dir, props.checkpoint_file
    fixed_threshold = 10**(props.log10_fixed_threshold)

    task, net, pca = context.scene.task, context.scene.net, context.scene.pca

    if os.path.isdir(f'{checkpoint_dir}/analyse-temp'):
        if len(os.listdir(f'{checkpoint_dir}/analyse-temp'))>0:
            print('Recovering instead from analyse-temp')
            recover_fixed_points_from_temp(task.config, checkpoint_file)

    fixed_point_dict = torch.load(f'{checkpoint_dir}/fixed_points.pt', map_location='cpu')
    
    q = np.array(fixed_point_dict['q'])
    fp_states = np.array([x.numpy() for x in fixed_point_dict['state']])
    fp_activity = np.array([net.activation_func(x).numpy() for x in fixed_point_dict['state']])
    fp_activity_reduced = pca.transform(fp_activity)

    fp_test_input = np.array([x.numpy() for x in fixed_point_dict['test_input']])
    fp_true_input = np.array([x.numpy() for x in fixed_point_dict['true_input']])
    fp_output = np.array([x.numpy() for x in fixed_point_dict['output']])
    fp_target = np.array([x.numpy() for x in fixed_point_dict['target']])

    if clustered:
        fixed_i = context.scene.fixed_i
    else:
        fixed_i = np.where((q <= fixed_threshold))[0]


    collection_name = 'Clustered Fixed Points' if clustered else 'Fixed Points'
    if bpy.data.collections.get(collection_name):
        fp_collection = bpy.data.collections.get(collection_name)
        fp_collection.hide_viewport = False

        for obj in fp_collection.objects:
            bpy.data.objects.remove(obj, do_unlink=True)
    else:
        fp_collection = bpy.data.collections.new(collection_name)
        bpy.context.scene.collection.children.link(fp_collection)

    other_collection_name = 'Clustered Fixed Points' if not clustered else 'Fixed Points'
    if bpy.data.collections.get(other_collection_name):
        other_collection = bpy.data.collections.get(other_collection_name)
        other_collection.hide_viewport = True

    for i, fp in enumerate(fp_activity_reduced[fixed_i]):
        name = f'fp_{fixed_i[i]}'
        if clustered:
            name += '-cluster_medioid'
        fp_vis = get_vis_dims(fp)
        bpy.ops.mesh.primitive_uv_sphere_add(
            segments=6, ring_count=6, radius=0.025, location=(fp_vis[0], fp_vis[1], fp_vis[2])
        )
        obj = bpy.data.objects['Sphere']
        obj.name = name
        context.scene.collection.objects.unlink(obj)
        fp_collection.objects.link(obj)

    bpy.types.Scene.fixed_i = fixed_i
    bpy.types.Scene.fp_q = q
    bpy.types.Scene.fp_states = fp_states
    bpy.types.Scene.fp_activity = fp_activity
    bpy.types.Scene.fp_activity_reduced = fp_activity_reduced

    bpy.types.Scene.fp_test_input = fp_test_input
    bpy.types.Scene.fp_true_input = fp_true_input
    bpy.types.Scene.fp_output = fp_output
    bpy.types.Scene.fp_target = fp_target


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Find Fixed Points ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

class OBJECT_OT_find_fixed_points(bpy.types.Operator):
    bl_idname = "rnn.find_fixed_points"
    bl_label = "Find"

    @classmethod
    def poll(cls, context):
        return False
        try:
            props = context.scene.rnn_props
            return props.initialised
        except:
            return False
    
    def execute(self, context):
        global search_process
        props = context.scene.rnn_props

        if not props.is_searching:

            checkpoint_file = props.checkpoint_file
            n_search_inits = props.n_search_inits

            task, net = context.scene.task, context.scene.net

            search_process = mt.Thread(target=find_fixed_points, args=(task, net, checkpoint_file, n_search_inits, True), daemon=True)
           
            props.is_searching = True
            props.search_progress = 0
            search_process.start()

            bpy.app.timers.register(search_timer_func)

            return {'RUNNING_MODAL'}

        else:

            props.is_searching = False
            search_process.join()
            search_timer_func()

            return {'FINISHED'}

def find_fixed_points(task, net, checkpoint_path, num_x_0, verbose):
    global search_queue

    context = bpy.context

    props = context.scene.rnn_props
    checkpoint_dir = props.checkpoint_dir

    if not os.path.exists(f'{checkpoint_dir}/analyse-temp'):
        os.mkdir(f'{checkpoint_dir}/analyse-temp')
        
    probe = context.scene.data

    states,_,outputs = net(probe['inputs'], noise=probe['noise'])

    inputs, targets = probe['inputs'].detach().cpu().numpy(), probe['targets'].detach().cpu().numpy()
    states, outputs = states.detach().cpu().numpy(), outputs.detach().cpu().numpy()

    ###############################
    test_inputs = np.zeros_like(inputs)
    ###############################


    W_rec, W_in, b = net.W_rec.weight.detach().cpu().numpy(), net.W_in.weight.detach().cpu().numpy(), net.W_in.bias.detach().cpu().numpy()

    X = states.reshape(-1, task.config.n_neurons)
    U_true = inputs.reshape(-1, task.config.n_inputs)
    U_test = test_inputs.reshape(-1, task.config.n_inputs)
    Z_pred = outputs.reshape(-1, task.config.n_outputs)
    Z_true = targets.reshape(-1, task.config.n_outputs)

    random_i = np.random.permutation(states.shape[0]*states.shape[1])[:num_x_0]
    X, U_true, U_test, Z_pred, Z_true = X[random_i], U_true[random_i], U_test[random_i], Z_pred[random_i], Z_true[random_i]

    for i in range(num_x_0):
        if not props.is_searching:
            break
        q, x = minimise_from_x_0_single(i, 
                                        X[i], U_test[i], U_true[i], Z_pred[i], Z_true[i],
                                        W_rec, W_in, b, checkpoint_dir, verbose)
        recover_fixed_points_from_temp(task.config, checkpoint_path)

        search_queue.put((i, q, x))

search_queue = queue.Queue()
search_process = None
def search_timer_func():
    if not search_queue.empty():
        global search_process

        props = bpy.context.scene.rnn_props
        fixed_threshold = 10**(props.log10_fixed_threshold)

        pca = bpy.context.scene.pca

        while not search_queue.empty():
            i, q, x = search_queue.get()

            if q < fixed_threshold:
                collection_name = 'Fixed Points'
                if bpy.data.collections.get(collection_name):
                    fp_collection = bpy.data.collections.get(collection_name)
                    fp_collection.hide_viewport = False
                else:
                    fp_collection = bpy.data.collections.new(collection_name)
                    bpy.context.scene.collection.children.link(fp_collection)

                x = pca.transform(x.reshape(1, -1)).reshape((-1,))
                
                name = f'fp-new_{i}'
                x_vis = get_vis_dims(x)
                vertices = [(x_vis[0], x_vis[1], x_vis[2])]
                edges = []
                faces = []
                mesh = bpy.data.meshes.new(name)
                mesh.from_pydata(vertices, edges, faces)
                mesh.update()
                obj = bpy.data.objects.new(name, mesh)
                fp_collection.objects.link(obj) 

            props.search_progress = i+1
            
        if not props.is_searching or i == props.n_search_inits-1:

            if i == props.n_search_inits-1:
                props.is_searching = False
                search_process.join()
            else:
                search_timer_func()

            show_fixed_points(bpy.context)
            bpy.app.timers.unregister(decomp_timer_func)

    return 1

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Cluster Fixed Points ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

class OBJECT_OT_cluster_fixed_points(bpy.types.Operator):
    bl_idname = "rnn.cluster_fixed_points"
    bl_label = "Cluster"

    @classmethod
    def poll(cls, context):
        try:
            props = context.scene.rnn_props
            has_fps = context.scene.fp_q is not None
            return props.initialised and has_fps
        except:
            return False
    
    def execute(self, context):
        cluster_fixed_points()

        return {'FINISHED'}

def cluster_fixed_points():
    context = bpy.context

    props = context.scene.rnn_props
    checkpoint_dir = props.checkpoint_dir
    fixed_threshold = 10**(props.log10_fixed_threshold)
    cluster_distance = props.cluster_distance

    q = context.scene.fp_q
    fp_activity = context.scene.fp_activity
    fp_activity_reduced = context.scene.fp_activity_reduced

    fixed_i = np.where((q <= fixed_threshold))[0]

    distance = np.zeros((len(fixed_i), len(fixed_i)))
    clusters = []
    for i, fp_i in enumerate(fixed_i):
        fp_i_cluster_i = None
        for cluster_i, cluster in enumerate(clusters):
            if fp_i in cluster:
                fp_i_cluster_i = cluster_i
        if fp_i_cluster_i is None:
            clusters.append([fp_i])
            fp_i_cluster_i = -1

        for j, fp_j in enumerate(fixed_i):
            distance[i,j] = np.sqrt(np.sum((fp_activity_reduced[fp_i] - fp_activity_reduced[fp_j])**2))

            if j<=i: 
                continue

            if distance[i,j] <= cluster_distance:
                clusters[fp_i_cluster_i].append(fp_j)

    unique_fixed_i = np.zeros((len(clusters),), dtype=fixed_i.dtype)
    for cluster_i, cluster in enumerate(clusters):
        cluster_fps = fp_activity_reduced[cluster]
        centroid = np.mean(cluster_fps, axis=0)
        cluster_distances_to_centroid = np.sqrt(np.sum((cluster_fps - np.tile(centroid, (len(cluster), 1)))**2, axis=1))
        medioid_i = cluster[np.argmin(cluster_distances_to_centroid)]
        unique_fixed_i[cluster_i] = medioid_i

    bpy.types.Scene.fixed_i = unique_fixed_i

    show_fixed_points(clustered=True)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Show Eigendecomposition ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

class OBJECT_OT_decompose_fixed_points(bpy.types.Operator):
    bl_idname = "rnn.decompose_fixed_points"
    bl_label = "Generate Eigenmodes"

    @classmethod
    def poll(cls, context):
        try:
            props = context.scene.rnn_props
            fps = bpy.data.collections.get('Fixed Points')
            cluster_fps = bpy.data.collections.get('Clustered Fixed Points')
            n_fps = 0 if fps is None else len(fps.objects)
            n_cluster_fps = 0 if cluster_fps is None else len(cluster_fps.objects)
            return props.initialised and (n_fps>0 or n_cluster_fps>0)
        except:
            return False
    
    def execute(self, context):
        global decomp_process

        props = context.scene.rnn_props

        if not props.is_decomposing:
            bpy.types.Scene.eig_indices = None
            bpy.types.Scene.eig_states = None
            bpy.types.Scene.eig_activity = None
            bpy.types.Scene.eig_activity_reduced = None
            
            if 'Eigenmodes' in bpy.data.collections:
                parent = bpy.data.collections.get('Eigenmodes')
                for subcollection in parent.children_recursive:
                    for obj in subcollection.objects:
                        bpy.data.objects.remove(obj, do_unlink=True)
            else:
                parent = bpy.data.collections.new('Eigenmodes')
                bpy.context.scene.collection.children.link(parent)

            mode_types = ['integrative', 'unstable', 'stable', 'periodic oscillatory', 'unstable oscillatory', 'stable oscillatory']
            for type in mode_types:
                collection_name = f'{type.capitalize()}'
                collection = bpy.data.collections.get(collection_name)
                if collection is not None:
                    for obj in collection.objects:
                        bpy.data.objects.remove(obj, do_unlink=True)
                else:
                    collection = bpy.data.collections.new(collection_name)
                    parent.children.link(collection)

            decomp_process = mt.Thread(target=decompose_fixed_points, daemon=True)

            props.is_decomposing = True
            props.decomp_progress = 0
            decomp_process.start()

            bpy.app.timers.register(decomp_timer_func)

            return {'RUNNING_MODAL'}

        else:

            props.is_decomposing = False
            decomp_process.join()
            while not decomp_queue.empty():
                decomp_timer_func()


            return {'FINISHED'}

def decompose_fixed_points():
    global decomp_queue

    context = bpy.context
    
    props = context.scene.rnn_props
    checkpoint_dir = props.checkpoint_dir
    fixed_threshold = 10**(props.log10_fixed_threshold)
    eig_eps = props.eig_eps
    n_oscillation_points = props.n_oscillation_points

    pca = context.scene.pca
    q = context.scene.fp_q
    fp_states = context.scene.fp_states
    fp_activity = context.scene.fp_activity
    fp_activity_reduced = context.scene.fp_activity_reduced

    fixed_i = context.scene.fixed_i

    n_total = len(fixed_i) * net.n_neurons
    n_seen = 0
    for k in fixed_i:
        if not props.is_decomposing:
            break

        J = get_Jacobian_at(fp_states[k], net)
        E, L, R = eig(J, left=True, right=True)

        for e_i, eigenvalue in enumerate(E):
            if not props.is_decomposing:
                break

            fp, v = fp_states[k], R[:,e_i]
            type, state_mode = None, None
            if np.abs(eigenvalue.imag) < eig_eps:
                eigenvalue = eigenvalue.real
                
                if  np.abs(eigenvalue.real) < eig_eps:
                    type='integrative'
                elif eigenvalue.real > eig_eps:
                    type='unstable'
                elif eigenvalue.real < -eig_eps:
                    type='stable'

                state_mode = np.array([fp, fp + v.real])

            else:
                z, w = eigenvalue.real, eigenvalue.imag

                if  np.abs(z) < eig_eps:
                    type='periodic oscillatory'
                elif z > eig_eps:
                    type='unstable oscillatory'
                elif z < -eig_eps:
                    type='stable oscillatory'
                
                t_span = np.linspace(0, 2*np.pi/np.abs(w), n_oscillation_points)

                state_mode = np.zeros((n_oscillation_points, net.n_neurons))
                for t_i, t in enumerate(t_span):
                    state_mode[t_i] = fp + (1 * ((np.cos(w*t) + (1j)*np.sin(w*t)) * v)).real

            rate_mode = net.activation_func(torch.tensor(state_mode)).numpy()
            pca_mode = pca.transform(rate_mode)

            n_seen += 1
            mode_vis = get_vis_dims(pca_mode)
            decomp_queue.put((mode_vis[:,0], mode_vis[:,1], mode_vis[:,2],
                              type, k, e_i, n_seen / n_total,
                              state_mode, rate_mode, pca_mode))

decomp_queue = queue.Queue()
decomp_process = None
def decomp_timer_func():
    try:
        if not decomp_queue.empty():
            global decomp_process

            props = bpy.context.scene.rnn_props

            q_size = decomp_queue.qsize()
            for _ in range(q_size):
                x, y, z, t, k, e_i, p, states, rates, rates_reduced = decomp_queue.get()

                collection = bpy.data.collections.get(f'{t.capitalize()}')

                name = f'fp_{k}-eigenvalue_{e_i}'
                vertices = [(x[j], y[j], z[j]) for j in range(len(x))]
                edges = [(j, j+1) for j in range(len(x)-1)]
                faces = []
                mesh = bpy.data.meshes.new(name)
                mesh.from_pydata(vertices, edges, faces)
                mesh.update()
                obj = bpy.data.objects.new(name, mesh)
                collection.objects.link(obj) 

                props.decomp_progress = p

                eig_indices, eig_states, eig_activity, eig_activity_reduced = bpy.context.scene.eig_indices, bpy.context.scene.eig_states, bpy.context.scene.eig_activity, bpy.context.scene.eig_activity_reduced
                indices = np.tile(np.array([k, e_i]).reshape((1,2)), reps=(states.shape[0],1))
                if eig_indices is None:
                    bpy.types.Scene.eig_indices = indices
                    bpy.types.Scene.eig_states = states
                    bpy.types.Scene.eig_activity = rates
                    bpy.types.Scene.eig_activity_reduced = rates_reduced
                else:
                    bpy.types.Scene.eig_indices = np.concatenate((eig_indices.reshape((-1,2)), indices), axis=0)
                    bpy.types.Scene.eig_states = np.concatenate((eig_states, states), axis=0)
                    bpy.types.Scene.eig_activity = np.concatenate((eig_activity, rates), axis=0)
                    bpy.types.Scene.eig_activity_reduced = np.concatenate((eig_activity_reduced, rates_reduced), axis=0)
                    
                if p == 1:

                    props.is_decomposing = False
                    decomp_process.join()
                    decomp_timer_func()

                    bpy.app.timers.unregister(decomp_timer_func)

    except Exception as e:
        print(e)

    return 0.1


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Tune Fixed Points ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #



class OBJECT_OT_tune_fixed_points(bpy.types.Operator):
    bl_idname = "rnn.tune_fixed_points"
    bl_label = "Tune"

    @classmethod
    def poll(cls, context):
        try:
            props = context.scene.rnn_props
            has_fps = context.scene.fp_q is not None
            return props.initialised and has_fps
        except:
            return False
    
    def execute(self, context):
        tune_fixed_points()

        return {'FINISHED'}

def tune_fixed_points():
    context = bpy.context

    props = context.scene.rnn_props
    target = props.tuning_target

    task = context.scene.task

    target_var = None
    if 'q' == target:
        target_var = context.scene.fp_q
    else:
        net = bpy.context.scene.net
        W_out, b = net.W_out.weight.detach().numpy(), net.W_out.bias.detach().numpy()
        fp_out = []
        for activity in bpy.context.scene.fp_activity:
            activity = activity.reshape((-1, 1))
            output = W_out @ activity + b.reshape((-1,1))
            fp_out.append(output.squeeze())
        fp_out = np.array(fp_out)

        if 'hd' == target:
            if 'ego' not in task.name and 'allo' not in task.name:
                raise ValueError('Invalid tuning target for current model')
            target_var = np.arctan2(
                fp_out[:,0], fp_out[:,1]
            )
        elif 'sd' == target:
            if 'ego' not in task.name and 'allo' not in task.name:
                raise ValueError('Invalid tuning target for current model')
            target_var = np.arctan2(
                fp_out[:,2], fp_out[:,3]
            )

    cmap = mpl.colormaps['inferno']
    norm = mpl.colors.Normalize(vmin=np.min(target_var), vmax=np.max(target_var))

    fp_colors = []
    for v in target_var:
        fp_colors.append(cmap(norm(v)))

    fp_collection = bpy.data.collections.get('Fixed Points')
    cluster_fp_collection = bpy.data.collections.get('Clustered Fixed Points')

    if cluster_fp_collection is not None and not cluster_fp_collection.hide_viewport:
        fp_collection = cluster_fp_collection

    for obj in fp_collection.objects:
        i = int(obj.name.split('-')[0].split('_')[1])

        if not obj.type == 'MESH':
            continue

        # obj.color = fp_colors[i]

        tune_mat = obj.active_material
        # for mat in obj.data.material_slots:
        #     if mat and mat.name != f'{obj.name}-mat':
        #         # bpy.data.materials.remove(mat, do_unlink=True)
        #         mat = None
        #     elif mat.name == f'{obj.name}-mat':
        #         tune_mat = mat
        
        if tune_mat is None:
            tune_mat = bpy.data.materials.new(f'{obj.name}-{target}')
            obj.data.materials.append(tune_mat)
            tune_mat.use_nodes = True

        tune_mat.name = f'{obj.name}-{target}'
        tune_mat.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = fp_colors[i]



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Get State ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

class OBJECT_OT_get_state(bpy.types.Operator):
    bl_idname = "rnn.get_state"
    bl_label = "Get State"

    @classmethod
    def poll(cls, context):
        try:
            props = context.scene.rnn_props
            return props.initialised
        except:
            return False
    
    def execute(self, context):
        create_sim_target()

        return {'FINISHED'}


def create_sim_target():
    context = bpy.context

    props = context.scene.rnn_props

    activity_reduced, fp_activity_reduced, eig_activity_reduced = context.scene.activity_reduced, context.scene.fp_activity_reduced, context.scene.eig_activity_reduced

    n_dimensions = activity_reduced.shape[2]
    activity_reduced = activity_reduced.reshape(-1, n_dimensions)

    activity_to_include = []
    for a_r in [activity_reduced, fp_activity_reduced, eig_activity_reduced]:
        if a_r is not None:
            activity_to_include.append(a_r)
    all_activity_reduced = np.concatenate(activity_to_include, axis=0)

    states, fp_states, eig_states = context.scene.states, context.scene.fp_states, context.scene.eig_states

    n_neurons = states.shape[2]
    states = states.reshape(-1, n_neurons)

    states_to_include = []
    for s in [states, fp_states, eig_states]:
        if s is not None:
            states_to_include.append(s)
    all_states = np.concatenate(states_to_include, axis=0)

    activity, fp_activity, eig_activity = context.scene.activity, context.scene.fp_activity, context.scene.eig_activity

    n_neurons = activity.shape[2]
    activity = activity.reshape(-1, n_neurons)

    activity_to_include = []
    for a in [activity, fp_activity, eig_activity]:
        if a is not None:
            activity_to_include.append(a)
    all_activity = np.concatenate(activity_to_include, axis=0)



    cursor = context.scene.cursor.location

    all_activity_vis = get_vis_dims(all_activity_reduced)
    a_r = np.stack((all_activity_vis[:,0], all_activity_vis[:,1], all_activity_vis[:,2]), axis=1)
    min_dist_i = np.argmin(np.sqrt(((cursor[0] - a_r[:,0])**2 + (cursor[1] - a_r[:,1])**2 + (cursor[2] - a_r[:,2])**2)))

    closest_state = all_states[min_dist_i]
    closest_activity = all_activity[min_dist_i]
    closest_activity_reduced = all_activity_reduced[min_dist_i]

    cursor_vis = get_vis_dims(closest_activity_reduced)
    cursor_activity = (cursor_vis[0], cursor_vis[1], cursor_vis[2])

    name = f'sim_{len(context.scene.sim_targets)}'
    if name in bpy.data.objects:
        obj = bpy.data.objects[name]
        bpy.data.objects.remove(obj, do_unlink=True)

    if bpy.data.collections.get('Simulations'):
        collection = bpy.data.collections.get('Simulations')
    else:
        collection = bpy.data.collections.new('Simulations')
        bpy.context.scene.collection.children.link(collection)

    bpy.ops.mesh.primitive_uv_sphere_add(
        segments=6, ring_count=6, radius=0.025, location=cursor_activity
        )
    obj = bpy.data.objects['Sphere']
    obj.name = name
    context.scene.collection.objects.unlink(obj)
    collection.objects.link(obj)
    
    props.sim_target = obj
    bpy.types.Scene.sim_targets.append(obj)
    bpy.types.Scene.sim_initial_states.append(closest_state)
    bpy.types.Scene.sim_initial_activity.append(closest_activity)
    bpy.types.Scene.sim_initial_activity_reduced.append(closest_activity_reduced)
    bpy.types.Scene.sim_keyframed_activity.append(None)
    bpy.types.Scene.sim_keyframed_activity_reduced.append(None)

    

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Bake Simulation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

class OBJECT_OT_bake_simulation(bpy.types.Operator):
    bl_idname = "rnn.bake_simulation"
    bl_label = "Bake Simulation"

    step: bpy.props.BoolProperty(name='step', default=False)

    @classmethod
    def poll(cls, context):
        try:
            props = context.scene.rnn_props
            target = props.sim_target
            return props.initialised and target is not None and target in context.scene.sim_targets
        except:
            return False
    
    def execute(self, context):
        if context.scene.fp_states is None:
            context.scene.rnn_props.use_linear_sim = False

        bake_simulation(self.step)

        return {'FINISHED'}

def bake_simulation(step):
    context = bpy.context

    props = context.scene.rnn_props
    start_frame, n_timesteps = props.sim_start_frame, 1 if step else props.sim_length
    obj = props.sim_target

    task, net, pca = context.scene.task, context.scene.net, context.scene.pca

    sim_i = int(obj.name.split('_')[1])
    x_0 = context.scene.sim_initial_states[sim_i]

    inputs = torch.zeros((1, n_timesteps, task.config.n_inputs))
    state_noise = torch.zeros((1, n_timesteps, task.config.n_neurons))
    rate_noise = torch.zeros((1, n_timesteps, task.config.n_neurons))
    output_noise = torch.zeros((1, n_timesteps, task.config.n_outputs))

    states, activity, fps = None, None, None
    if props.use_linear_sim:
        states = np.zeros((n_timesteps+1, net.n_neurons))
        fps = np.zeros((n_timesteps, net.n_neurons))

        states[0] = x_0
        for t in range(n_timesteps):
            x_prev = states[t]
            J, fp_x = get_nearest_Jacobian(context, x_prev)
            fps[t] = fp_x
            x_next = x_prev + (task.config.dt/task.config.tau) * J@x_prev
            states[t+1] = x_next

        activity = net.activation_func(torch.tensor(states[:-1])).numpy()

    else:
        x_0 = torch.tensor(x_0, dtype=net.x_0.dtype)
        
        states, activity, _ = net(inputs, x_0=x_0, noise=(state_noise, rate_noise, output_noise))
        activity = np.concatenate((
            net.activation_func(x_0).numpy().reshape(1,-1), activity.detach().numpy()[0]
        ), axis=0)

    states = states.detach().numpy()
    activity_reduced = pca.transform(activity)

    if step:
        obj_vis = get_vis_dims(activity_reduced[1])
        obj.location = (obj_vis[0], obj_vis[1], obj_vis[2])
    else:
        obj.animation_data_clear()
        for t in range(n_timesteps):
            obj_vis = get_vis_dims(activity_reduced[t])
            obj.location = (obj_vis[0], obj_vis[1], obj_vis[2])
            obj.keyframe_insert(data_path='location', frame=start_frame + t)

    if props.use_linear_sim:

        if f'{obj.name}-fixed_point_tracker' in bpy.data.objects:
            subname = f'{obj.name}-fixed_point_tracker'
            vertices = [(0, 0, 0), (0, 0, 0)]
            edges = [(0, 1)]
            faces = []
            mesh = bpy.data.meshes.new(subname)
            mesh.from_pydata(vertices, edges, faces)
            mesh.update()
            subobj = bpy.data.objects.new(subname, mesh)
            collection = bpy.data.collections.get('Simulations')
            collection.objects.link(subobj)

        subobj = bpy.data.objects[f'{obj.name}-fixed_point_tracker']

        fps_activity = net.activation_func(torch.tensor(fps)).numpy()
        fps_activity_reduced = pca.transform(fps_activity)

        vertices = subobj.data.vertices
        for t in range(n_timesteps):
            v = vertices[0]
            vertex_vis = get_vis_dims(fps_activity_reduced[t])
            vertices[0].co = [vertex_vis[0], vertex_vis[1], vertex_vis[2]]
            vertices[0].keyframe_insert(data_path='co', frame=start_frame + t)
            vertices[1].co = [vertex_vis[0], vertex_vis[1], vertex_vis[2]]
            vertices[1].keyframe_insert(data_path='co', frame=start_frame + t)

    if step:
        bpy.types.Scene.sim_initial_states[sim_i] = states[0]
        bpy.types.Scene.sim_initial_activity[sim_i] = activity[0]
        bpy.types.Scene.sim_initial_activity_reduced[sim_i] = activity_reduced[1]
    else:
        bpy.types.Scene.sim_keyframed_activity[sim_i] = (start_frame, activity)
        bpy.types.Scene.sim_keyframed_activity_reduced[sim_i] = (start_frame, activity_reduced)


def get_nearest_Jacobian(x):
    context = bpy.context

    fp_x = context.scene.fp_states
    fixed_i = context.scene.fixed_i

    x = np.tile(x.reshape(1,-1), (len(fixed_i),1))

    min_dist_i = np.argmin(np.linalg.norm(fp_x - x, axis=1))
    # J = get_Jacobian_at(fp_x[min_dist_i], context.scene.net)

    J = get_Jacobian_at(context.scene.fp_states[33], context.scene.net)

    return J, context.scene.fp_states[33]


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Rotate ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

class OBJECT_OT_rotate(bpy.types.Operator):
    bl_idname = "rnn.rotate"
    bl_label = "Rotate"

    bake: bpy.props.BoolProperty(name='bake', default=False)
    reset: bpy.props.BoolProperty(name='reset', default=False)

    @classmethod
    def poll(cls, context):
        try:
            props = context.scene.rnn_props
            return props.initialised
        except:
            return False
    
    def execute(self, context):
        props = context.scene.rnn_props

        if self.reset:
            
            props.new_dimensions = props.visualised_dimensions

            return {'FINISHED'}
        
        elif len(bpy.data.objects) == 0:

            props.visualised_dimensions = props.new_dimensions

            return {'FINISHED'}

        else:

            # global rotate_process

            # rotate_process = mt.Thread(target=bake_rotation, args=(self.bake,), daemon=True)

            # props.is_rotating = True
            # props.rotate_progress = 0
            # rotate_process.start()

            # bpy.app.timers.register(rotate_timer_func)

            props.visualised_dimensions = props.new_dimensions
            print(props.visualised_dimensions[0], props.visualised_dimensions[1], props.visualised_dimensions[2])


            return {'RUNNING_MODAL'}

rotate_queue = queue.Queue()
rotate_process = None
def rotate_timer_func():
    if not rotate_queue.empty():
        global rotate_process

        props = bpy.context.scene.rnn_props

        for frame, obj, data_path, position, animated in rotate_queue.get():
            if data_path == 'location':
                obj.location = position
            elif data_path == 'co':
                mesh_name, vertex_index = obj
                try:
                    obj = bpy.data.meshes[mesh_name].vertices[vertex_index]
                except:
                    obj = None
                obj.co = position

            if animated:
                obj.keyframe_insert(data_path=data_path, frame=frame)
                props.rotate_progress = ( frame + 1 - props.rotate_start_frame ) / props.rotate_length
            
            else:
                props.rotate_progress = 1
        

        if props.rotate_progress == 1:
            props.is_rotating = False
            props.visualised_dimensions = props.new_dimensions
            rotate_process.join()

            bpy.app.timers.unregister(rotate_timer_func)

    return 0.1

def bake_rotation(animated):
    props = bpy.context.scene.rnn_props

    if animated:
        S = np.linspace(0, 1, props.rotate_length)
        for t in range(props.rotate_length):
            rotate_queue.put(rotate(s=S[t], frame=props.rotate_start_frame + t, animated=animated))
    else:
        rotate_queue.put(rotate(s=1, frame=None, animated=animated))

def rotate(s, frame, animated):
    context = bpy.context

    props = context.scene.rnn_props
    dims = props.new_dimensions
    old_dims = props.visualised_dimensions

    data = []

    for collection in bpy.data.collections:
        if collection.name == 'Simulations':
            for target_i, target in enumerate(context.scene.sim_targets):
                if context.scene.sim_keyframed_activity_reduced[target_i] is None:
                    activity = context.scene.sim_initial_activity[target_i]
                    activity_reduced = get_inter_dims(dims, old_dims, activity, s=s)
                    position = (activity_reduced[0], (activity_reduced[1]), (activity_reduced[2]))
                    data.append((frame, target, 'location', position, animated))
                else:
                    start_frame, activity = context.scene.sim_keyframed_activity[target_i]
                    for t, frame_activity in enumerate(activity):
                        activity_reduced = get_inter_dims(dims, old_dims, frame_activity, s=s)
                        position = (activity_reduced[0], activity_reduced[1], activity_reduced[2])
                        data.append((start_frame+t, target, 'location', position, animated))

        elif collection.name == 'Operative Trajectories':
            for traj_obj in collection.objects:
                i = int(traj_obj.name.split('_')[1])
                activity = context.scene.activity[i]
                for t, vertex in enumerate(traj_obj.data.vertices):
                    activity_reduced = get_inter_dims(dims, old_dims, activity[t], s=s)
                    position = (activity_reduced[0], activity_reduced[1], activity_reduced[2])
                    data.append((frame, (traj_obj.name, t), 'co', position, animated))

        elif 'Fixed Points' in collection.name:
            for fp_obj in collection.objects:
                i = int(fp_obj.name.split('-')[0].split('_')[1])
                activity = context.scene.fp_activity[i]
                activity_reduced = get_inter_dims(dims, old_dims, activity, s=s)
                position = (activity_reduced[0], activity_reduced[1], activity_reduced[2])
                data.append((frame, fp_obj, 'location', position, animated))

        elif collection.name == 'Eigenmodes':
            eig_indices = context.scene.eig_indices
            for subcollection in collection.children_recursive:
                for eig_obj in subcollection.objects:
                    i, e_i = int(eig_obj.name.split('-')[0].split('_')[1]), int(eig_obj.name.split('-')[1].split('_')[1])
                    mask = (eig_indices[:,0]==i) & (eig_indices[:,1]==e_i)
                    activity = context.scene.eig_activity[mask,:]
                    for v_i, vertex in enumerate(eig_obj.data.vertices):
                        activity_reduced = get_inter_dims(dims, old_dims, activity[v_i], s=s)
                        position = (activity_reduced[0], activity_reduced[1], activity_reduced[2])
                        data.append((frame, (eig_obj.name, v_i), 'co', position, animated))

    return data


def get_inter_dims(new_dims, old_dims, activity, s):
    pca = bpy.context.scene.pca

    loadings = pca.components_.transpose(1,0)
    a = get_vis_dims(loadings, dims=old_dims)#pca.components_[old_dims]
    b = get_vis_dims(loadings, dims=new_dims)#pca.components_[new_dims]

    u = a#a + s * (b - a)
    # for dim in range(3):
    #     norm = np.linalg.norm(u[dim])
    #     if norm != 0:
    #         u[dim] *= 1/norm

    try:
        x = u.T @ activity.reshape((-1,1))

        return x.squeeze()
    except:
        return np.array([0,0,0])

def get_vis_dims(activity, dims=None):
    if len(activity.shape) >= 3:
        raise ValueError
    
    vis_activity = np.empty((3,))
    if len(activity.shape) == 2:
        activity = activity.transpose(1,0)
        vis_activity = np.empty((3,activity.shape[1]))

    if dims is None:
        dims = bpy.context.scene.rnn_props.visualised_dimensions

    for i in range(3):
        if dims[i] == 0:
            vis_activity[i] = 0
        elif dims[i] < 0:
            vis_activity[i] = -1 * activity[dims[i]]
        else:
            vis_activity[i] = activity[dims[i]]

    if len(activity.shape) == 2:
        vis_activity = vis_activity.transpose(1,0)

    return vis_activity

##################################################################################################################################
############################################################ PANEL ###############################################################
##################################################################################################################################

class OBJECT_PT_rnn_panel(bpy.types.Panel):
    bl_idname = "OBJECT_PT_analysis_panel"
    bl_label = "Dynamical Analysis"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'RNN'

    @classmethod
    def poll(cls, context):
        try:
            props = context.scene.rnn_props
            return (context.mode == 'OBJECT')
        except:
            return False

    def draw(self, context):
        layout = self.layout
        props = context.scene.rnn_props

        box = layout.box()
        box.label(text="Instance")
        box.row().operator("rnn.open_filebrowser", icon="FILE_FOLDER", text="Select model checkpoint")

        if props.checkpoint_selected:
            box.label(text='Dimension Reduction Settings')
            box.row().prop(props, 'reduction_technique', text="Method")
            if 'pca' in props.reduction_technique:
                box.row().prop(props, 'n_pca_components', text='Components' if 'pca'==props.reduction_technique else 'PCA Components')
            if 'pca' != props.reduction_technique:
                box.row().prop(props, 'n_nl_reduction_samples', text='Samples')
                box.row().prop(props, 'n_nl_components', text=f'Non-Linear Components' if 'pca' in props.reduction_technique else 'Components')
                box.row().prop(props, 'max_nl_iter', text='Max Iterations')
            if 'tsne' == props.reduction_technique:
                box.row().prop(props, 'tsne_perplexity', text='Perplexity')
            if 'mds' == props.reduction_technique:
                box.row().prop(props, 'msd_n_init', text='Initialisations')

        box.row().operator('rnn.initialise', text='Abandon' if props.is_initialising else 'Initialise')

        if props.is_initialising:
            factor = props.init_progress / 3
            text = ['Loading Model', 'Creating Data', 'Reducing Dimensions', 'Finished'][props.init_progress]
            box.row().progress(text=text, factor=factor)

        if props.initialised:
            box = layout.box()
            box.label(text="Visualised Dimensions")
            row = box.row()
            row.prop(props, 'new_dimensions', text='')
            row.operator('rnn.rotate', text='Revert').reset = True
            op = box.row().operator('rnn.rotate')
            op.reset, op.bake = False, False

            if props.technique_used == 'pca':
                box.row().prop(props, 'is_viewing_variance', toggle=True)
                if props.is_viewing_variance:
                    pca = context.scene.pca
                    for i in range(pca.n_components_):
                        factor = pca.explained_variance_ratio_[i]
                        box.row().progress(text=f'{i+1}: {np.round(factor, 3)}', factor=factor)

            # row = box.row()
            # row.prop(props, 'rotate_start_frame')
            # row.prop(props, 'rotate_length')
            # if not props.is_rotating:
            #     op = box.row().operator('rnn.rotate', text='Bake')
            #     op.reset, op.bake = False, True
            # else:
            #     factor = props.rotate_progress
            #     box.row().progress(text='Progress', factor=factor)

            box = layout.box()
            box.label(text="Simulation")
            if context.scene.fp_states is not None:
                box.row().prop(props, 'use_linear_sim')
            if bpy.data.collections.get('Simulations'):
                box.row().prop_search(props, 'sim_target', bpy.data.collections.get('Simulations'), 'objects', text='Target')
            box.row().operator('rnn.get_state', text='New Target at Cursor')
            box.row().operator('rnn.bake_simulation', text='Step').step = True
            row = box.row()
            row.prop(props, 'sim_start_frame')
            row.prop(props, 'sim_length')
            box.row().operator('rnn.bake_simulation', text='Bake').step = False

            box = layout.box()
            box.label(text="Operative Trajectories")
            box.row().prop(props, 'n_operative_trajectories', slider=True)
            box.row().operator("rnn.plot_operative_trajectories")

            if props.technique_used == 'pca':
                box = layout.box()
                box.label(text="Fixed Points")
                box.row().prop(props, 'log10_fixed_threshold', slider=True)
                box.row().operator("rnn.show_fixed_points")
                box.row().prop(props, 'n_search_inits', slider=True)
                box.row().operator("rnn.find_fixed_points", text="Stop" if props.is_searching else "Find")

                if props.is_searching or not search_queue.empty():
                    factor = props.search_progress / props.n_search_inits
                    box.row().progress(text='Search Progress', factor=factor)

                box.row().prop(props, 'cluster_distance', slider=True)
                box.row().operator("rnn.cluster_fixed_points")
                box.row().prop(props, 'tuning_target')
                box.row().operator("rnn.tune_fixed_points")

                box = layout.box()
                box.label(text="Linearisation")
                box.row().prop(props, 'eig_eps', slider=True)

                if not props.is_decomposing:
                    box.row().operator('rnn.decompose_fixed_points')

                else:
                    factor = props.decomp_progress
                    box.row().progress(text='Decomposition Progress', factor=factor)

                





##################################################################################################################################
####################################################### REGISTRATION #############################################################
##################################################################################################################################

operators = [
    OBJECT_OT_rnn_initialise,
    OBJECT_OT_rnn_filebrowser,
    OBJECT_OT_rnn_plot_operative_trajectories,
    OBJECT_OT_show_fixed_points,
    OBJECT_OT_find_fixed_points,
    OBJECT_OT_cluster_fixed_points,
    OBJECT_OT_decompose_fixed_points,
    OBJECT_OT_get_state,
    OBJECT_OT_bake_simulation,
    OBJECT_OT_rotate,
    OBJECT_OT_tune_fixed_points
]

timers = [
    (init_timer_func, init_process, init_queue),
    (search_timer_func, search_process, search_queue),
    (decomp_timer_func, decomp_process, decomp_queue),
    (rotate_timer_func, rotate_process, rotate_queue)
]



def register():

    bpy.utils.register_class(RNN_Properties)
    bpy.types.Scene.rnn_props = bpy.props.PointerProperty(type=RNN_Properties, name='RNN Properties')

    for operator in operators:
        bpy.utils.register_class(operator)

    bpy.utils.register_class(OBJECT_PT_rnn_panel)
    



def unregister():
    bpy.utils.unregister_class(RNN_Properties)

    for operator in operators:
        bpy.utils.unregister_class(operator)

    for timer_func, _, queue in timers:
        # bpy.app.timers.unregister(timer_func)
        while not queue.empty():
            _ = queue.ge()
        

    bpy.utils.unregister_class(OBJECT_PT_rnn_panel)

