import torch
from torch.utils.data import DataLoader
from torch.multiprocessing import Manager

import time
import datetime
import os
import sys

from config import *
from data import *
from net import *
from train import *
from task import *

from hessianfree import HessianFree

import matplotlib
import matplotlib.pyplot as plt

from typing import List, Any, Union

import torch.multiprocessing as mp

import importlib

class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()


############################################################################################################################################
################################################################ BUILD #####################################################################
############################################################################################################################################

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# Function defining full model fit (training, testing, checkpointing)                                                                      #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


'''
build
Creates, trains, and tests model from start to finish
---------------------------------------------------------------------------------------------
Receives
    task :
        Task to which model pertains
    net (optional) :
        If continuing previous build, initialised model from that build
    optimiser (optional) :
        If continuing previous build, initialised optimiser from that build
    train_losses (optional) :
        If continuing previous build, training losses from that build
    test_losses (optional) :
        If continuing previous build, testing losses from that build

Returns
    None

To disk
    config.savedir/
    ----<start time>-<config.name>/
        Build directory
    --------checkpoint-<KEY>:<VALUE>/
            Checkpoint directory, where checkpoint is defined as the model satisfying some checkpoint
            Checkpoints are key:value pairs; can be surpassing some epoch number, converging to some loss, or falling
            below some loss
    ------------net.pt
                PyTorch dictionary at checkpoint
                Contains :
                    net_state_dict : state_dict of model
                    optimiser_state_dict : state_dict of model optimiser
                    train_losses : list of losses during training up to checkpoint
                    test_losses : list of losses during testing up to checkpoint
                    config : dict of build config parameters
    ------------<PLOT>.png
                PNG files of PLOT derived from model with test dataset at checkpoint (see test)

'''

def build(task: Task, net: RNN = None, optimiser: torch.optim.Optimizer = None, train_losses: List[float] = None, test_losses: List[float] = None):
    config = task.config
    config.seed()
    torch.manual_seed(config.build_seed)
    np.random.seed(config.build_seed)
    if torch.config.precise:
        torch.set_default_dtype(torch.float64)
        if net is not None:
            net.double()

    # Define name for the build and save directory
    config_name = config.get_name()
    build_dir = f'{config.savedir}/{config.time}-{config_name}'
    if not os.path.isdir(build_dir):
        os.makedirs(build_dir)
        outfile = open(f'{build_dir}/build.out', 'w')
    else:
        outfile = open(f'{build_dir}/build.out', 'a')
    sys.stdout = Tee(sys.stdout, outfile)

    # Define time at which to terminate training as max_hours from start timne
    killtime = time.time() + (config.max_hours * 60 * 60)

    print(f'{task.config.device} build: {build_dir}')

    print(config)

    # Initialise model and optimiser according to build configuration (if not already supplied)
    if net is None:
        if task.config.rank is not None:
            print(f'Using low-rank network of rank {task.config.rank}')
            net = LowRankRNN(task, rank=int(task.config.rank))
        else:
            net = RNN(task)

    if optimiser is None:
        if config.optimiser_name == 'Adam':
            optimiser = torch.optim.Adam(net.parameters(), lr=config.max_lr)
        elif config.optimiser_name == 'AdamW':
            optimiser = torch.optim.AdamW(net.parameters(), lr=config.max_lr)
        elif config.optimiser_name == 'HF' or config.optimiser_name == 'HessianFree':
            optimiser = HessianFree(net.parameters(), lr=config.max_lr,
                                    damping=config.HF_damping,delta_decay=config.HF_delta_decay,cg_max_iter=config.HF_cg_max_iter,verbose=False)
        else:
            raise Exception('Invalid optimiser_name: use Adam, AdamW, HF/HessianFree')

    # Intialise loader for training data, and one fixed batch for testing data
    train_dataset = TaskDataset(task, for_training=True)
    train_batch_loader = DataLoader(train_dataset, batch_size=config.batch_size, num_workers=config.num_loader_workers, collate_fn=train_dataset.collate_fn)

    test_dataset = TaskDataset(task, for_training=False)
    test_batch = test_dataset.get_batch()

    # Private function to compute loss on test set
    def _get_test_loss() -> float:
        with torch.no_grad():
            loss,_ = task.get_loss(net=net, batch=test_batch)
            return loss.item()
        
    # Initialise lists for tracking losses (and get an initial loss)
    if train_losses is None:
        train_losses = []
    if test_losses is None:
        test_losses = [_get_test_loss()]

    print(f'\n({task.config.device}) Test dataset loss without training: {test_losses[-1]:.4f}\n')

    # Private function for saving a checkpoint for some key-value
    def _save_checkpoint(key: str, value: Union[int, float], test: bool = False):
        # Define file structure for checkpoint (create directories if this is first checkpoint)
        checkpoint_dir = f'{build_dir}/checkpoint-{key}:{value}'
        if not os.path.isdir(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        checkpoint_filepath = f'{checkpoint_dir}/net.pt'

        # Create and save checkpoint dictionary
        checkpoint = {
            'net_state_dict': net.state_dict(),
            'optimiser_state_dict': optimiser.state_dict(),
            'test_losses': test_losses,
            'train_losses': train_losses,
            'config': dict(config.dict),
        }
        
        torch.save(checkpoint, checkpoint_filepath)
        print(f'- ({task.config.device}) Saved model for checkpoint {key}={value}')

        # If testing, use task-specific testing function to generate plots and save them to checkpoint
        if test:
            with torch.no_grad():
                test_result = task.test_func(task=task, net=net, batch=test_batch, checkpoint_path=checkpoint_filepath, **task.test_func_args)

                if test_result is not None:
                    for val in test_result.values():
                        if type(val) == matplotlib.figure.Figure:
                            plt.close(val)
    
    # If so desired (not by default), save initial net
    if config.save_initial_net and len(train_losses)==0:
        _save_checkpoint('epochs', 0)

    
    # Private function for updating learning rate according to schedule
    def _update_learning_rate(epoch=0):
        # If no fixed-duration of training is set, learning rate is scheduled according to loss
        # It will decrease according to config.lr_anneal_schedule as the loss approaches some minimum (usually 0)
        if config.n_epochs <= 0:
            if len(test_losses) >= 2:
                p = config.lr_anneal_schedule
                l_0, l_T, l = test_losses[0], config.training_threshold, test_losses[-1]
                r_max, r_min = config.max_lr, config.min_lr
                lr = ((r_max - r_min) * (l - l_T)**p / (l_0 - l_T)**p) + r_min
            else:
                lr = config.max_lr
            
        # If there is a fixed duration for training, then learning rate is scheduled according to epoch
        # Initially it will increase from a minimum to a maximum, and then it will anneal back to the minimum
        else:
            n = config.lr_initial_schedule
            lam = config.lr_anneal_schedule
            M = config.max_lr
            m = config.min_lr
            t = epoch
            lr = M * np.exp(n) * (lam / n)**n * t**n * np.exp(-lam * t) + m
        
        # How learning rate is updated depends on which optimiser is being used
        if 'lr' in optimiser.param_groups[0]:
            optimiser.param_groups[0]['lr'] = lr
        if 'alpha' in optimiser.param_groups[0]:
            optimiser.param_groups[0]['alpha'] = lr

    _update_learning_rate()

    # For convenience, convert loss thresholds at which to save into a numpy array
    save_at_losses = np.array(config.save_losses)
    # If build is continuing from a previous build, use the directory to see which loss thresholds already have checkpoints
    if os.path.isdir(build_dir):
        saved_losses = [1e9]
        for dirname in os.listdir(build_dir):
            if 'checkpoint-loss' in dirname:
                loss = float(dirname.split(':')[-1])
                saved_losses.append(loss)
        saved_losses.sort(reverse=True)
        saved_losses = np.array(saved_losses)
    else:
        saved_losses = np.array([])

    # And similarly for which epochs already have checkpoints
    save_at_epochs = config.save_epochs
    if os.path.isdir(build_dir):
        saved_epochs = [0]
        for dirname in os.listdir(build_dir):
            if 'checkpoint-epochs' in dirname:
                epoch = int(dirname.split(':')[-1])
                saved_epochs.append(epoch)
        saved_epochs.sort()
        saved_epochs = np.array(saved_epochs)
    else:
        saved_epochs = np.array([0])

    start_time = time.time()
    # Offset for epoch number of build is continuing from a previous build
    epoch_offset = len(train_losses) // ((config.batch_size // config.minibatch_size) * config.num_batch_repeats)
    for epoch, epoch_losses in train(task, net, train_batch_loader, optimiser, n_epochs=config.n_epochs, killtime=killtime):
        epoch += epoch_offset
        train_losses += epoch_losses

        # Exit conditions
        if config.n_epochs <= 0:
            # Loss goal exceeded
            if test_losses[-1] <= config.training_threshold:
                print(f'({task.config.device}) build completed')
                break
            # Loss converges
            if len(train_losses) >= config.training_convergence_std_threshold_window:
                training_window_std = np.std(train_losses[-config.training_convergence_std_threshold_window:])
                print(f'-({task.config.device}) Training window std. dev.: {training_window_std:.4E}')

                if training_window_std < config.training_convergence_std_threshold:
                    _save_checkpoint(key='loss', value=np.round(test_losses[-1], 2), test=True)
                    print(f'({task.config.device}) build converged')
                    break

        # If on testing epoch, test model and report
        if epoch % config.test_epochs == 0:
            test_losses.append(_get_test_loss())
            print('- ({}) test dataset loss after {} epochs ({} updates): {:.4f}'.format(
                task.config.device, epoch, len(train_losses), test_losses[-1], 4))
            print('- ({}) Total time elapsed: {}'.format(task.config.device, datetime.timedelta(seconds=time.time() - start_time)))


        # Save conditions
        thresholds_met = save_at_losses[test_losses[-1] <= save_at_losses]
        threshold = None if len(thresholds_met)==0 else np.min(thresholds_met)

        # If a new loss checkpoint has been surpassed, save
        if threshold is not None and threshold not in saved_losses:
            saved_losses = np.append(saved_losses, threshold)
            _save_checkpoint(key='loss', value=threshold, test=False)

        # If on a save epoch, save
        if epoch - saved_epochs[-1] >= save_at_epochs:
            saved_epochs = np.append(saved_epochs, epoch - (epoch%save_at_epochs))
            _save_checkpoint(key='epochs', value=saved_epochs[-1], test=False)


        _update_learning_rate(epoch=epoch+epoch_offset)

        sys.stdout.flush()

    # Free memory
    del net
    del optimiser
    del train_dataset
    del train_batch_loader
    del test_dataset
    del test_batch
    del train_losses
    torch.cuda.empty_cache()
    outfile.close()



'''
build_from_command_line
Parses command line args into build parameters before building
---------------------------------------------------------------------------------------------

Assumes command line input of the format:

python3 -m build <TASK_NAME> -<KEY> <VALUE> -<KEY> <VALUE>
where TASK_NAME is a task name defined in one of the task files in Tasks
      KEY and VALUE are optional flag pairs and represent config settings (e.g. -n_neurons 100)
-N can be included after TASK_NAME or a VALUE to indicate that the following settings should not be included in the
    build name
-c <FILEPATH> can be included after TASK_NAME to indicate a build should continue from the checkpoint found at 
    FILEPATH (must end in .pt) (cannot be used with -N)
-P before a KEY indicates that the following parameter can take on multiple values, and these should be trained on all available GPUs in parallel
    e.g. -P -n_neurons 10 100 1000 would train three models on three GPUs, with 10, 100, and 1000 neurons respectively
    if only one value is supplied, then multiple copies of the same model will be trained in paralle, equal to the number of GPUs available

'''

def build_from_command_line():
    args = sys.argv
    assert len(args) >= 2

    task_name = args[1]

    importlib.import_module(f'Tasks.{task_name.replace("-", "_")}')

    # Extract task object from first argument
    task = None
    try:
        task = Task.named(task_name)
    except:
        raise ValueError(f'{task_name} is not a defined task name.')
    
    config_manager = Manager()
    task.config.manage(config_manager)


    net, optimiser, train_losses, test_losses = None, None, None, None

    if len(args) > 1:
        params = {}
        name_param_keys = set()
        include = True
        parallel_param_settings = {}

        for arg_i in range(1, len(args)):

            # -N flag indicates parameters should stop being included in build name
            if args[arg_i] == '-N':
                assert '-c' not in args

                include = False
                continue

            # If contunuing a previous build:
            elif args[arg_i] == '-c':
                assert '-P' not in args
                assert arg_i==2

                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

                # Load checkpoint from specified path
                checkpoint = None
                try:
                    checkpoint = torch.load(args[arg_i+1], map_location=device)
                except Exception as e:
                    print(f'Checkpoint load failed: {e}')

                # Update device being trained on
                checkpoint['config']['device'] = device
                if checkpoint['config']['precise']:
                    torch.set_default_dtype(torch.float64)
                
                # Get task from checkpoint
                task = Task.from_checkpoint(checkpoint)

                # Get model from checkpoint
                if task.config.rank is not None:
                    net = LowRankRNN(task, rank=int(task.config.rank))
                else:
                    net = RNN(task)
                net.load_state_dict(checkpoint['net_state_dict'])

                # Get optimiser from checkpoint
                if '-optimiser_name' in args:
                    task.config.update(optimiser_name = args[args.index('-optimiser_name')+1])
                if task.config.optimiser_name == 'Adam':
                    optimiser = torch.optim.Adam(net.parameters(), lr=task.config.max_lr)
                elif task.config.optimiser_name == 'AdamW':
                    optimiser = torch.optim.AdamW(net.parameters(), lr=task.config.max_lr)
                elif task.config.optimiser_name == 'HF' or task.config.optimiser_name == 'HessianFree':
                    optimiser = HessianFree(net.parameters(), lr=task.config.max_lr,
                                            damping=task.config.HF_damping,delta_decay=task.config.HF_delta_decay,cg_max_iter=task.config.HF_cg_max_iter,verbose=False)
                if task.config.optimiser_name == checkpoint['config']['optimiser_name']:
                    optimiser.load_state_dict(checkpoint['optimiser_state_dict'])

                # Get losses from checkpoint
                train_losses = checkpoint['train_losses']
                test_losses = checkpoint['test_losses']

            # Otherwise, an argument prefixed with '-' indicates a config parameter
            # Attempt to save this parameter to task config (type must match default value)
            elif args[arg_i][0] == '-':
                if args[arg_i] == '-P':
                    key, value = args[arg_i+1][1:], None
                    parallel_param_settings[key] = []
                else:
                    key, value = args[arg_i][1:], args[arg_i+1]
                
                if key in task.config.dict:
                    try:
                        type_cast = type(task.config.dict[key])

                        if args[arg_i] == '-P':
                            for value in args[arg_i+2:]:
                                if value[0] == '-':
                                    break
                                else:
                                    parallel_param_settings[key].append(type_cast(value))
                        else:
                            if type_cast == type(True):
                                value = False if 'f' in value.lower() else True
                            elif task.config.dict[key] is None:
                                params[key] = value
                                continue
                            params[key] = type_cast(value)

                            if include:
                                name_param_keys.add(key)
                    except Exception as e:
                        print(f'Setting parameter {key} failed: {e}')
                        continue
                else:
                    print(f'No parameter named {key}')
                    continue

        task.config.update(**params)
        if '-c' not in args:
            task.config.make_name(include=name_param_keys)

        if '-P' in args:
            mp.set_start_method('spawn')

            is_equal, num_models = True, 0
            for key in parallel_param_settings.keys():
                num_models = len(parallel_param_settings[key])
                if key not in name_param_keys:
                    name_param_keys.add(key)
                for other_key in parallel_param_settings.keys():
                    if key != other_key:
                        is_equal = is_equal and len(parallel_param_settings[key]) == len(parallel_param_settings[other_key])
            assert is_equal
            # assert num_models == torch.cuda.device_count()

            task.config.seed()

            processes = []
            for i in range(num_models):
                parallel_task = task.copy()
                for key, vals in parallel_param_settings.items():
                    parallel_task.config.dict[key] = vals[i]
                    if torch.cuda.is_available():
                        parallel_task.config.dict['device'] = f'cuda:{i % torch.cuda.device_count()}'
                    else:
                        parallel_task.config.dict['device'] = f'cpu'
                    parallel_task.config.dict['time'] = task.config.time
                parallel_task.config.make_name(include=name_param_keys)

                p = mp.Process(target=build, args=(parallel_task,))
                p.start()
                processes.append(p)

            for p in processes:
                p.join()

        else:

            build(task, net=net, optimiser=optimiser, train_losses=train_losses, test_losses=test_losses)


    


if __name__ == '__main__':
    
    build_from_command_line()
