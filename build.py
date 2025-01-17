import torch
from torch.utils.data import DataLoader

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

import Tasks


############################################################################################################################################
################################################################ TRAIN #####################################################################
############################################################################################################################################

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# Function defining full model fit (training, testing, checkpointing)                                                                      #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


'''
build
Creates, trains, and tests model from start to finish
---------------------------------------------------------------------------------------------
Receives
    config :
        Configuration object for the build
    max_hours :
        Number of hours at which to terminate process
    kwargs :
        other kwargs which can override config parameters

Returns
    None

To disk
    config.savedir/
    ----<start time>-<config.name>/
        Build directory
    --------threshold:<T>/
            Checkpoint directory, where checkpoint is defined as the model exceeding some loss 
            threshold T after E weight epochs, with loss L at last update
    ------------epoch:<E>-loss:<L>.pt
                PyTorch-saved dictionary at threshold checkpoint
                Contains :
                    net_state_dict : state_dict of model
                    optimiser_state_dict : state_dict of model optimiser
                    loss : list of losses during training up to checkpoint
                    config : dict of build config parameters
    ------------<PLOT>.png
                PNG files of PLOT derived from model with test dataset at checkpoint (see test)

'''

def build(task, net=None, optimiser=None, train_losses=None, test_losses=None):
    config = task.config
    config.seed()
    torch.manual_seed(config.build_seed)
    np.random.seed(config.build_seed)

    # Define name for the build and save directory
    config_name = config.get_name()
    build_dir = f'{config.savedir}/{config.time}-{config_name}'

    # Define time at which to terminate training as max_hours from start timne
    killtime = time.time() + (config.max_hours * 60 * 60)

    print(f'Build: {build_dir}')

    print(f'\n{"-"*25} CONFIG {"-"*25}\n')
    for key, val in config.__dict__.items():
        flag = '\t' if val == task_register[config.task].config[key] else '  ****  '
        print(f'{flag}{key}: {val}')
    print('-'*58)
    
    

    # Initialise model according to build configuration
    if net is None:
        if task.config.rank is not None:
            print(f'Using low-rank network of rank {task.config.rank}')
            net = LowRankRNN(config, rank=int(task.config.rank))
        else:
            net = RNN(config)

    # Initialise optimiser according to build training algorithm
    if optimiser is None:
        if config.optimiser_name == 'Adam' or config.optimiser_name == 'AdamHF':
            optimiser = torch.optim.Adam(net.parameters(), lr=config.max_lr)
        elif config.optimiser_name == 'AdamW':
            optimiser = torch.optim.AdamW(net.parameters(), lr=config.max_lr)
        elif config.optimiser_name == 'HF' or config.optimiser_name == 'HessianFree':
            optimiser = HessianFree(net.parameters(), lr=config.max_lr,
                                    damping=config.HF_damping,delta_decay=config.HF_delta_decay,cg_max_iter=config.HF_cg_max_iter,verbose=False)
        else:
            raise Exception('Invalid optimiser_name: use Adam, AdamW, HF/HessianFree, or AdamHF')

    train_dataset = TaskDataset(task, for_training=True)
    train_batch_loader = DataLoader(train_dataset, batch_size=config.batch_size, num_workers=config.num_loader_workers, collate_fn=train_dataset.collate_fn)

    test_dataset = TaskDataset(task, for_training=False)
    test_batch = test_dataset.get_batch()

    # Private func to compute loss on test set
    def _get_test_loss():
        with torch.no_grad():
            loss,_ = task.get_loss(net=net, batch=test_batch)
            return loss.item()
        
    # Initialise variable to keep track of performance on test dataset
    if train_losses is None:
        train_losses = []
    if test_losses is None:
        test_losses = [_get_test_loss()]

    print('\nTest dataset loss without training: {}\n'.format(round(test_losses[-1], 4)))

    def _save_checkpoint(key, value, test=False):
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
            'config': config.__dict__,
        }
        
        torch.save(checkpoint, checkpoint_filepath)
        print(f'- Saved model for checkpoint {key}={value}')

        if test:
            with torch.no_grad():
                test_result = task.test_func(task=task, net=net, batch=test_batch, checkpoint_path=checkpoint_filepath, **task.test_func_args)

                if test_result is not None:
                    for val in test_result.values():
                        if type(val) == matplotlib.figure.Figure:
                            plt.close(val)
    
    if config.save_initial_net:
        _save_checkpoint('epochs', 0)

    
    def _update_learning_rate(epoch=0):
        if config.n_epochs <= 0:
            if len(test_losses) >= 2:
                p = config.lr_anneal_schedule
                l_0, l_T, l = test_losses[0], config.training_threshold, test_losses[-1]
                r_max, r_min = config.max_lr, config.min_lr
                lr = ((r_max - r_min) * (l - l_T)**p / (l_0 - l_T)**p) + r_min
            else:
                lr = config.max_lr
            
        else:
            n = config.lr_initial_schedule
            lam = config.lr_anneal_schedule
            M = config.max_lr
            m = config.min_lr
            t = epoch
            lr = M * np.exp(n) * (lam / n)**n * t**n * np.exp(-lam * t) + m

        if 'lr' in optimiser.param_groups[0]:
            optimiser.param_groups[0]['lr'] = lr
        if 'alpha' in optimiser.param_groups[0]:
            optimiser.param_groups[0]['alpha'] = lr

    _update_learning_rate()

    save_at_losses = np.array(config.save_losses)
    if os.path.isdir(build_dir):
        saved_losses = []
        for dirname in os.listdir(build_dir):
            if 'checkpoint-loss' in dirname:
                loss = float(dirname.split(':')[-1])
                saved_losses.append(loss)
        saved_losses.sort(reverse=True)
        saved_losses = np.array(saved_losses)
    else:
        saved_losses = np.array([])

    save_at_epochs = config.save_epochs
    if os.path.isdir(build_dir):
        saved_epochs = []
        for dirname in os.listdir(build_dir):
            if 'checkpoint-epochs' in dirname:
                epoch = int(dirname.split(':')[-1])
                saved_epochs.append(epoch)
        saved_epochs.sort()
        saved_epochs = np.array(saved_epochs)
    else:
        saved_epochs = np.array([0])

    start_time = time.time()
    optimiser_swapped = False
    epoch_offset = len(train_losses) // ((config.batch_size // config.minibatch_size) * config.num_batch_repeats)
    for epoch, epoch_losses in train(task, net, train_batch_loader, optimiser, n_epochs=config.n_epochs, killtime=killtime):
        train_losses += epoch_losses

        # Exit conditions for threshold-based
        if config.n_epochs <= 0:
            if test_losses[-1] <= config.training_threshold:
                print('Build completed')
                break

            if len(train_losses) >= config.training_convergence_std_threshold_window:
                training_window_std = np.std(train_losses[-config.training_convergence_std_threshold_window:])
                print(f'-Training window std. dev.: {training_window_std}')

                if training_window_std < config.training_convergence_std_threshold:
                    _save_checkpoint(key='loss', value=np.round(test_losses[-1], 2), test=True)
                    print('Build converged')
                    break

        if epoch % config.test_epochs == 0:
            test_losses.append(_get_test_loss())
            print('- Test dataset loss after {} epochs ({} updates): {}'.format(
                epoch, len(train_losses), round(test_losses[-1], 4)))
            print('- Total time elapsed: {}'.format(datetime.timedelta(seconds=time.time() - start_time)))


        # Save conditions
        thresholds_met = save_at_losses[test_losses[-1] <= save_at_losses]
        threshold = None if len(thresholds_met)==0 else np.min(thresholds_met)

        if threshold is not None and threshold not in saved_losses:
            saved_losses = np.append(saved_losses, threshold)
            _save_checkpoint(key='loss', value=threshold, test=False)

        if epoch - saved_epochs[-1] >= save_at_epochs:
            saved_epochs = np.append(saved_epochs, epoch - (epoch%save_at_epochs))
            _save_checkpoint(key='epochs', value=saved_epochs[-1], test=False)
            
        if not optimiser_swapped and config.optimiser_name == 'AdamHF':
            if test_losses[-1] < 0.1:
                optimiser_swapped = True
                optimiser = HessianFree(net.parameters(), lr=optimiser.lr,
                                        damping=config.HF_damping,delta_decay=config.HF_delta_decay,cg_max_iter=config.HF_cg_max_iter,verbose=False)


        _update_learning_rate(epoch=epoch+epoch_offset)

    # Free memory
    del net
    del optimiser
    del train_dataset
    del train_batch_loader
    del test_dataset
    del test_batch
    del train_losses
    torch.cuda.empty_cache()





def build_from_command_line(task, args):
    if len(args) > 1:
        params = {}
        name_param_keys = []
        include = True
        for arg_i in range(1, len(args)):
            if args[arg_i] == '-N':
                include = False
                continue

            if args[arg_i][0] == '-':
                key, value = args[arg_i][1:], args[arg_i+1]
                
                if key in task.config.__dict__:
                    try:
                        type_cast = type(task.config.__dict__[key])
                        if type_cast == type(True):
                            value = False if 'f' in value.lower() else True
                        elif task.config.__dict__[key] is None:
                            params[key] = value
                            continue
                        params[key] = type_cast(value)

                        if include:
                            name_param_keys.append(key)
                    except Exception as e:
                        print(f'Setting parameter {key} failed: {e}')
                        continue
                else:
                    print(f'No parameter named {key}')
                    continue

        task.config.update(**params)
        task.config.make_name(include=name_param_keys)

    build(task)
    

def build_from_checkpoint(checkpoint_filepath):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(checkpoint_filepath, map_location=device)
    checkpoint['config']['device'] = device
    
    task = Task.from_checkpoint(checkpoint)

    if task.config.rank is not None:
        net = LowRankRNN(task.config, rank=int(task.config.rank))
    else:
        net = RNN(task.config)
    net.load_state_dict(checkpoint['net_state_dict'])

    # Initialise optimiser according to build training algorithm
    if task.config.optimiser_name == 'Adam':
        optimiser = torch.optim.Adam(net.parameters(), lr=task.config.max_lr)
    elif task.config.optimiser_name == 'AdamW':
        optimiser = torch.optim.AdamW(net.parameters(), lr=task.config.max_lr)
    elif task.config.optimiser_name == 'HF' or task.config.optimiser_name == 'HessianFree':
        optimiser = HessianFree(net.parameters(), lr=task.config.max_lr,
                                damping=task.config.HF_damping,delta_decay=task.config.HF_delta_decay,cg_max_iter=task.config.HF_cg_max_iter,verbose=False)
    optimiser.load_state_dict(checkpoint['optimiser_state_dict'])

    train_losses = checkpoint['train_losses']
    test_losses = checkpoint['test_losses']

    build(task, net=net, optimiser=optimiser, train_losses=train_losses, test_losses=test_losses)


    


if __name__ == '__main__': 
    import Tasks

    assert len(sys.argv) >= 2

    if '-c' in sys.argv:
        build_from_checkpoint(sys.argv[-1])
    else:

        task_name = sys.argv[1]

        task = None
        try:
            task = task_register[task_name].copy()

        except:
            raise ValueError(f'{task_name} is not a defined task name.')
        
        if task is not None:
            build_from_command_line(task, sys.argv)