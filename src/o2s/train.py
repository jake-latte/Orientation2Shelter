import torch
import numpy as np
import time
from PIL import Image

import o2s






############################################################################################################################################
################################################################ TRAIN #####################################################################
############################################################################################################################################

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# Function defining training loop for RNN                                                                                                  #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


'''
train (generator)
Executes the training loop for given task/RNN, one epoch at a time
---------------------------------------------------------------------------------------------
Receives
    task :
        Task to train on
    net :
        RNN to train
    batch_loader :
        DataLoader object to load task dataset batches
    optimiser :
        torch.optim.Optimiser object for gradient descent
    killtime :
        time at which to terminate training loop (unix time)

Yields
    int :
        Number of epoch just completed
    list :
        Loss at each update of epoch just completed

Terminates
    - when killtime is exceeded
    - when specified number of epochs is completed

'''
def train(task: o2s.task.Task, net: o2s.net.RNN, batch_loader: torch.utils.data.DataLoader, optimiser: torch.optim.Optimizer, killtime: float = None, **kwargs):
    config = task.config

    epoch = 0
    epoch_losses = []

    # Function for checking end conditions:
    # Either killtime is exceeded or specified number of epochs have been completed
    def _check_end_conditions():
        if killtime is not None and time.time() > killtime:
            print('Training killed')
            return epoch, epoch_losses
        
        if config.n_epochs > 0 and epoch > config.n_epochs:
            return epoch, epoch_losses
    
    # Each batch is repeated config.num_batch_repeats times, and split into minibatches of size config.minibatch_size
    for batch in batch_loader:
        start_time = time.time()
        epoch += 1
        epoch_losses = []

        _check_end_conditions()
        
        net.train(True)

        for repeat in range(config.num_batch_repeats):

            _check_end_conditions()
            
            # Shuffle batch with each repeat
            shuffle_i = torch.randperm(config.batch_size)
            for key, value in batch.items():
                if key=='noise':
                    state_noise = None if value[0] is None else value[0][shuffle_i]
                    rate_noise = None if value[1] is None else value[1][shuffle_i]
                    output_noise = None if value[2] is None else value[2][shuffle_i]
                    batch[key] = state_noise, rate_noise, output_noise 
                elif key=='vars':
                    batch[key] = {k: v[shuffle_i] for k, v in batch[key].items()}
                else:
                    batch[key] = value[shuffle_i]

            for minibatch_i in range(0, config.batch_size, config.minibatch_size):

                _check_end_conditions()

                # Split batch into minibatches
                minibatch = {}
                device = 'cpu' if config.conserve_vram else config.device
                for key, value in batch.items():
                    if key=='noise':
                        state_noise = None if value[0] is None else value[0][minibatch_i:minibatch_i+config.minibatch_size].to(device)
                        rate_noise = None if value[1] is None else value[1][minibatch_i:minibatch_i+config.minibatch_size].to(device)
                        output_noise = None if value[2] is None else value[2][minibatch_i:minibatch_i+config.minibatch_size].to(device)
                        minibatch[key] = state_noise, rate_noise, output_noise
                    elif key=='vars':
                        minibatch[key] = {k: v[minibatch_i:minibatch_i+config.minibatch_size].to(device) for k,v in batch[key].items()}
                    else:
                        minibatch[key] = value[minibatch_i:minibatch_i+config.minibatch_size].to(device)

                def _backward_from_loss():
                    loss, outputs = task.get_loss(net, minibatch)

                    loss.backward(retain_graph=config.optimiser_name=='HF')

                    return loss, outputs

                # Gradient descent step
                optimiser.zero_grad()
                loss,_ = optimiser.step(_backward_from_loss)

                # Save losses
                epoch_losses.append(loss.item())

        net.train(False)
        
        # Extract current learning rate from optimiser depending on type
        lr = None
        if 'lr' in optimiser.param_groups[0]:
            lr = optimiser.param_groups[0]['lr']
        elif 'alpha' in optimiser.param_groups[0]:
            lr = optimiser.param_groups[0]['alpha']

        # Message console
        print('({}) Epoch: {}; Updates: {} (lr={:.4E}); Average Loss: {:0.4f}; Time Elapsed: {:0.1f}s'.format(
            config.device, epoch, len(epoch_losses), lr,  np.mean(epoch_losses).item(), time.time() - start_time))
        
        # Yield epoch results (generally to build function) for inter-epoch processing
        yield epoch, epoch_losses