# Orientation To Shelter

## Overview

The best way to describe this project is to step backward from how it is usually used. When used to initialise, train, and test a model end-to-end, navigate into the project directory and run:

```
python3 -m build <TASK> -<PARAM_NAME> <PARAM_VALUE> ...
```

Where TASK is the name of a defined task, and PARAM_NAME and PARAM_VALUE are key-value pairs which defined hyperparametrisations of the task. For example: `python3 -m build HD-0D -n_neurons 1000` will train a model to perform the head-direction integration task (see `Tasks/HD_0D.py`) with an RNN of 1000 hidden units. Any parameters not specifed at the command line will take on their default values (see `default_params` in `config.py` for general parameters, or the `task_specific_params` associated with any defined task). These commands will run the `build` module, which handles this end-to-end training (see `build.py`).

`build` will extract TASK and use it to extract the associated task to train on. Tasks are defined by `Task` objects, which gather in one place the methodologies and hyperparametrisations of training RNNs to learn some task. To give an idea of what such an object looks like, the below example is given, which defines a task where an RNN will be asked to extract the mean from a noisy input stream, and output a sine wave with that frequency:
```
params = dict(min_freq=0, max_freq=10, input_std=0.1, grace_period=10)

def create_data(config, inputs, targets, mask):
    batch_size, n_timesteps = config.batch_size, config.n_timesteps
    
    frequencies = (config.max_freq - config.min_freq) * torch.rand((batch_size,)) + config.min_freq

    normal_dist = torch.distributions.normal.Normal(loc=frequencies, scale=torch.ones((batch_size,)) * config.input_std)
    inputs = normal_dist.sample(sample_shape=(batch_size, n_timesteps, 1))

    time = torch.linspace(0, 1, n_timesteps)
    targets = torch.sin(2 * torch.pi * frequencies.unsqueeze(1) * time).reshape((batch_size, n_timesteps, 1))

    mask[:, :grace_period] = False

    vars = {'f': frequencies}

    return inputs, targets, vars, mask

input_map = {0: 'f'}
target_map = {0: 'sin_f'}

task = Task(name='oscillator', 
            n_inputs=1, n_outputs=1, 
            task_specific_params=params, create_data_func=create_data, 
            input_map=input_map, target_map=target_map)
```

'oscillator' is now the name of this task, and is stored in a global register (`task_register`). `build` uses TASK as the key for this register to extract the task to be trained on. It then uses the hyperparameters given to update the task object accordingly (e.g. specifiying 1000 neurons).

At this point, `build` initialises the objects used in training: the network (see `net.py`), the optimiser (can be Adam or Hessian-Free), and the datasets (see `data.py`). It will then begin the training loop (see the `train` generator in `train.py`). At each iteration of the loop, the network is presented with input data and predicts some output, which is compared to a target output to define a loss (inputs and targets (and a mask on inputs for loss calculation) are defined in a task's `create_data_func`, and its loss calculation in its `loss_func`). The optimiser updates the network params, and a training step is complete.

After each step, `build` performs several checks. These include scheduling the learning rate, and checking to see if any exit conditions (e.g. trianing has achieved a certain performance threshold or a maximum time has been reached) or save conditions (e.g. completing a certain number of updates) have been met. In the latter case, a 'checkpoint' will be saved to a local directory (`trained-models` by defualt). In the former case, if exiting because a performance threshold has been met or training has converged, testing will also be performed. Which tests are run is determined by a task's `test_func`, but in general, this will involve generating several plots based on a model's performance at that stage, and saving them to that checkpoint directory (see `test_funcs.py` for plotting functions).

The result of `build` should be a subdirectory in the project directory which contains the trained model, and a number of plots attesting to its performance, tuning, etc., at various stages in the course of its training

Each of these components (classes `Task`, `RNN`, `Config`, `TaskDataset`) can of course be combined in any number of ways outside of `build` to run some experiment or test some model. But understanding `build` is a good way to understand the organisation of the code (and the exact methodology used for training RNNs).



## Contents

### build.py

Contains the function `build`, which creates, trains, and tests an RNN model on a given task, end-to-end. It also contains the logic for doing so from the command line.

### config.py

Defines the class `Config`, which is essentially a convenience wrapper for a dictionary containing hyperpameter names as keys, and their values. The default key-value pairs are also found here, in the global `default_params`.

### data.py

Defines the class `TaskDataset`, which is the middle-man between `Task` objects -- which define data creation routines pertaining to a task -- and the PyTorch backend, which expects large datasets defined *a priori* loaded batch-by-batch. In practice, this means `TaskDataset` just offers generated data batch-by-batch; importantly, this also provides compatability with PyTorch's `DataLoader` so that intensive data creation routines (e.g. when using RatInABox) can be run on multiple cores at once.

### hessianfree.py

The Hessian-Free optimiser (adapted from https://github.com/fmeirinhos/pytorch-hessianfree.)

### net.py

Defines the `RNN` class, which implements the continuous-time RNN as a PyTorch module. It also defines `LowRankRNN`, which constrains the connectivity of an RNN to have a specific rank.

### task.py

Defines the `Task` class, which brings together the data-creation routines, task-specific hyperparameters, testing routines, and loss functions which define a given 'task' for an RNN to learn.

### Tasks/

Contains several python files which define the tasks pertaining to this project (HD-0D is Cueva's head-direction task, PI-2D is his path integration task; learning PI_HD_SD-2D is our end goal). Also in here are `vars_<0|1|2>D.py` files, which contain utility functions for creating head-direction, path-integration, and shelter-orienting variables in 0, 1, and 2 dimensions. Most tasks involve piecing together different combinations of these for targets.

### test_funcs.py

Functions for calculating tuning profiles of networks (`get_tuning_generalised`) and for plotting the results.

### train.py

Defines the `train` generator, which implements the training loop for an RNN.