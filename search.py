import os
import random
import itertools
import json
import subprocess
import time
import sys
import torch
import torch.multiprocessing as mp

from task import Task
from Tasks.util import register_all_tasks
register_all_tasks()

from net import RNN
from build import build


def run_build(task, device):
    if device.startswith('cuda'):
        torch.cuda.set_device(int(device.split(':')[-1]))
    build(task)

if __name__ == '__main__':
    # Define the subset of hyperparameters and their possible values
    search_hyperparameters = {
        'n_neurons': [25, 100, 225, 400],
        'activation_func_name': ['ReTanh', 'Tanh', 'ReLU'],
        'learn_W_in': [True, False],
        'learn_W_out': [True, False],
        'weight_lambda': [0.0, 0.01, 0.1, 0.5],
        'rate_lambda': [0.0, 0.01, 0.1, 0.5],
        'weight_loss_type': [1, 2],
        'rate_loss_type': [1, 2],
    }

    task_name = sys.argv[1]

    if '-d' in sys.argv:
        arg_i = sys.argv.index('-d')
        results_folder = sys.argv[arg_i+1]
    else:
        results_folder = f'{time.time()}-search-{task_name}'
    os.makedirs(results_folder, exist_ok=True)

    if '-n' in sys.argv:
        arg_i = sys.argv.index('-n')
        n = int(sys.argv[arg_i+1])
    else:
        n = 10


    # Generate all possible combinations of the hyperparameters
    all_combinations = list(itertools.product(*search_hyperparameters.values()))

    # Pick a random subset of the combinations
    random_subset = random.sample(all_combinations, min(n, len(all_combinations)))  # Adjust the number as needed

    test = {'all_combinations': all_combinations, 'subsets': random_subset}
    torch.save(test, os.path.join(results_folder, 'metadata.pt'))


    # Train models using the selected hyperparameters on all available GPUs
    n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0

    mp.set_start_method('spawn', force=True)
    processes = []
    for i, param_values in enumerate(random_subset):
        params = dict(zip(search_hyperparameters.keys(), param_values))
        name = f'{i}'
        device = f'cuda:{i % n_gpus}' if n_gpus > 0 else 'cpu'
        task = Task.named(task_name, name=name, savedir=results_folder, build_seed=0, max_lr=1e-4, **params)
        p = mp.Process(target=run_build, args=(task, device))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
    
    