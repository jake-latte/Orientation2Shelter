

import sys
import os
import numpy as np
import pandas as pd
import torch
import mmap
import shutil

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


savedir = sys.argv[1]
assert os.path.isdir(savedir)

print(f'Checking convergence for {savedir}')

results = {
    'name': [],
    'converged': [],
    'checkpoint_name': [],
    'n_epochs': [],
    'test_loss': [],
    **{param: [] for param in search_hyperparameters}
}

for builddir in os.listdir(savedir):
    if not os.path.isdir(os.path.join(savedir, builddir)):
        continue
    
    if 'build.out' not in os.listdir(os.path.join(savedir, builddir)):
        continue

    with open(os.path.join(savedir, builddir, 'build.out'), 'r') as outfile:
        converged = None
        checkpoint_name = None
        with mmap.mmap(outfile.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            for i, line in enumerate(reversed(mm.read().splitlines())):
                if i == 0:
                    converged = b'converged' in line
                else:
                    if b'Saved model' in line:
                        checkpoint_name_ = line.decode().split(' ')[-1]
                        checkpoint_name_ = checkpoint_name_.replace('=', ':')
                        checkpoint_name = f'checkpoint-{checkpoint_name_}/net.pt'
                        break
        
        n_epochs, test_loss, params = None, None, {p: None for p in search_hyperparameters}
        if checkpoint_name is not None:
            checkpoint = torch.load(os.path.join(savedir, builddir, checkpoint_name), map_location='cpu')
            n_epochs = len(checkpoint['train_losses'])
            test_loss = np.mean(np.array(checkpoint['test_losses'])[-min(checkpoint['config']['training_convergence_std_threshold_window'], len(checkpoint['test_losses'])):]).item()
            params = {}
            for param in search_hyperparameters:
                params[param] = checkpoint['config'][param]

        results['name'].append(builddir)
        results['converged'].append(converged)
        results['checkpoint_name'].append(checkpoint_name)
        results['n_epochs'].append(n_epochs)
        results['test_loss'].append(test_loss)
        for param in search_hyperparameters:
            results[param].append(params[param])

df = pd.DataFrame(results)
print(df)


if not os.path.isdir(os.path.join(savedir, 'package')):
    os.makedirs(os.path.join(savedir, 'package'))

df.to_csv(os.path.join(savedir, 'package', 'results.csv'), index=False)

for i, build in df.iterrows():
    if build['checkpoint_name'] is not None:
        if os.path.isdir(os.path.join(savedir, 'package', build['name'])):
            for f in os.listdir(os.path.join(savedir, 'package', build['name'])):
                os.remove(os.path.join(savedir, 'package', build['name'], f))
        else:
            os.makedirs(os.path.join(savedir, 'package', build['name']), exist_ok=True)
    


        shutil.copyfile(os.path.join(savedir, build['name'], 'build.out'), os.path.join(savedir, 'package', build['name'], 'build.out'))
        shutil.copyfile(os.path.join(savedir, build['name'], build['checkpoint_name']), os.path.join(savedir, 'package', build['name'], 'net.pt'))

            
    