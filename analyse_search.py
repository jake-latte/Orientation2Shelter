

import sys
import os
import numpy as np
import pandas as pd
import torch
import mmap
import shutil

from search import search_hyperparameters
from o2s.Test import test_gamut
import gc


savedir = sys.argv[1]
assert os.path.isdir(savedir)

can_skip_package =  os.path.exists(os.path.join(savedir, 'package', 'results.csv'))

if '-P' in sys.argv or not can_skip_package:
    print(f'Creating package for {savedir}')
    if not os.path.isdir(os.path.join(savedir, 'package')):
        os.makedirs(os.path.join(savedir, 'package'))

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
                    if os.path.isfile(os.path.join(savedir, 'package', build['name'], f)):
                        os.remove(os.path.join(savedir, 'package', build['name'], f))
            else:
                os.makedirs(os.path.join(savedir, 'package', build['name']), exist_ok=True)

            shutil.copyfile(os.path.join(savedir, build['name'], 'build.out'), os.path.join(savedir, 'package', build['name'], 'build.out'))
            shutil.copyfile(os.path.join(savedir, build['name'], build['checkpoint_name']), os.path.join(savedir, 'package', build['name'], 'net.pt'))
    
else:
    print('Using existing package')
    df = pd.read_csv(os.path.join(savedir, 'package', 'results.csv'))


if '-T' in sys.argv:
    start_i = 0 if '-i' not in sys.argv else int(sys.argv[sys.argv.index('-i') + 1])
    print(f'Testing converged models in {savedir}')
    for i, build in df.iterrows():
        if i< start_i:
            continue
        if build['converged']:
            print(f'Testing {build}')

            if os.path.exists(os.path.join(savedir, 'package', build['name'], 'figures')):
                continue

            os.makedirs(os.path.join(savedir, 'package', build['name'], 'figures'))

            checkpoint_path = os.path.join(savedir, 'package', build['name'], 'net.pt')
            with torch.no_grad():
                figures = test_gamut(checkpoint_path,
                                     include_stability=False, include_lesions=False, include_fourier=False, include_eigenspectra=False)
                del figures

            gc.collect()
            torch.cuda.empty_cache()
    
