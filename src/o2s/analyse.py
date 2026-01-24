import numpy as np
import pandas as pd
import sys
import time
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from typing import Dict

from scipy.optimize import minimize
from scipy.linalg import eig
import multiprocessing as mp
from torch.multiprocessing import cpu_count

import torch



import o2s


def ReTanh(x):
    return np.maximum(np.zeros_like(x), np.tanh(x))


def minimise_from_x_0(x_0: np.ndarray, u: np.ndarray, t: int, vars: Dict[str, np.ndarray], W_rec: np.ndarray, W_in: np.ndarray, b: np.ndarray, 
                      checkpoint_dir: str, verbose: bool, i: int, queue: mp.Queue, queue_i: int):
    if queue is not None:
        assert queue_i is not None

    start_time = time.time()

    if verbose:
        print(f'\tMinimising starting point {i+1}, x_0={x_0[:5]}..., u={u}, t={t}, hd={vars["hd"]}, sd={vars["sd"]}')

    def _F(x):
        return (-x + W_rec@ReTanh(x) + W_in@u + b)

    def _q(x):
        return (1/2) * np.linalg.norm(_F(x))**2


    res = minimize(_q, x_0, method='Powell', tol=10e-9, options={'maxiter': 1000})

    if verbose:
        print('\tMinimisation {} {} with value {:.4E} (in {:.2f}s)'.format(
            i+1, 'succeeded' if res.success else 'failed', res.fun, time.time() - start_time))
        
    result = {
        'i': i,
        'q': res.fun,
        'state': res.x,
        'input': u,
        'time': t,
        'vars': vars
    }
    
    if checkpoint_dir is not None:
        torch.save(result, f'{checkpoint_dir}/analyse-temp/{time.time()}-{i}.pt')

    if queue is not None:
        queue.put(queue_i)

    return result



def find_fixed_points(task, net, checkpoint_path, num_x_0=100, keep_inputs=[], input_values=[], keep_times=[], verbose=False, num_processes=None):  

    checkpoint_dir = '/'.join(checkpoint_path.split('/')[:-1])
    if not os.path.exists(f'{checkpoint_dir}/analyse-temp'):
        os.mkdir(f'{checkpoint_dir}/analyse-temp')
        
    probe = o2s.data.TaskDataset(task).get_batch()

    states = net(probe['inputs'], noise=probe['noise'])[0].cpu().detach().numpy()
    inputs = probe['inputs'].cpu().detach().numpy()
    vars = {k: v.cpu().detach().numpy() for k,v in probe['vars'].items()}
    for k,v in vars.items():
        if len(v.shape) == 1 or v.shape[1] == 1:
            assert len(v) == task.config.batch_size, 'Unexpected shape for variable {}'.format(k)
            vars[k] = np.tile(v.reshape((task.config.batch_size,1)), (1, task.config.n_timesteps))

    # Remove unwanted inputs
    inputs[:,:,[i for i in range(inputs.shape[2]) if i not in keep_inputs]] = 0
    for keep_input, input_value in zip(keep_inputs, input_values):
        if input_value is None:
            continue
        inputs[:,:,keep_input] = input_value

    # Select time slices
    T = np.tile(np.arange(task.config.n_timesteps), (task.config.batch_size, 1))
    if len(keep_times) > 0:
        T = T[:,keep_times]

    states = states[np.arange(states.shape[0])[:, None], T]
    inputs = inputs[np.arange(inputs.shape[0])[:, None], T]
    vars = {k: v[np.arange(v.shape[0])[:, None], T] for k,v in vars.items()}

    # Flatten batch/time dimensions
    states = states.reshape(-1, task.config.n_neurons)
    inputs = inputs.reshape(-1, task.config.n_inputs)
    T = T.reshape(-1)
    vars = {k: v.reshape(-1) for k,v in vars.items()}

    # Select random starting points
    random_i = np.random.permutation(states.shape[0])[:num_x_0]
    X = states[random_i]
    U = inputs[random_i]
    T = T[random_i]
    vars = {i: {k: v[random_i[i]] for k,v in vars.items()} for i in range(num_x_0)}

    W_rec, W_in, b = net.W_rec.weight.cpu().detach().numpy(), net.W_in.weight.cpu().detach().numpy(), net.W_in.bias.cpu().detach().numpy()

    c = cpu_count() if num_processes is None else num_processes
    c = min(c, num_x_0)

    if verbose:
        print('Running on {} cores'.format(c))
    
    queue = mp.Queue(maxsize=c)
    processes = []
    for i in range(c):
        queue_i = i
        p = mp.Process(target=minimise_from_x_0, args=(X[i], U[i], T[i], vars[i], W_rec, W_in, b, checkpoint_dir, verbose, i, queue, queue_i))
        p.start()
        processes.append(p)

    for next_i in range(c, num_x_0):
        queue_i = queue.get()

        processes[queue_i].join()
        p = mp.Process(target=minimise_from_x_0, args=(X[next_i], U[next_i], T[next_i], vars[next_i], W_rec, W_in, b, checkpoint_dir, verbose, next_i, queue, queue_i))
        p.start()
        processes[queue_i] = p

    for p in processes:
        p.join()

    queue.close()
    queue.join_thread()



    recover_fixed_points_from_temp(checkpoint_path)

    # if len(os.listdir(f'{checkpoint_dir}/analyse-temp'))==0:
    #     os.rmdir(f'{checkpoint_dir}/analyse-temp')






    

def recover_fixed_points_from_temp(checkpoint_path):
    checkpoint_dir = '/'.join(checkpoint_path.split('/')[:-1])

    if 'fixed_points.pt' in os.listdir(checkpoint_dir):
        result = torch.load(f'{checkpoint_dir}/fixed_points.pt', map_location='cpu')
    else:
        result = {
            'i': [],
            'q': [],
            'state': [],
            'input': [],
            'time': [],
            'vars': {}
        }

    n_recovered = 0
    for tempfile in os.listdir(f'{checkpoint_dir}/analyse-temp'):
        if '.pt' not in tempfile:
            continue
        tempresult = torch.load(f'{checkpoint_dir}/analyse-temp/{tempfile}', map_location='cpu')
        for key in result.keys():
            if key=='vars':
                for vars_key in tempresult['vars'].keys():
                    if vars_key not in result['vars']:
                        result['vars'][vars_key] = []
                    result['vars'][vars_key].append(tempresult['vars'][vars_key])
            else:
                result[key].append(tempresult[key])
        n_recovered += 1
        os.remove(f'{checkpoint_dir}/analyse-temp/{tempfile}')

    print(f'Recovered {n_recovered} points')

    torch.save(result, f'{checkpoint_dir}/fixed_points.pt')


if __name__ == '__main__':
    import o2s.Tasks

    assert len(sys.argv) >= 2

    checkpoint_path = sys.argv[1]

    print(f'Analysing {checkpoint_path}')

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    task = o2s.task.Task.from_checkpoint(checkpoint)
    task.config.update(device='cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_default_dtype(torch.float64 if task.config.precise else torch.float32)

    if '-r' in sys.argv:
        recover_fixed_points_from_temp(checkpoint_path)

    else:

        num_x_0 = 100
        if '-n' in sys.argv:
            argi = sys.argv.index('-n')
            num_x_0 = int(sys.argv[argi+1])

        keep_inputs = []
        if '-u' in sys.argv:
            argi = sys.argv.index('-u')
            for input_name in sys.argv[argi+1:]:
                if input_name[0] == '-':
                    break
                keep_inputs.append(task.input_map[input_name])

        input_values = []
        if '-v' in sys.argv:
            assert len(keep_inputs) > 0, 'Must specify inputs to set values for'
            argi = sys.argv.index('-v')
            for input_value in sys.argv[argi+1:]:
                if input_value[0] == '-':
                    break
                elif input_value == '.':
                    input_values.append(None)
                else:
                    input_values.append(float(input_value))

        assert len(keep_inputs) == len(input_values), 'Number of inputs and values must match'

        if '-s' in sys.argv:
            argi = sys.argv.index('-s')
            task = task.get_subtask(sys.argv[argi+1])


        keep_times = []
        if '-t' in sys.argv:
            argi = sys.argv.index('-t')
            for keep_time in sys.argv[argi+1:]:
                if keep_time[0] == '-':
                    break
                elif ':' in keep_time:
                    start, end = keep_time.split(':')
                    keep_times.extend(list(range(int(start), int(end))))
                else:
                    keep_times.append(int(keep_time))

        task.config.update(batch_size=num_x_0, 
                           n_timesteps=task.config.n_timesteps if (len(keep_times)==0 or '-s' in sys.argv) else max(max(keep_times)+1, task.config.init_duration+1))
        net = o2s.net.RNN(task)
        net.load_state_dict(checkpoint['net_state_dict'])
        assert (net.W_rec.weight.dtype == torch.float64) if task.config.precise else (net.W_rec.weight.dtype == torch.float32)

        find_fixed_points(task, net, 
                          checkpoint_path=checkpoint_path, 
                          num_x_0=num_x_0, 
                          keep_inputs=keep_inputs, 
                          input_values=input_values,
                          keep_times=keep_times, verbose=True)
    

