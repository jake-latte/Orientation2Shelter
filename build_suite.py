import os
import time
import itertools
import sys

timestamp = time.time()
parentdir = sys.argv[1] if sys.argv[1] != '-n' else f'builds-{time.time()}'
print(parentdir)

if not os.path.exists(parentdir):
    os.mkdir(parentdir)

global_params = {
    'training_threshold': 0.0,
    'training_convergence_std_threshold': 0.001,
    'training_convergence_std_threshold_window': 1000,
    'learning_rate_schedule': 0.75,
    'optimiser_name': 'HF',
    'max_learning_rate': 0.01,
    'batch_size': 500,
    'minibatch_size': 500,
    'num_batch_repeats': 1,
    'n_timesteps': 1000,
    'test_batch_size': 1000,
    'test_n_timesteps': 1000,
    'n_epochs': 100,
    'save_updates': 100,
    'print_interval': 1
}

task_type_params = {
}

search_params = {
    'weight_lambda': [0.0, 0.1],
    'rate_lambda': [0.0, 0.1]
}

all_param_combinations = list(itertools.product(search_params['weight_lambda'], search_params['rate_lambda']))

tasks = [
    ('HD', '0D'), ('HD', '1D'), ('HD', '2D'),
    ('SD', '0D'), ('SD', '1D'), ('SD', '2D'),
    ('SD', '0D_trans'), ('SD', '1D_trans'), ('SD', '2D_trans'),
    ('HD_SD', '0D'), ('HD_SD', '1D'), ('HD_SD', '2D'),
    ('PI', '1D'), ('PI', '2D'),
    ('PI_HD', '1D'), ('PI_HD', '2D'),
    ('PI_SD', '1D'), ('PI_SD', '2D'),
    ('PI_HD_SD', '1D'), ('PI_HD_SD', '2D'),
]

queue = sys.argv[2]
if queue == 'gpu':
    global_params['max_hours'] = 48
elif queue == 'normal' or queue == 'highmem':
    global_params['max_hours'] = 120
elif queue == 'large':
    global_params['max_hours'] = 240

start_i = int(sys.argv[3])
end_i = int(sys.argv[4])
i = -1

for j, params in enumerate(all_param_combinations):

    print(f'{j+1}: {params}')

    for output_type, input_type in tasks:
        i += 1

        args = f'{output_type}-{input_type} -N'
        paramsdir = ''
        for arg_name, arg_val in zip(search_params.keys(), params):
            args += f' -{arg_name} {arg_val}'
            paramsdir += f'{arg_name}:{arg_val}-'

        savedir = f'{parentdir}/{paramsdir[:-1]}'
        args += f' -savedir {savedir}'

        if input_type in task_type_params:
            for arg_name, arg_val in task_type_params[input_type].items():
                args += f' -{arg_name} {arg_val}' 

        if output_type in task_type_params:
            for arg_name, arg_val in task_type_params[output_type].items():
                args += f' -{arg_name} {arg_val}' 

        for arg_name, arg_val in global_params.items():
            args += f' -{arg_name} {arg_val}' 

        task = args.split('-N')[0].strip(' ')

        if start_i <= i < end_i:

            existing_builddir = None
            for dir in os.listdir(f'{savedir}'):
                dir_task = dir.split('task:')
                if len(dir_task) != 2:
                    continue
                else:
                    dir_task = dir_task[1]
                if task == dir_task and os.path.isdir(f'{savedir}/{dir}'):
                    existing_builddir = f'{savedir}/{dir}'
            
            last_checkpoint, last_checkpoint_num = None, None
            if existing_builddir is not None:
                for dir in os.listdir(existing_builddir):
                    if 'checkpoint-updates' in dir:
                        update_num = int(dir.split(':')[1])
                        if last_checkpoint_num is None or update_num > last_checkpoint_num:
                            last_checkpoint_num = update_num
                            last_checkpoint = f'{existing_builddir}/{dir}/net.pt'

            if last_checkpoint is not None:
                args = f'-c {last_checkpoint}'

            print(f'{task} ({i})')

            # if input('skip? ') == 'y':
            #     continue

            print(args)
            os.system(f'bash build_{queue}.sh {args}')

            # os.system(f'python3 -m build {args}')
            # print(f'python3 -m build {args}')

            # if input('quit? ') == 'y':
            #     quit()