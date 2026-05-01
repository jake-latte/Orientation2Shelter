import os
import subprocess
import sys

root = '/ceph/branco/Jake/training_data'

if __name__ == '__main__':

    arg_str = ' '.join(sys.argv[1:])
    subprocess.run(f'bash/run_gpu.sh foundation-all data-scripts/train_foundation_wide.py {arg_str}', shell=True, capture_output=False)
