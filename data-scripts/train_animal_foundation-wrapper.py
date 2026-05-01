import os
import subprocess
import sys

root = '/ceph/branco/Jake/training_data'

if __name__ == '__main__':

    for animal_dir in os.listdir(root):
            this_animal = os.path.join(root, animal_dir)
            if not (os.path.isdir(this_animal) and animal_dir.isdigit()):
                continue

            arg_str = ' '.join(sys.argv[1:])
            subprocess.run(f'bash/run_gpu.sh foundation-{animal_dir} data-scripts/train_animal_foundation.py {animal_dir} {arg_str}', shell=True, capture_output=False)
