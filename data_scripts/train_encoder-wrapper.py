import os
import subprocess
import sys

root = '/ceph/branco/Jake/training_data'

if __name__ == '__main__':

    for animal_dir in os.listdir(root):
            this_animal = os.path.join(root, animal_dir)
            if not os.path.isdir(this_animal):
                continue

            for session_dir in os.listdir(this_animal):
                this_session = os.path.join(this_animal, session_dir)
                if not os.path.isdir(this_session):
                    continue

                if 'data.csv' in list(os.listdir(this_session)):

                    arg_str = ' '.join(sys.argv[1:])
                    subprocess.run(f'bash/run_gpu.sh enc-{animal_dir}-{session_dir} data-scripts/train_encoder.py {animal_dir} {session_dir} {arg_str}', shell=True, capture_output=False)
