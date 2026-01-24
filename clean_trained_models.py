import os

for model_dir in os.listdir('trained-models'):
    if not os.path.isdir(os.path.join('trained-models', model_dir)):
        continue

    non_checkpoint_files = []
    has_checkpoint_files = False
    for file in os.listdir(os.path.join('trained-models', model_dir)):
        if 'checkpoint' in file:
            has_checkpoint_files = True
        else:
            non_checkpoint_files.append(file)
    
    if not has_checkpoint_files:
        print(f"Removing {model_dir}")
        if len(non_checkpoint_files)==0 or (len(non_checkpoint_files)==1 and non_checkpoint_files[0]=='build.out'):
            os.system(f"rm -r trained-models/{model_dir}")
            