import os
import importlib

def register_all_tasks():
    for module_name in os.listdir('Tasks'):
        if '.py' in module_name and module_name != 'util.py':
            print(f'importing {module_name} as {module_name[:-3]}')
            importlib.import_module(f'Tasks.{module_name[:-3]}')