"""
Importing this package (import o2s.Tasks) will:
1) import every .py module in this directory
2) collect all subclasses of o2s.task.Task defined in those modules
3) expose them as attributes of o2s.Tasks (and via from o2s.Tasks import ...)
"""

from importlib import import_module
from pkgutil import iter_modules
from pathlib import Path
import inspect

import o2s
from o2s.task import Task  # your base class
import o2s.Templates

__all__ = []  # what `from o2s.Tasks import *` will export

_pkg_dir = Path(__file__).resolve().parent


# Import every non-package module in this directory
for m in iter_modules([str(_pkg_dir)]):
    if m.ispkg:
        print(m)
        continue
    if m.name.startswith("_"):
        continue  # ignore private helpers like _util.py if you ever add them

    mod = import_module(f"{__name__}.{m.name}")  # e.g. o2s.Tasks.task1

    # Re-export Task subclasses defined in that module
    for name, obj in vars(mod).items():
        if (
            inspect.isclass(obj)
            and issubclass(obj, Task)
            and obj is not Task
            and obj.__module__ == mod.__name__  # avoid pulling in imported classes
        ):
            globals()[name] = obj
            __all__.append(name)
