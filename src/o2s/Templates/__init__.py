"""
Importing this package (import o2s.Template) will:
1) import every .py module in this directory
2) collect all functions defined in those modules
3) expose them as attributes of o2s.Templates (and via from o2s.Templates import ...)
"""

from importlib import import_module
from pkgutil import iter_modules
from pathlib import Path
import inspect

__all__ = []  # what `from o2s.Templates import *` will export

_pkg_dir = Path(__file__).resolve().parent

# Import every non-package module in this directory
for m in iter_modules([str(_pkg_dir)]):
    if m.ispkg:
        print(m)
        continue
    if m.name.startswith("_"):
        continue  # ignore private helpers like _util.py if you ever add them

    mod = import_module(f"{__name__}.{m.name}")  # e.g. o2s.Templates.vars_0D
