# src/o2s/__init__.py

from importlib import import_module
from pkgutil import iter_modules
from pathlib import Path
import inspect

def is_defined_here(name, obj, modname):
    # functions/classes: check __module__
    if inspect.isfunction(obj) or inspect.isclass(obj):
        return getattr(obj, "__module__", None) == modname
    # constants/dicts: no reliable provenance; accept them by name convention
    return True

__all__ = []  # what `from o2s import *` will export
_pkg_dir = Path(__file__).resolve().parent

# task must be imported before build (because build depends on Task)
PRIORITY = ["task", "net"]
EXCLUDE = {"build"}

def _export_public(mod):
    """
    Re-export public names from `mod` onto the o2s package namespace,
    WITHOUT overwriting already-defined names (e.g. module attributes like o2s.build).
    """
    for name, obj in vars(mod).items():
        if name.startswith("_"):
            continue
        if inspect.ismodule(obj):
            continue
        if not is_defined_here(name, obj, mod.__name__):
            continue

        # Critical: do not clobber existing package attributes (like the module itself)
        if name in globals():
            continue

        globals()[name] = obj
        __all__.append(name)

def _import_and_export(module_name: str):
    """
    Import o2s.<module_name> and bind it as o2s.<module_name>,
    so you can do o2s.build.build(), o2s.task.Task, etc.
    """
    mod = import_module(f"{__name__}.{module_name}")

    # Bind module object first (so o2s.build is the build module)
    globals()[module_name] = mod
    if module_name not in __all__:
        __all__.append(module_name)

    # Optionally also lift public symbols to top-level (but never overwriting modules)
    _export_public(mod)

# 1) Import/export priority modules first
_seen = set()
for mn in PRIORITY:
    if (_pkg_dir / f"{mn}.py").exists():
        _import_and_export(mn)
        _seen.add(mn)

# 2) Import/export all other non-package modules in this directory (exclude Tasks/)
for m in iter_modules([str(_pkg_dir)]):
    if m.ispkg:
        continue              # skip subpackages like Tasks/
    if m.name.startswith("_"):
        continue
    if m.name in _seen:
        continue
    if m.name in EXCLUDE:
        continue
    if m.name == "__init__":
        continue

    _import_and_export(m.name)
    _seen.add(m.name)
