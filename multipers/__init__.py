from importlib import import_module as _import_module
from importlib.metadata import version as _version
import sys

__version__ = _version("multipers")
# Doc
from . import (
    filtrations,
    grids,
    logs,
    multiparameter_module_approximation,
    simplex_tree_multi,
    slicer,
)

if sys.platform != "win32":
    from . import ops


# Shortcuts
from ._slicer_meta import Slicer
from .multiparameter_module_approximation import module_approximation
from .simplex_tree_multi import SimplexTreeMulti
from ._signed_measure_meta import signed_measure

__all__ = [
    "data",
    "filtrations",
    "grids",
    "logs",
    "multiparameter_module_approximation",
    "plots",
    "simplex_tree_multi",
    "slicer",
    "signed_measure",
    "Slicer",
    "module_approximation",
    "SimplexTreeMulti",
]

if sys.platform != "win32":
    __all__.append("ops")


def __getattr__(name):
    if name in {"data", "plots"}:
        module = _import_module(f".{name}", __name__)
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
