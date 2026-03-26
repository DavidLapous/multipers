from importlib.metadata import version as _version
import sys

__version__ = _version("multipers")
# Doc
from . import (
    data,
    filtrations,
    grids,
    io,
    logs,
    multiparameter_module_approximation,
    plots,
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
    "io",
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
