from importlib.metadata import version as _version

__version__ = _version("multipers")
# Doc
from multipers import (
    data,
    filtrations,
    grids,
    io,
    multiparameter_module_approximation,
    simplex_tree_multi,
    slicer,
)
from multipers._signed_measure_meta import signed_measure

# Shortcuts
from multipers._slicer_meta import Slicer
from multipers.multiparameter_module_approximation import module_approximation
from multipers.simplex_tree_multi import SimplexTreeMulti

__all__ = [
    "data",
    "filtrations",
    "grids",
    "io",
    "multiparameter_module_approximation",
    "simplex_tree_multi",
    "slicer",
    "signed_measure",
    "Slicer",
    "module_approximation",
    "SimplexTreeMulti",
]
