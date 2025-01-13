from importlib.metadata import version as _version

__version__ = _version("multipers")
# Doc
from multipers import (
    data,
    grids,
    io,
    multiparameter_module_approximation,
    simplex_tree_multi,
    slicer,
)

# Shortcuts
from multipers._slicer_meta import Slicer
from multipers.simplex_tree_multi import SimplexTreeMulti
from multipers._signed_measure_meta import signed_measure
from multipers.multiparameter_module_approximation import module_approximation

__all__ = [
    "signed_measure",
    "module_approximation",
    "Slicer",
    "SimplexTreeMulti",
    "data",
    "grids",
    "io",
    "multiparameter_module_approximation",
    "slicer",
    "simplex_tree_multi",
]
