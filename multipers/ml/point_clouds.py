"""Backward-compatible import path for filtered complexes."""

from warnings import warn

from multipers.ml.filtered_complex import (
    FilteredComplexPreprocess,
    PointCloud2FilteredComplex,
    PointCloud2SimplexTree,
)

warn(
    "multipers.ml.point_clouds is deprecated; use multipers.ml.filtered_complex",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "PointCloud2FilteredComplex",
    "PointCloud2SimplexTree",
    "FilteredComplexPreprocess",
]
