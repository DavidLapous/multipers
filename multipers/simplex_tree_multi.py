from __future__ import annotations

from copy import deepcopy
from functools import reduce
from operator import or_
from typing import Any, Iterable, Literal, Optional, Sequence
from warnings import warn
from itertools import chain, combinations

import gudhi as gd
import numpy as np
from gudhi.simplex_tree import SimplexTree as _GudhiSimplexTree

from multipers import _simplex_tree_multi_nanobind as _nb
from multipers.array_api import api_from_tensor
from multipers.grids import Lstrategies, compute_grid, sanitize_grid
from multipers.point_measure import sparsify
import multipers.logs as _mp_logs


SAFE_CONVERSION = False
_available_strategies = Lstrategies

available_simplextrees = tuple(_nb.available_simplextrees)
for _cls in available_simplextrees:
    globals()[_cls.__name__] = _cls

available_dtype = {cls().dtype for cls in available_simplextrees}
SimplexTreeMulti_type = (
    reduce(or_, available_simplextrees) if available_simplextrees else Any
)


def _t_minus_inf(dtype):
    dtype = np.dtype(dtype)
    if np.issubdtype(dtype, np.floating):
        return dtype.type(-np.inf)
    return np.iinfo(dtype).min


def _reconstruct_gudhi_simplextree(state):
    np_state = np.asarray(state, dtype=np.int8)
    simplex_tree = _GudhiSimplexTree.__new__(_GudhiSimplexTree)
    simplex_tree.__setstate__(np_state)
    return simplex_tree


def _normalize_filtration_value(self, filtration, *, copy=False):
    if not self.is_kcritical:
        arr = np.asarray(filtration, dtype=self.dtype)
        return np.array(arr, copy=True) if copy else arr
    rows = [np.asarray(row, dtype=self.dtype) for row in filtration]
    rows = [row for row in rows if not np.isinf(row).all()]
    if self.filtration_container != "Flat":
        rows.sort(key=lambda row: tuple(row.tolist()))
    return [np.array(row, copy=True) for row in rows] if copy else rows


def _coords_from_filtration(filtration, grid, dtype=np.int32):
    arr = np.asarray(filtration, dtype=float)
    original_ndim = arr.ndim
    if original_ndim == 1:
        arr = arr[None, :]
    out = np.empty(arr.shape, dtype=dtype)
    for p, g in enumerate(grid):
        values = arr[:, p]
        coords = np.searchsorted(np.asarray(g), values, side="left")
        coords = coords.astype(dtype, copy=False)
        coords[np.isneginf(values)] = -1
        coords[np.isposinf(values)] = len(g)
        out[:, p] = coords
    return out[0] if original_ndim == 1 else out


def _values_from_coords(filtration, grid, dtype=np.float64):
    arr = np.asarray(filtration, dtype=np.int64)
    original_ndim = arr.ndim
    if original_ndim == 1:
        arr = arr[None, :]
    out = np.empty(arr.shape, dtype=dtype)
    for p, g in enumerate(grid):
        values = arr[:, p]
        out[:, p] = np.asarray(g, dtype=dtype)[np.clip(values, 0, len(g) - 1)]
        out[values < 0, p] = -np.inf
        out[values >= len(g), p] = np.inf
    return out[0] if original_ndim == 1 else out


def _rebuild_from_current_simplices(self, target_cls, transform):
    out = target_cls()
    out.set_num_parameter(self.num_parameters)
    simplices = list(self.get_simplices())
    max_dim = max((len(simplex) - 1 for simplex, _ in simplices), default=-1)
    for dim in range(max_dim + 1):
        simplices_dim = [
            np.asarray(simplex, dtype=np.int32)
            for simplex, _ in simplices
            if len(simplex) - 1 == dim
        ]
        if not simplices_dim:
            continue
        vertex_array = np.asarray(simplices_dim, dtype=np.int32).T
        if self.is_kcritical:
            filtration_values = [
                transform(filtration)
                for simplex, filtration in simplices
                if len(simplex) - 1 == dim
            ]
            max_crit = max(np.asarray(f).shape[0] for f in filtration_values)
            packed = np.empty(
                (len(filtration_values), max_crit, self.num_parameters),
                dtype=np.asarray(filtration_values[0]).dtype,
            )
            for i, filtration in enumerate(filtration_values):
                arr = np.asarray(filtration)
                packed[i, : arr.shape[0], :] = arr
                if arr.shape[0] < max_crit:
                    packed[i, arr.shape[0] :, :] = arr[-1]
            out.insert_batch(vertex_array, packed)
        else:
            filtrations = np.asarray(
                [
                    transform(filtration)
                    for simplex, filtration in simplices
                    if len(simplex) - 1 == dim
                ]
            )
            out.insert_batch(vertex_array, filtrations)
    out._is_function_simplextree = self._is_function_simplextree
    return out


def _get_class(dtype, kcritical=False, ftype="Contiguous"):
    dtype = np.dtype(dtype)
    for cls in available_simplextrees:
        sample = cls()
        if (
            np.dtype(sample.dtype) == dtype
            and sample.is_kcritical == kcritical
            and sample.filtration_container.lower() == str(ftype).lower()
        ):
            return cls
    raise TypeError(
        f"No SimplexTreeMulti implementation for dtype={dtype}, kcritical={kcritical}, ftype={ftype}."
    )


def is_simplextree_multi(input) -> bool:
    return any(isinstance(input, cls) for cls in available_simplextrees)


def _safe_simplextree_multify(
    simplextree: gd.SimplexTree, cls, num_parameters=2, default_values=None
):
    simplices = [[] for _ in range(simplextree.dimension() + 1)]
    filtration_values = [[] for _ in range(simplextree.dimension() + 1)]
    st_multi = cls()
    st_multi.set_num_parameter(num_parameters)
    if default_values is None:
        default_values = np.zeros(num_parameters - 1, dtype=cls().dtype) + _t_minus_inf(
            cls().dtype
        )
    default_values = np.asarray(default_values, dtype=cls().dtype)
    if default_values.squeeze().ndim == 0:
        default_values = (
            np.zeros(num_parameters - 1, dtype=cls().dtype) + default_values
        )
    for simplex, filtration in simplextree.get_simplices():
        filtration_values[len(simplex) - 1].append(
            np.concatenate([[filtration], default_values]).astype(
                cls().dtype, copy=False
            )
        )
        simplices[len(simplex) - 1].append(simplex)
    for batch_simplices, batch_filtrations in zip(simplices, filtration_values):
        if not batch_simplices:
            continue
        st_multi.insert_batch(
            np.asarray(batch_simplices, dtype=np.int32).T,
            np.asarray(batch_filtrations, dtype=cls().dtype),
        )
    return st_multi


def SimplexTreeMulti(
    input=None,
    num_parameters: int = -1,
    dtype: type = np.float64,
    kcritical: bool = False,
    ftype="Contiguous",
    default_values=None,
    safe_conversion: bool = False,
    max_dim: int = -1,
    return_type_only: bool = False,
    **kwargs,
) -> SimplexTreeMulti_type:
    cls = _get_class(dtype, kcritical, ftype)
    if return_type_only:
        return cls

    if input is None:
        out = cls()
        out.set_num_parameter(2 if num_parameters <= 0 else num_parameters)
        return out

    if is_simplextree_multi(input):
        out = cls()
        out._copy_from_any(input)
        if num_parameters > 0 and num_parameters != input.num_parameters:
            out.set_num_parameter(num_parameters)
        return out

    from multipers.slicer import is_slicer

    if is_slicer(input, allow_minpres=False):
        out = cls()
        out._from_slicer(input, max_dim=max_dim)
        if num_parameters > 0 and num_parameters != input.num_parameters:
            out.set_num_parameter(num_parameters)
        return out

    if isinstance(input, gd.SimplexTree):
        if num_parameters <= 0:
            num_parameters = 1
        if default_values is None:
            default_values = np.asarray(
                [_t_minus_inf(dtype)] * num_parameters, dtype=dtype
            )
            default_values[0] = dtype(0)
        else:
            default_values = np.asarray(default_values, dtype=dtype)
            if default_values.ndim == 0:
                default_values = np.asarray([default_values], dtype=dtype)
            if len(default_values) > num_parameters:
                default_values = default_values[:num_parameters]
            elif len(default_values) < num_parameters:
                padding = np.full(
                    num_parameters - len(default_values),
                    _t_minus_inf(dtype),
                    dtype=dtype,
                )
                default_values = np.concatenate((padding, default_values))

        if safe_conversion or SAFE_CONVERSION:
            return _safe_simplextree_multify(
                input,
                cls,
                num_parameters=num_parameters,
                default_values=default_values[1:],
            )
        out = cls()
        try:
            out._from_gudhi_state(input.__getstate__(), num_parameters, default_values)
            return out
        except Exception:
            return _safe_simplextree_multify(
                input,
                cls,
                num_parameters=num_parameters,
                default_values=default_values[1:],
            )

    raise TypeError(
        "`input` requires to be of type `SimplexTree`, `SimplexTreeMulti`, `Slicer`, or `None`."
    )


def _repr(self):
    return f"SimplexTreeMulti[dtype={np.dtype(self.dtype).name},num_param={self.num_parameters},kcritical={self.is_kcritical},is_squeezed={self.is_squeezed},max_dim={self.dimension}]"


def _len(self):
    return self.num_simplices


def _copy(self):
    stree = type(self)()
    stree._copy_from_any(self)
    stree.filtration_grid = self.filtration_grid
    stree._is_function_simplextree = self._is_function_simplextree
    return stree


def _deepcopy(self, memo=None):
    return self.copy()


def _filtration(self, simplex):
    return self[simplex]


def _getitem(self, simplex):
    return _normalize_filtration_value(self, self._get_filtration(simplex), copy=False)


def _iter(self):
    yield from self.get_simplices()


def _getstate(self):
    return self._serialize_state(), self.filtration_grid, self._is_function_simplextree


def _setstate(self, state):
    if isinstance(state, tuple) and len(state) == 3:
        serialized, filtration_grid, is_function = state
    else:
        serialized, filtration_grid, is_function = state, [], False
    self._deserialize_state(serialized)
    self.filtration_grid = filtration_grid
    self._is_function_simplextree = is_function


def _reconstruct_from_pickle(cls, state):
    obj = cls()
    obj.__setstate__(state)
    return obj


def _reduce(self):
    return (_reconstruct_from_pickle, (type(self), self.__getstate__()))


def _astype(self, dtype=None, kcritical=None, ftype=None, filtration_container=None):
    dtype = self.dtype if dtype is None else dtype
    kcritical = self.is_kcritical if kcritical is None else kcritical
    if filtration_container is not None and ftype is not None:
        if str(filtration_container).lower() != str(ftype).lower():
            raise ValueError(
                "Got conflicting `ftype` and `filtration_container` arguments."
            )
    if filtration_container is not None:
        ftype = filtration_container
    ftype = self.filtration_container if ftype is None else ftype

    cls = _get_class(dtype, kcritical, ftype)
    if cls is type(self):
        return self

    out = cls()
    out._copy_from_any(self)
    out.filtration_grid = self.filtration_grid
    out._is_function_simplextree = self._is_function_simplextree
    return out


def _insert(self, simplex, filtration=None):
    num_parameters = self.num_parameters
    if filtration is None:
        filtration = np.array(
            [_t_minus_inf(self.dtype)] * num_parameters, dtype=self.dtype
        )
    filtration = np.asarray(filtration, dtype=self.dtype)
    if self.is_kcritical and filtration.ndim == 1:
        if simplex in self:
            simplex = tuple(simplex)

            def powerset(iterable):
                s = tuple(iterable)
                return chain.from_iterable(
                    combinations(s, r) for r in range(1, len(s) + 1)
                )

            for face in powerset(simplex):
                current = [
                    np.asarray(row, dtype=self.dtype) for row in self[list(face)]
                ]
                current.append(filtration)
                self._assign_filtration(
                    list(face), np.asarray(current, dtype=self.dtype)
                )
            return True
        filtration = filtration[None, :]
    return self._insert_simplex(np.asarray(simplex, dtype=np.int32), filtration, False)


def _assign_filtration(self, simplex, filtration):
    filtration = np.asarray(filtration, dtype=self.dtype)
    if self.is_kcritical and filtration.ndim == 1:
        filtration = filtration[None, :]
    self._assign_filtration(np.asarray(simplex, dtype=np.int32), filtration)
    return self


def _insert_batch(self, vertex_array, filtrations=np.empty((0, 0))):
    vertex_array = np.asarray(vertex_array, dtype=np.int32)
    if vertex_array.size == 0:
        return self
    n = vertex_array.shape[1]
    empty_filtration = np.size(filtrations) == 0
    if not self.is_kcritical:
        filtrations = (
            np.asarray(filtrations, dtype=self.dtype)
            if not empty_filtration
            else filtrations
        )
        for i in range(n):
            filtration = None if empty_filtration else filtrations[i]
            self._insert_simplex(
                vertex_array[:, i],
                None if filtration is None else filtration,
                False,
            )
        if empty_filtration:
            self.make_filtration_non_decreasing()
        return self

    filtrations = (
        np.asarray(filtrations, dtype=self.dtype)
        if not empty_filtration
        else filtrations
    )
    for i in range(n):
        filtration = None if empty_filtration else filtrations[i]
        self._insert_simplex(
            vertex_array[:, i],
            None if filtration is None else filtration,
            True,
        )
    if empty_filtration:
        self.make_filtration_non_decreasing()
    return self


def _get_simplices(self):
    simplices = []
    for simplex, filtration in self._iter_simplices():
        simplices.append(
            (
                np.asarray(simplex, dtype=np.int32),
                _normalize_filtration_value(self, filtration, copy=True),
            )
        )
    return simplices


def _get_skeleton(self, dimension):
    simplices = []
    for simplex, filtration in self._get_skeleton(dimension):
        simplices.append(
            (
                np.asarray(simplex, dtype=np.int32),
                _normalize_filtration_value(self, filtration, copy=True),
            )
        )
    return simplices


def _get_boundaries(self, simplex):
    boundaries = []
    for face, filtration in self._get_boundaries(np.asarray(simplex, dtype=np.int32)):
        boundaries.append(
            (
                np.asarray(face, dtype=np.int32),
                _normalize_filtration_value(self, filtration, copy=True),
            )
        )
    return boundaries


def _flagify(self, dim=2):
    minus_inf = np.asarray(
        [_t_minus_inf(self.dtype)] * self.num_parameters, dtype=self.dtype
    )
    for simplex, filtration in list(self.get_simplices()):
        if len(simplex) - 1 >= dim:
            self._assign_filtration(simplex, minus_inf)
    self.make_filtration_non_decreasing()
    return self


def _num_vertices_prop(self):
    return _num_vertices_raw[type(self)](self)


def _num_simplices_prop(self):
    return _num_simplices_raw[type(self)](self)


def _dimension_prop(self):
    return _dimension_raw[type(self)](self)


def _upper_bound_dimension(self):
    return _upper_bound_dimension_raw[type(self)](self)


def _simplex_dimension(self, simplex):
    return _simplex_dimension_raw[type(self)](self, np.asarray(simplex, dtype=np.int32))


def _contains(self, simplex):
    if len(simplex) == 0:
        return False
    if isinstance(simplex[0], Iterable):
        s, f = simplex
        if not _find_simplex_raw[type(self)](self, np.asarray(s, dtype=np.int32)):
            return False
        return np.all(np.asarray(f) >= np.asarray(self[s]))
    return _find_simplex_raw[type(self)](self, np.asarray(simplex, dtype=np.int32))


def _remove_maximal_simplex(self, simplex):
    _remove_maximal_simplex_raw[type(self)](self, np.asarray(simplex, dtype=np.int32))
    return self


def _prune_above_dimension(self, dimension):
    return _prune_above_dimension_raw[type(self)](self, int(dimension))


def _expansion(self, max_dim):
    _expansion_raw[type(self)](self, int(max_dim))
    return self


def _make_filtration_non_decreasing(self):
    return _make_filtration_non_decreasing_raw[type(self)](self)


def _reset_filtration(self, filtration, min_dim=0):
    _reset_filtration_raw[type(self)](
        self, np.asarray(filtration, dtype=self.dtype), min_dim
    )
    return self


def _get_simplices_of_dimension(self, dim):
    return _get_simplices_of_dimension_raw[type(self)](self, dim)


def _key(self, simplex):
    return _get_key_raw[type(self)](self, np.asarray(simplex, dtype=np.int32))


def _set_keys_to_enumerate(self):
    _set_keys_to_enumerate_raw[type(self)](self)
    return None


def _set_key(self, simplex, key):
    _set_key_raw[type(self)](self, np.asarray(simplex, dtype=np.int32), key)
    return None


def _to_scc_python_fallback(self, filtration_dtype=None, flattened=False):
    filtration_dtype = self.dtype if filtration_dtype is None else filtration_dtype
    if flattened:
        simplices = list(self.get_simplices())
        filtrations = np.asarray(
            [np.asarray(f, dtype=filtration_dtype).reshape(-1) for _, f in simplices],
            dtype=filtration_dtype,
        )
        simplex_to_index = {
            tuple(np.asarray(simplex, dtype=np.int32).tolist()): i
            for i, (simplex, _) in enumerate(simplices)
        }
        boundaries = []
        for simplex, _ in simplices:
            simplex_tuple = tuple(np.asarray(simplex, dtype=np.int32).tolist())
            if len(simplex_tuple) <= 1:
                boundaries.append(tuple())
            else:
                faces = [
                    tuple(np.asarray(face, dtype=np.int32).tolist())
                    for face, _ in self.get_boundaries(simplex)
                ]
                boundaries.append(tuple(simplex_to_index[face] for face in faces))
        return filtrations, tuple(boundaries)

    simplices = list(self.get_simplices())
    max_dim = max((len(simplex) - 1 for simplex, _ in simplices), default=-1)
    blocks = []
    previous_index = {}
    for dim in range(max_dim + 1):
        simplices_dim = []
        filtrations_dim = []
        boundaries_dim = []
        current_index = {}
        for simplex, filtration in simplices:
            simplex_tuple = tuple(np.asarray(simplex, dtype=np.int32).tolist())
            if len(simplex_tuple) - 1 != dim:
                continue
            current_index[simplex_tuple] = len(simplices_dim)
            simplices_dim.append(simplex_tuple)
            if self.is_kcritical:
                filtration_array = np.asarray(filtration, dtype=filtration_dtype)
                if filtration_array.ndim == 1:
                    filtration_array = filtration_array[None, :]
                filtration_array = filtration_array[
                    ~np.isinf(filtration_array).all(axis=1)
                ]
                filtrations_dim.append(tuple(filtration_array))
            else:
                filtrations_dim.append(np.asarray(filtration, dtype=filtration_dtype))
            if dim == 0:
                boundaries_dim.append(tuple())
            else:
                faces = [
                    tuple(np.asarray(face, dtype=np.int32).tolist())
                    for face, _ in self.get_boundaries(simplex)
                ]
                boundaries_dim.append(tuple(previous_index[face] for face in faces))
        previous_index = current_index
        if self.is_kcritical:
            blocks.append((tuple(filtrations_dim), tuple(boundaries_dim)))
        else:
            blocks.append(
                (
                    np.asarray(filtrations_dim, dtype=filtration_dtype),
                    tuple(boundaries_dim),
                )
            )
    return blocks[::-1]


def _to_scc(self, filtration_dtype=None, flattened=False):
    filtration_dtype = self.dtype if filtration_dtype is None else filtration_dtype
    try:
        blocks = _to_scc_blocks_raw[type(self)](self, flattened)
    except Exception:
        return _to_scc_python_fallback(
            self,
            filtration_dtype=filtration_dtype,
            flattened=flattened,
        )

    if flattened:
        filtrations, boundaries = blocks
        return np.asarray(filtrations, dtype=filtration_dtype), tuple(
            tuple(boundary) for boundary in boundaries
        )

    if self.is_kcritical:
        return [
            (
                tuple(np.asarray(f, dtype=filtration_dtype) for f in filtrations),
                tuple(tuple(boundary) for boundary in boundaries),
            )
            for filtrations, boundaries in blocks[::-1]
        ]

    return [
        (
            np.asarray(filtrations, dtype=filtration_dtype),
            tuple(tuple(boundary) for boundary in boundaries),
        )
        for filtrations, boundaries in blocks[::-1]
    ]


def _get_filtration_values(self, degrees=(-1,), inf_to_nan=False, return_raw=False):
    out = _get_filtration_values_raw[type(self)](self, list(degrees))
    filtrations_values = [np.asarray(filtration) for filtration in out]
    if (
        inf_to_nan
        and filtrations_values
        and np.dtype(filtrations_values[0].dtype).kind == "f"
    ):
        for filtration in filtrations_values:
            filtration[filtration == np.inf] = np.nan
            filtration[filtration == -np.inf] = np.nan
    return filtrations_values


def _clean_filtration_grid(self, api=None):
    if not self.is_squeezed:
        raise ValueError("No grid to clean.")
    filtration_grid = self.filtration_grid
    if api is None:
        api = api_from_tensor(filtration_grid[0])
    self.filtration_grid = None
    cleaned_coordinates = tuple(
        np.asarray(coords, dtype=np.int32) for coords in compute_grid(self)
    )
    new_st = self.grid_squeeze(cleaned_coordinates)
    self._copy_from_any(new_st)
    self.filtration_grid = tuple(
        api.ascontiguous(f[g]) for f, g in zip(filtration_grid, cleaned_coordinates)
    )
    return self


def _get_filtration_grid(
    self,
    resolution=None,
    degrees=None,
    drop_quantiles=0,
    grid_strategy="exact",
    threshold_min=None,
    threshold_max=None,
):
    degrees = (-1,) if degrees is None else degrees
    filtrations_values = np.concatenate(
        self._get_filtration_values(degrees, inf_to_nan=True), axis=1
    )
    filtrations_values = [np.unique(filtration) for filtration in filtrations_values]
    filtrations_values = [
        filtration[:-1] if len(filtration) and np.isnan(filtration[-1]) else filtration
        for filtration in filtrations_values
    ]
    return compute_grid(
        filtrations_values,
        resolution=resolution,
        strategy=grid_strategy,
        drop_quantiles=drop_quantiles,
        threshold_min=threshold_min,
        threshold_max=threshold_max,
    )


def _grid_squeeze(
    self,
    filtration_grid=None,
    coordinate_values=True,
    strategy="exact",
    resolution=None,
    coordinates=False,
    grid_strategy=None,
    inplace=False,
    threshold_min=None,
    threshold_max=None,
    **filtration_grid_kwargs,
):
    if grid_strategy is not None:
        warn(
            "`grid_strategy` is deprecated, use `strategy` instead.", DeprecationWarning
        )
        strategy = grid_strategy
    if self.is_squeezed:
        _mp_logs.warn_copy("Squeezing an already squeezed slicer.")
        temp = self.unsqueeze()
        subgrid = compute_grid(
            self.filtration_grid,
            strategy=strategy,
            resolution=resolution,
            threshold_min=threshold_min,
            threshold_max=threshold_max,
        )
        return temp.grid_squeeze(
            sanitize_grid(subgrid, numpyfy=True),
            coordinates=coordinates,
            inplace=inplace,
        )
    if filtration_grid is None:
        filtration_grid = self.get_filtration_grid(
            grid_strategy=strategy,
            resolution=resolution,
            threshold_min=threshold_min,
            threshold_max=threshold_max,
            **filtration_grid_kwargs,
        )
    else:
        filtration_grid = sanitize_grid(filtration_grid)
        if len(filtration_grid) != self.num_parameters:
            raise ValueError(
                f"Invalid grid to squeeze onto. Got {len(filtration_grid)=} != {self.num_parameters=}."
            )
    api = api_from_tensor(filtration_grid[0])
    c_filtration_grid = tuple(
        api.asnumpy(f).astype(np.float64) for f in filtration_grid
    )
    if coordinate_values and inplace:
        self.filtration_grid = filtration_grid
    if inplace or not coordinate_values:
        _squeeze_inplace_raw[type(self)](self, c_filtration_grid, coordinate_values)
        return self
    out = self.copy()
    _squeeze_inplace_raw[type(out)](out, c_filtration_grid, True)
    out.filtration_grid = filtration_grid
    return out


def _unsqueeze(self, grid=None):
    grid = self.filtration_grid if grid is None else grid
    cgrid = sanitize_grid(grid, numpyfy=True)
    new_st = SimplexTreeMulti(
        return_type_only=True,
        dtype=np.float64,
        kcritical=self.is_kcritical,
        ftype=self.filtration_container,
    )()
    new_st = _rebuild_from_current_simplices(
        self,
        type(new_st),
        lambda filtration: _values_from_coords(filtration, cgrid, dtype=np.float64),
    )
    return new_st


def _is_squeezed(self):
    return (
        self.num_vertices > 0
        and self.filtration_grid is not None
        and len(self.filtration_grid) > 0
        and len(self.filtration_grid[0]) > 0
    )


def _filtration_bounds(self, degrees=None, q=0, split_dimension=False):
    try:
        a, b = q
    except Exception:
        a, b = q, q
    degrees = range(self.dimension + 1) if degrees is None else degrees
    filtrations_values = self._get_filtration_values(degrees, inf_to_nan=True)
    boxes = np.array(
        [
            np.nanquantile(filtration, [a, 1 - b], axis=1)
            for filtration in filtrations_values
        ],
        dtype=float,
    )
    if split_dimension:
        return boxes
    return np.asarray([np.nanmin(boxes, axis=(0, 1)), np.nanmax(boxes, axis=(0, 1))])


def _fill_lowerstar(self, F, parameter):
    _fill_lowerstar_raw[type(self)](self, np.asarray(F, dtype=self.dtype), parameter)
    return self


def _fill_distance_matrix(self, distance_matrix, parameter, node_value=0):
    assert parameter < self.num_parameters
    c_distance_matrix = np.asarray(distance_matrix, dtype=self.dtype)
    F = np.zeros(shape=self.num_vertices, dtype=self.dtype) + node_value
    self.fill_lowerstar(F, parameter)
    for simplex, filtration in self.get_skeleton(1):
        if len(simplex) == 2:
            filtration = np.asarray(filtration, dtype=self.dtype)
            filtration[..., parameter] = c_distance_matrix[simplex[0], simplex[1]]
            self._assign_filtration(simplex, filtration)
    self.make_filtration_non_decreasing()
    return self


def _project_on_line(self, parameter=0, basepoint=None, direction=None):
    assert parameter < self.num_parameters
    if basepoint is None:
        basepoint = np.array([np.inf] * self.num_parameters)
        basepoint[parameter] = 0
    if direction is None:
        direction = np.array([0] * self.num_parameters)
        direction[parameter] = 1
    serialized = _to_std_state_raw[type(self)](
        self,
        np.asarray(basepoint, dtype=np.float64),
        np.asarray(direction, dtype=np.float64),
        int(parameter),
    )
    return _reconstruct_gudhi_simplextree(serialized)


def _linear_projections(self, linear_forms: np.ndarray):
    linear_forms = np.asarray(linear_forms, dtype=np.float64)
    if linear_forms.size == 0:
        return []
    assert linear_forms.shape[1] == self.num_parameters
    out = []
    for linear_form in linear_forms:
        serialized = _to_std_linear_projection_state_raw[type(self)](self, linear_form)
        out.append(_reconstruct_gudhi_simplextree(serialized))
    return out


def _eq(self, other):
    return _eq_raw[type(self)](self, other)


def _euler_characteristic(self, dtype=None):
    dtype = self.dtype if dtype is None else dtype
    out = {}
    for simplex, filtration in self.get_simplices():
        key = tuple(np.asarray(filtration).reshape(-1))
        dim = (len(simplex) - 1) % 2
        out[key] = out.get(key, 0) + (-1) ** dim
    new_pts = np.fromiter(
        out.keys(), dtype=np.dtype((dtype, self.num_parameters)), count=len(out)
    )
    new_weights = np.fromiter(out.values(), dtype=np.int32, count=len(out))
    idx = np.nonzero(new_weights)
    return new_pts[idx], new_weights[idx]


def _set_num_parameter(self, num):
    _set_num_parameter_raw[type(self)](self, int(num))
    return None


def _pts_to_indices(self, pts, simplices_dimensions):
    pts = np.asarray(pts, dtype=self.dtype)
    found_indices, not_found_indices = _pts_to_indices_raw[type(self)](
        self, pts, np.asarray(simplices_dimensions, dtype=np.int32)
    )
    found_indices = np.asarray(found_indices, dtype=np.int32)
    not_found_indices = np.asarray(not_found_indices, dtype=np.int32)
    if found_indices.size == 0:
        found_indices = np.empty((0, self.num_parameters), dtype=np.int32)
    if not_found_indices.size == 0:
        not_found_indices = np.empty((0, 2), dtype=np.int32)
    return found_indices, not_found_indices


def _get_edge_list(self):
    return _get_edge_list_raw[type(self)](self)


def _reconstruct_from_edge_list(self, edges, swap=True, expand_dimension=0):
    reduced_tree = type(self)()
    reduced_tree.set_num_parameter(self.num_parameters)
    reduced_tree.filtration_grid = self.filtration_grid
    reduced_tree._is_function_simplextree = self._is_function_simplextree
    if self.num_vertices > 0:
        vertices = np.fromiter(
            (splx[0] for splx, _ in self.get_skeleton(0)), dtype=np.int32
        )[None, :]
        vertices_filtration = np.asarray(
            [f for _, f in self.get_skeleton(0)], dtype=self.dtype
        )
        reduced_tree.insert_batch(vertices, vertices_filtration)
    if self.num_simplices - self.num_vertices > 0:
        edges_filtration = np.asarray(
            [(e[1][0], e[1][1]) for e in edges], dtype=self.dtype
        )
        edges_idx = np.asarray([(e[0][0], e[0][1]) for e in edges], dtype=np.int32).T
        reduced_tree.insert_batch(edges_idx, edges_filtration)
    if swap:
        filtration_grid = self.filtration_grid
        is_function = self._is_function_simplextree
        self._copy_from_any(reduced_tree)
        self.filtration_grid = filtration_grid
        self._is_function_simplextree = is_function
    if expand_dimension > 0:
        self.expansion(expand_dimension)
    return self if swap else reduced_tree


def _collapse_edges(
    self,
    num=1,
    max_dimension=0,
    progress=False,
    strong=True,
    full=False,
    ignore_warning=False,
    auto_clean=True,
):
    if num == 0:
        return self
    if num == -1:
        num = 100
        full = False
    elif num == -2:
        num = 100
        full = True
    assert self.num_parameters == 2
    if self.dimension > 1 and not ignore_warning:
        warn("This method ignores simplices of dimension > 1 !")
    max_dimension = self.dimension if max_dimension <= 0 else max_dimension
    from multipers.multiparameter_edge_collapse import _collapse_edge_list

    edges = _collapse_edge_list(
        self.get_edge_list(), num=num, full=full, strong=strong, progress=progress
    )
    self._reconstruct_from_edge_list(edges, swap=True, expand_dimension=max_dimension)
    if self.is_squeezed and auto_clean:
        self._clean_filtration_grid()
    return self


_num_vertices_raw = {}
_num_simplices_raw = {}
_dimension_raw = {}
_upper_bound_dimension_raw = {}
_simplex_dimension_raw = {}
_find_simplex_raw = {}
_remove_maximal_simplex_raw = {}
_prune_above_dimension_raw = {}
_expansion_raw = {}
_make_filtration_non_decreasing_raw = {}
_reset_filtration_raw = {}
_get_simplices_of_dimension_raw = {}
_get_key_raw = {}
_set_key_raw = {}
_set_keys_to_enumerate_raw = {}
_to_scc_blocks_raw = {}
_get_filtration_values_raw = {}
_squeeze_inplace_raw = {}
_squeeze_to_raw = {}
_unsqueeze_to_raw = {}
_fill_lowerstar_raw = {}
_to_std_state_raw = {}
_to_std_linear_projection_state_raw = {}
_eq_raw = {}
_set_num_parameter_raw = {}
_pts_to_indices_raw = {}
_get_edge_list_raw = {}


def _install_python_api():
    for cls in available_simplextrees:
        _num_vertices_raw[cls] = cls.num_vertices
        _num_simplices_raw[cls] = cls.num_simplices
        _dimension_raw[cls] = cls.dimension
        _upper_bound_dimension_raw[cls] = cls.upper_bound_dimension
        _simplex_dimension_raw[cls] = cls.simplex_dimension
        _find_simplex_raw[cls] = cls.find_simplex
        _remove_maximal_simplex_raw[cls] = cls.remove_maximal_simplex
        _prune_above_dimension_raw[cls] = cls.prune_above_dimension
        _expansion_raw[cls] = cls.expansion
        _make_filtration_non_decreasing_raw[cls] = cls.make_filtration_non_decreasing
        _reset_filtration_raw[cls] = cls.reset_filtration
        _get_simplices_of_dimension_raw[cls] = cls.get_simplices_of_dimension
        _get_key_raw[cls] = cls.get_key
        _set_key_raw[cls] = cls.set_key
        _set_keys_to_enumerate_raw[cls] = cls.set_keys_to_enumerate
        _to_scc_blocks_raw[cls] = cls._to_scc_blocks
        _get_filtration_values_raw[cls] = cls._get_filtration_values
        _squeeze_inplace_raw[cls] = cls._squeeze_inplace
        _squeeze_to_raw[cls] = cls._squeeze_to
        _unsqueeze_to_raw[cls] = cls._unsqueeze_to
        _fill_lowerstar_raw[cls] = cls.fill_lowerstar
        _to_std_state_raw[cls] = cls._get_to_std_state
        _to_std_linear_projection_state_raw[cls] = (
            cls._get_to_std_linear_projection_state
        )
        _eq_raw[cls] = cls.__eq__
        _set_num_parameter_raw[cls] = cls.set_num_parameter
        _pts_to_indices_raw[cls] = cls.pts_to_indices
        _get_edge_list_raw[cls] = cls.get_edge_list

        cls.__repr__ = _repr
        cls.__len__ = _len
        cls.__iter__ = _iter
        cls.__deepcopy__ = _deepcopy
        cls.__getstate__ = _getstate
        cls.__setstate__ = _setstate
        cls.__reduce__ = _reduce
        cls.copy = _copy
        cls.filtration = _filtration
        cls.__getitem__ = _getitem
        cls.astype = _astype
        cls.insert = _insert
        cls.assign_filtration = _assign_filtration
        cls.insert_batch = _insert_batch
        cls.get_simplices = _get_simplices
        cls.get_skeleton = _get_skeleton
        cls.get_boundaries = _get_boundaries
        cls.flagify = _flagify
        cls.__contains__ = _contains
        cls.remove_maximal_simplex = _remove_maximal_simplex
        cls.prune_above_dimension = _prune_above_dimension
        cls.expansion = _expansion
        cls.make_filtration_non_decreasing = _make_filtration_non_decreasing
        cls.reset_filtration = _reset_filtration
        cls.get_simplices_of_dimension = _get_simplices_of_dimension
        cls.key = _key
        cls.set_key = _set_key
        cls.set_keys_to_enumerate = _set_keys_to_enumerate
        cls._to_scc = _to_scc
        cls._get_filtration_values = _get_filtration_values
        cls._clean_filtration_grid = _clean_filtration_grid
        cls.get_filtration_grid = _get_filtration_grid
        cls.grid_squeeze = _grid_squeeze
        cls.unsqueeze = _unsqueeze
        cls.filtration_bounds = _filtration_bounds
        cls.fill_lowerstar = _fill_lowerstar
        cls.fill_distance_matrix = _fill_distance_matrix
        cls.project_on_line = _project_on_line
        cls.linear_projections = _linear_projections
        cls.__eq__ = _eq
        cls.euler_characteristic = _euler_characteristic
        cls.set_num_parameter = _set_num_parameter
        cls.pts_to_indices = _pts_to_indices
        cls.get_edge_list = _get_edge_list
        cls._reconstruct_from_edge_list = _reconstruct_from_edge_list
        cls.collapse_edges = _collapse_edges
        cls.num_vertices = property(_num_vertices_prop)
        cls.num_simplices = property(_num_simplices_prop)
        cls.dimension = property(_dimension_prop)
        cls.is_squeezed = property(_is_squeezed)


_install_python_api()

from multipers._simplextree_algorithms import (  # noqa: E402
    _euler_signed_measure,
    _hilbert_signed_measure,
    _rank_signed_measure,
)

__all__ = [
    *(cls.__name__ for cls in available_simplextrees),
    "SAFE_CONVERSION",
    "available_simplextrees",
    "available_dtype",
    "SimplexTreeMulti_type",
    "SimplexTreeMulti",
    "is_simplextree_multi",
    "_available_strategies",
    "_euler_signed_measure",
    "_hilbert_signed_measure",
    "_rank_signed_measure",
]
