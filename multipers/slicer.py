from __future__ import annotations

import os
from functools import reduce
from operator import or_
from typing import Any, Optional
from warnings import warn

import numpy as np

import multipers
import multipers.logs as _mp_logs
from multipers import _slicer_nanobind as _nb
from multipers.array_api import api_from_tensor, api_from_tensors
from multipers.grids import (
    compute_grid,
    evaluate_in_grid,
    sanitize_grid,
    _push_pts_to_lines,
)


available_slicers = tuple(_nb.available_slicers)
for _cls in available_slicers:
    globals()[_cls.__name__] = _cls

available_columns = {cls().col_type for cls in available_slicers}
default_column_type = (
    next(iter(available_columns)) if available_columns else "UNORDERED_SET"
)
available_dtype = {cls().dtype for cls in available_slicers}
available_pers_backend = {cls().pers_backend for cls in available_slicers}
available_filtration_container = {
    cls().filtration_container for cls in available_slicers
}

Slicer_type = reduce(or_, available_slicers) if available_slicers else Any
_valid_dtype = Any
_valid_pers_backend = str
_column_type = str
_filtration_container_type = str


def _has_filtration_grid(self) -> bool:
    return (
        self.filtration_grid is not None
        and len(self.filtration_grid) > 0
        and len(self.filtration_grid[0]) > 0
    )


def _is_minpres(self) -> bool:
    return self.minpres_degree >= 0


def _dimension(self):
    dims = self.get_dimensions()
    return dims[-1] if len(dims) else -np.inf


def _info(self):
    print(self._info_string())


def _repr(self):
    return (
        f"slicer[backend={self.pers_backend},dtype={np.dtype(self.dtype).name},"
        f"num_param={self.num_parameters},vineyard={self.is_vine},kcritical={self.is_kcritical},"
        f"is_squeezed={self.is_squeezed},is_minpres={self.is_minpres},max_dim={self.dimension}]"
    )


def _astype(
    self,
    vineyard=None,
    kcritical=None,
    dtype=None,
    col=None,
    pers_backend=None,
    filtration_container=None,
):
    vineyard = self.is_vine if vineyard is None else vineyard
    kcritical = self.is_kcritical if kcritical is None else kcritical
    dtype = self.dtype if dtype is None else dtype
    col = self.col_type if col is None else col
    pers_backend = self.pers_backend if pers_backend is None else pers_backend
    filtration_container = (
        self.filtration_container
        if filtration_container is None
        else filtration_container
    )
    new_slicer = get_matrix_slicer(
        vineyard, kcritical, dtype, col, pers_backend, filtration_container
    )
    if new_slicer is type(self):
        return self
    return new_slicer(self)


def _get_filtrations(
    self, unsqueeze=False, raw=False, view=False, packed=False, copy=None
):
    if copy is not None:
        if view:
            raise ValueError(
                "Got both copy and view arguments. Please provide only one."
            )
        view = not bool(copy)
    if unsqueeze:
        view = False
    if packed and unsqueeze:
        raise NotImplementedError("packed=True is incompatible with unsqueeze=True.")
    if packed and view:
        raise ValueError("packed=True does not support view=True.")
    if packed and not self.is_kcritical:
        raise ValueError(
            "packed=True is only available for k-critical filtrations. Use view=False for one-critical detached output."
        )
    if unsqueeze and not self.is_squeezed:
        raise ValueError(f"Already unsqueezed. Got {unsqueeze=}")

    out = self._get_filtrations_impl(raw=raw, view=view, packed=packed)
    if not unsqueeze:
        return out

    grid = self.filtration_grid
    grid_size = np.array([len(g) for g in grid], dtype=np.int32)
    if self.is_kcritical:
        return [
            evaluate_in_grid(
                np.asarray(current, dtype=np.int32).clip(None, grid_size - 1), grid
            )
            for current in out
        ]
    return evaluate_in_grid(
        np.asarray(out, dtype=np.int32).clip(None, grid_size - 1), grid
    )


def _make_filtration_non_decreasing(self, safe=True):
    if self.is_kcritical:
        raise NotImplementedError(
            "make_filtration_non_decreasing is only available for one-critical slicers."
        )
    if self.is_squeezed and safe:
        for f in self.filtration_grid:
            if len(f) > 1 and not (np.asarray(f[1:]) >= np.asarray(f[:-1])).all():
                raise ValueError("Found non-sorted grid.")
    filtrations = self.get_filtrations(view=True)
    boundaries = self.get_boundaries()
    for i, boundary in enumerate(boundaries):
        for b in boundary:
            filtrations[b][:] = np.maximum(filtrations[b], filtrations[i])
    return self


def _compute_persistence(
    self, one_filtration=None, ignore_infinite_filtration_values=True, verbose=False
):
    if one_filtration is not None:
        if verbose:
            print(
                f"Computing persistence on custom filtration: shape={np.shape(one_filtration)}"
            )
        api = api_from_tensor(one_filtration)
        one_filtration = api.astensor(one_filtration)
        if one_filtration.ndim == 0 or one_filtration.ndim > 2:
            raise ValueError(
                f"Expected a filtration shape of the form ((num_1_param), num_generators). Got {one_filtration.shape=}"
            )
        squeeze = False
        if one_filtration.ndim == 1:
            one_filtration = one_filtration[None]
            squeeze = True
        one_filtration = api.asnumpy(one_filtration)
        out = []
        for row in one_filtration:
            self.set_slice(np.asarray(row, dtype=self.dtype))
            self.initialize_persistence_computation(ignore_infinite_filtration_values)
            out.append(tuple(np.asarray(bc) for bc in self.get_barcode()))
        return out[0] if squeeze else tuple(out)
    self.initialize_persistence_computation(ignore_infinite_filtration_values)
    return self.get_barcode()


def _sliced_filtration(self, basepoint, direction=None):
    self.push_to_line(basepoint, direction)
    return np.asarray(self.get_current_filtration())


def _filtration_bounds(self):
    values = np.asarray(self.get_filtrations_values(), dtype=self.dtype)
    if values.size == 0:
        return np.empty((2, 0), dtype=self.dtype)
    return np.asarray([values.min(axis=0), values.max(axis=0)], dtype=self.dtype)


def _get_filtration_grid(self, grid_strategy="exact", **infer_grid_kwargs):
    return compute_grid(
        self.get_filtrations_values().T, strategy=grid_strategy, **infer_grid_kwargs
    )


def _persistence_on_line(
    self,
    basepoint,
    direction=None,
    keep_inf=True,
    full=False,
    ignore_infinite_filtration_values=True,
):
    if np.issubdtype(np.dtype(self.dtype), np.floating):
        api = api_from_tensors(basepoint)
        basepoint = api.asnumpy(basepoint)
        direction = None if direction is None else api.asnumpy(direction)
        if api.has_grad(basepoint) or (
            direction is not None and api.has_grad(direction)
        ):
            _mp_logs.warn_autodiff(
                "Ignored gradient from input. Use a squeezed slicer for autodiff."
            )
        self.push_to_line(basepoint, direction)
        self.initialize_persistence_computation(ignore_infinite_filtration_values)
        bcs = tuple(np.asarray(stuff, dtype=self.dtype) for stuff in self.get_barcode())
        if not keep_inf:
            inf_value = type(self)._inf_value()
            bcs = tuple(
                np.asarray(
                    [a for a in stuff if a[0] < inf_value],
                    dtype=np.dtype((self.dtype, 2)),
                )
                for stuff in bcs
            )
        if full:
            bcs = self._bc_to_full(bcs, basepoint, direction)
        return bcs

    if not self.is_squeezed:
        raise ValueError(
            "Unsqueeze tensor, or provide a filtration grid. Cannot slice lines with integers."
        )
    api = api_from_tensors(basepoint, *self.filtration_grid)
    basepoint = api.astensor(basepoint)
    fil = evaluate_in_grid(np.asarray(self.get_filtrations()), self.filtration_grid)
    if basepoint.ndim == 0 or basepoint.ndim > 2:
        raise ValueError(
            f"Expected a basepoint shape of the form (num_parameters,). Got {basepoint.shape=}"
        )
    if basepoint.ndim == 1:
        basepoint = basepoint[None]
    if direction is not None:
        direction = api.astensor(direction)
        if direction.ndim == 0 or direction.ndim > 2:
            raise ValueError(
                f"Expected a direction shape of the form (num_parameters,). Got {direction.shape=}"
            )
        if direction.ndim == 1:
            direction = direction[None]
    projected_fil = _push_pts_to_lines(fil, basepoint, direction, api=api)
    bcs = self.compute_persistence(
        projected_fil,
        ignore_infinite_filtration_values=ignore_infinite_filtration_values,
    )
    if full:
        dirs = [None] * len(basepoint) if direction is None else direction
        bcs = tuple(
            self._bc_to_full(x, bp, dir_) for x, bp, dir_ in zip(bcs, basepoint, dirs)
        )
    return bcs


def _persistence_on_lines(
    self,
    basepoints,
    directions=None,
    keep_inf=True,
    full=False,
    ignore_infinite_filtration_values=True,
):
    api = api_from_tensors(basepoints)
    basepoints = api.asnumpy(basepoints)
    directions = None if directions is None else api.asnumpy(directions)
    if api.has_grad(basepoints) or (
        directions is not None and api.has_grad(directions)
    ):
        _mp_logs.warn_autodiff(
            "Ignored gradient from input. Use `persistence_on_line` for autodiff (supports multilines)."
        )

    if basepoints.ndim == 1:
        basepoints = basepoints[None]
    if directions is not None and directions.ndim == 1:
        directions = directions[None]

    out = []
    for i, bp in enumerate(basepoints):
        direction = None if directions is None else directions[i]
        if i == 0 or not self.is_vine:
            out.append(
                self.persistence_on_line(
                    bp,
                    direction,
                    keep_inf=keep_inf,
                    full=full,
                    ignore_infinite_filtration_values=ignore_infinite_filtration_values,
                )
            )
        else:
            self.push_to_line(bp, direction)
            self.update_persistence_computation()
            bcs = tuple(
                np.asarray(stuff, dtype=self.dtype) for stuff in self.get_barcode()
            )
            if not keep_inf:
                inf_value = type(self)._inf_value()
                bcs = tuple(
                    np.asarray(
                        [a for a in stuff if a[0] < inf_value],
                        dtype=np.dtype((self.dtype, 2)),
                    )
                    for stuff in bcs
                )
            if full:
                bcs = self._bc_to_full(bcs, bp, direction)
            out.append(bcs)
    return tuple(out)


def _getstate(self):
    return (
        self.get_boundaries(),
        self.get_dimensions(),
        self.get_filtrations(),
        self.filtration_grid,
        self.minpres_degree,
    )


def _setstate(self, dump):
    boundaries, dimensions, filtrations, filtration_grid, minpres_degree = dump
    copy = type(self)(boundaries, dimensions, filtrations)
    self._from_ptr(copy.get_ptr())
    self.minpres_degree = minpres_degree
    self.filtration_grid = filtration_grid


def _eq(self, other):
    if other.is_squeezed:
        return self == other.unsqueeze()
    if self.is_squeezed:
        return self.unsqueeze() == other
    self_boundaries = self.get_boundaries(packed=True)
    other_boundaries = other.get_boundaries(packed=True)
    return (
        np.array_equal(self.get_dimensions(), other.get_dimensions())
        and np.array_equal(self_boundaries[0], other_boundaries[0])
        and np.array_equal(self_boundaries[1], other_boundaries[1])
        and np.array_equal(
            self.get_filtrations_values(), other.get_filtrations_values()
        )
    )


def _bc_to_full(bcs, basepoint, direction=None):
    basepoint = np.asarray(basepoint)[None, None, :]
    direction = 1 if direction is None else np.asarray(direction)[None, None, :]
    return tuple(bc[:, :, None] * direction + basepoint for bc in bcs)


def _grid_squeeze(
    self,
    filtration_grid=None,
    strategy="exact",
    resolution=None,
    coordinates=True,
    inplace=False,
    grid_strategy=None,
    threshold_min=None,
    threshold_max=None,
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
        return temp.grid_squeeze(subgrid, coordinates=coordinates, inplace=inplace)

    if filtration_grid is None:
        filtration_grid = compute_grid(
            self.get_filtrations_values().T,
            strategy=strategy,
            resolution=resolution,
            threshold_min=threshold_min,
            threshold_max=threshold_max,
        )
    api = api_from_tensor(filtration_grid[0]) if len(filtration_grid) else None
    if api is None or api.__name__.endswith("numpy"):
        grid = tuple(np.asarray(g, dtype=self.dtype) for g in filtration_grid)
        c_grid = grid
    else:
        grid = tuple(api.ascontiguous(g) for g in filtration_grid)
        c_grid = tuple(np.asarray(api.asnumpy(g), dtype=self.dtype) for g in grid)
    if inplace or not coordinates:
        self.coarsen_on_grid_inplace([g.tolist() for g in c_grid], coordinates)
        if coordinates:
            self.filtration_grid = sanitize_grid(grid)
        return self
    out = self.coarsen_on_grid_copy([g.tolist() for g in c_grid])
    if coordinates:
        out.filtration_grid = sanitize_grid(grid)
    out.minpres_degree = self.minpres_degree
    return out


def _clean_filtration_grid(self):
    if not self.is_squeezed:
        raise ValueError("No grid to clean.")
    filtration_grid = self.filtration_grid
    self.filtration_grid = None
    cleaned_coordinates = tuple(
        np.asarray(coords, dtype=np.int32) for coords in compute_grid(self)
    )
    new_slicer = self.copy()
    new_slicer.grid_squeeze(cleaned_coordinates, inplace=True)
    self._from_ptr(new_slicer.get_ptr())
    self.filtration_grid = tuple(
        f[g] for f, g in zip(filtration_grid, cleaned_coordinates)
    )
    return self


def _minpres(
    self,
    degree=-1,
    degrees=None,
    backend="mpfree",
    vineyard=None,
    dtype=None,
    force=True,
    auto_clean=True,
    full_resolution=True,
):
    if degrees is None:
        degrees = []
    from multipers.ops import minimal_presentation

    return minimal_presentation(
        self,
        degree=degree,
        degrees=degrees,
        backend=backend,
        force=force,
        auto_clean=auto_clean,
        full_resolution=full_resolution,
    )


def _to_scc(
    self,
    path: os.PathLike,
    degree=-1,
    rivet_compatible=False,
    ignore_last_generators=False,
    strip_comments=False,
    reverse=False,
    unsqueeze=True,
):
    if degree == -1 and not rivet_compatible:
        degree = 1
    if self.is_squeezed and unsqueeze:
        kwargs = dict(
            path=path,
            degree=degree,
            rivet_compatible=rivet_compatible,
            ignore_last_generators=ignore_last_generators,
            strip_comments=strip_comments,
            reverse=reverse,
            unsqueeze=False,
        )
        self.unsqueeze().to_scc(**kwargs)
        return
    self._to_scc_raw(
        os.fspath(path),
        degree,
        rivet_compatible,
        ignore_last_generators,
        strip_comments,
        reverse,
    )


def _unsqueeze(self, grid=None, inf_overflow=True):
    if self.filtration_container == "Flat":
        raise NotImplementedError("There is no reasonable implementation (yet).")
    grid = self.filtration_grid if grid is None else grid
    grid = sanitize_grid(grid, numpyfy=True, add_inf=inf_overflow)
    grid_size = np.array([len(g) for g in grid], dtype=np.int32)

    if self.is_kcritical:
        current_filtration = self.get_filtrations()
        new_filtrations = [
            evaluate_in_grid(
                np.asarray(current_filtration[i], dtype=np.int32).clip(
                    None, grid_size - 1
                ),
                grid,
            )
            for i in range(len(self))
        ]
    else:
        filtrations = np.asarray(self.get_filtrations(), dtype=np.int32).clip(
            None, grid_size - 1
        )
        new_filtrations = evaluate_in_grid(filtrations, grid)

    real_dtype = np.asarray(grid[0]).dtype.type if len(grid) else self.dtype
    if (
        np.issubdtype(np.dtype(real_dtype), np.floating)
        and np.dtype(real_dtype) not in {np.dtype(dtype) for dtype in available_dtype}
        and np.dtype(np.float64) in {np.dtype(dtype) for dtype in available_dtype}
    ):
        real_dtype = np.float64
    new_slicer = get_matrix_slicer(
        self.is_vine,
        self.is_kcritical,
        real_dtype,
        self.col_type,
        self.pers_backend,
        self.filtration_container,
    )(
        self.get_boundaries(),
        self.get_dimensions(),
        new_filtrations,
    )
    new_slicer.minpres_degree = self.minpres_degree
    return new_slicer


def slicer2blocks(slicer, degree=-1, reverse=True):
    dims = np.asarray(slicer.get_dimensions(), dtype=np.int32)
    num_empty_blocks_to_add = 1 if degree == -1 else dims.min() - degree + 1
    _, counts = np.unique(dims, return_counts=True)
    indices = np.concatenate([[0], counts], dtype=np.int32).cumsum()
    filtration_values = slicer.get_filtrations()
    filtration_values = [
        filtration_values[indices[i] : indices[i + 1]] for i in range(len(indices) - 1)
    ]
    boundaries = slicer.get_boundaries()
    boundaries = [
        boundaries[indices[i] : indices[i + 1]] for i in range(len(indices) - 1)
    ]
    shift = np.concatenate([[0], indices], dtype=np.int32)
    boundaries = [
        tuple(np.asarray(x - s, dtype=np.int32) for x in block)
        for s, block in zip(shift, boundaries)
    ]
    blocks = [tuple((f, tuple(b))) for b, f in zip(boundaries, filtration_values)]
    blocks = ([(np.empty((0,)), [])] * num_empty_blocks_to_add) + blocks
    if reverse:
        blocks.reverse()
    return blocks


def to_simplextree(s: Slicer_type, max_dim: int = -1):
    from multipers.simplex_tree_multi import SimplexTreeMulti

    dims = np.asarray(s.get_dimensions(), dtype=np.int32)
    assert np.all(dims[:-1] <= dims[1:]), "Dims is not sorted."
    idx = np.searchsorted(dims, np.unique(dims)).astype(np.int32)
    idx = np.concatenate([idx, np.asarray([dims.shape[0]], dtype=np.int32)])
    if max_dim >= 0:
        idx = idx[: max_dim + 2]

    boundaries_indptr, boundaries_indices = s.get_boundaries(packed=True)
    boundaries_indptr = np.asarray(boundaries_indptr, dtype=np.int64)
    boundaries_indices = np.asarray(boundaries_indices, dtype=np.int32)
    simplex_vertices_sets = [set() for _ in range(len(dims))]

    if len(idx) > 1:
        a = idx[1]
        b = idx[-1]
        c = idx[2] if len(idx) > 2 else b
        for i in range(a, b):
            row = boundaries_indices[boundaries_indptr[i] : boundaries_indptr[i + 1]]
            if i < c:
                simplex_vertices_sets[i].update(int(x) for x in row)
            else:
                for face_idx in row:
                    simplex_vertices_sets[i].update(
                        simplex_vertices_sets[int(face_idx)]
                    )

    num_dims = len(idx) - 1
    boundaries = [None] * num_dims
    if num_dims > 0:
        boundaries[0] = np.arange(idx[1] - idx[0], dtype=np.int32)[None, :]
    for dim in range(1, num_dims):
        start_dim = idx[dim]
        end_dim = idx[dim + 1]
        boundaries_dim = np.empty((dim + 1, end_dim - start_dim), dtype=np.int32)
        for local_idx in range(end_dim - start_dim):
            simplex_idx = start_dim + local_idx
            vertices = sorted(simplex_vertices_sets[simplex_idx])
            if len(vertices) != dim + 1:
                raise ValueError("Input slicer is not simplicial.")
            boundaries_dim[:, local_idx] = vertices
        boundaries[dim] = boundaries_dim

    filtrations = s.get_filtrations()
    filtrations = [filtrations[idx[i] : idx[i + 1]] for i in range(num_dims)]
    st = SimplexTreeMulti(
        num_parameters=s.num_parameters, dtype=s.dtype, kcritical=s.is_kcritical
    )
    for dim in range(num_dims):
        if boundaries[dim].shape[1] == 0:
            continue
        if s.is_kcritical:
            max_crit_deg = max(
                len(filtrations[dim][j]) for j in range(boundaries[dim].shape[1])
            )
            filtrations_dim = np.empty(
                (boundaries[dim].shape[1], max_crit_deg, s.num_parameters),
                dtype=s.dtype,
            )
            for j in range(boundaries[dim].shape[1]):
                current_f = np.asarray(filtrations[dim][j], dtype=s.dtype).reshape(
                    -1, s.num_parameters
                )
                if current_f.shape[0] == 0:
                    raise ValueError(
                        "Input slicer contains an empty k-critical filtration."
                    )
                filtrations_dim[j, : current_f.shape[0], :] = current_f
                if current_f.shape[0] < max_crit_deg:
                    filtrations_dim[j, current_f.shape[0] :, :] = current_f[
                        current_f.shape[0] - 1
                    ]
            st.insert_batch(
                np.asarray(boundaries[dim], dtype=np.int32), filtrations_dim
            )
        else:
            st.insert_batch(
                np.asarray(boundaries[dim], dtype=np.int32),
                np.asarray(filtrations[dim], dtype=s.dtype),
            )
    return st


def _is_slicer(input) -> bool:
    return any(isinstance(input, cls) for cls in available_slicers)


def is_slicer(input, allow_minpres=True) -> bool:
    if _is_slicer(input):
        return True
    if allow_minpres and (isinstance(input, list) or isinstance(input, tuple)):
        return len(input) > 0 and all(_is_slicer(s) and s.is_minpres for s in input)
    return False


def to_blocks(input):
    if is_slicer(input):
        return slicer2blocks(input)
    if isinstance(input, list) or isinstance(input, tuple):
        return input
    from multipers.simplex_tree_multi import is_simplextree_multi

    if is_simplextree_multi(input):
        return input._to_scc()
    if isinstance(input, str) or isinstance(input, os.PathLike):
        from multipers.io import scc_parser

        return scc_parser(input)
    raise ValueError("Input cannot be converted to blocks.")


def _signed_measure_from_slicer(slicer: Slicer_type, shift: int = 0):
    if slicer.is_kcritical:
        raise NotImplementedError("Not implemented for k-critical filtrations yet.")
    dims = np.asarray(slicer.get_dimensions(), dtype=np.int32)
    weights = 1 - 2 * (((dims & 1) ^ (shift & 1)) & 1)
    return [
        (
            np.asarray(slicer.get_filtrations(view=False)),
            weights.astype(np.int32, copy=False),
        )
    ]


def _signed_measure_from_scc(minimal_presentation):
    if len(minimal_presentation) == 0:
        return [(np.empty((0, 0)), np.empty((0,), dtype=np.float64))]
    pts_per_block = [block[0] for block in minimal_presentation]
    block_sizes = [len(block_pts) for block_pts in pts_per_block]
    pts = np.concatenate(pts_per_block)
    weights = np.empty(sum(block_sizes), dtype=np.float64)
    offset = 0
    for i, block_size in enumerate(block_sizes):
        weights[offset : offset + block_size] = 1.0 if (i & 1) == 0 else -1.0
        offset += block_size
    return [(pts, weights)]


def get_matrix_slicer(
    is_vineyard, is_k_critical, dtype, col, pers_backend, filtration_container
):
    dtype = np.dtype(dtype).type
    for cls in available_slicers:
        sample = cls()
        if (
            sample.is_vine == is_vineyard
            and sample.is_kcritical == is_k_critical
            and np.dtype(sample.dtype) == np.dtype(dtype)
            and sample.col_type.lower() == str(col).lower()
            and sample.pers_backend.lower() == str(pers_backend).lower()
            and sample.filtration_container.lower() == str(filtration_container).lower()
        ):
            return cls
    raise ValueError(
        f"Unimplemented combo for {pers_backend} : {is_vineyard=}, {is_k_critical=}, {dtype=}, {col=}, {filtration_container=}"
    )


from multipers._slicer_algorithms import (  # noqa: E402
    _hilbert_signed_measure,
    _rank_from_slicer,
    from_bitmap,
)


def from_function_delaunay(
    points,
    grades,
    degree=-1,
    backend: Optional[_valid_pers_backend] = None,
    vineyard=None,
    dtype=np.float64,
    verbose=False,
    clear=True,
):
    from multipers.io import (
        function_delaunay_presentation_to_simplextree,
        function_delaunay_presentation_to_slicer,
    )

    if degree < 0:
        return function_delaunay_presentation_to_simplextree(
            points,
            grades,
            verbose=verbose,
            clear=clear,
            dtype=dtype,
        )
    s = multipers.Slicer(None, backend=backend, vineyard=vineyard, dtype=dtype)
    function_delaunay_presentation_to_slicer(
        s, points, grades, degree=degree, verbose=verbose, clear=clear
    )
    if degree >= 0:
        s.minpres_degree = degree
    return s


def _install_python_api():
    for cls in available_slicers:
        cls.__repr__ = _repr
        cls.__getstate__ = _getstate
        cls.__setstate__ = _setstate
        cls.__eq__ = _eq
        cls.astype = _astype
        cls.get_filtrations = _get_filtrations
        cls.compute_persistence = _compute_persistence
        cls.persistence_on_line = _persistence_on_line
        cls.persistence_on_lines = _persistence_on_lines
        cls.sliced_filtration = _sliced_filtration
        cls.filtration_bounds = _filtration_bounds
        cls.get_filtration_grid = _get_filtration_grid
        cls.grid_squeeze = _grid_squeeze
        cls._clean_filtration_grid = _clean_filtration_grid
        cls.minpres = _minpres
        cls.to_scc = _to_scc
        cls.unsqueeze = _unsqueeze
        cls._bc_to_full = staticmethod(_bc_to_full)
        cls.is_squeezed = property(_has_filtration_grid)
        cls.is_minpres = property(_is_minpres)
        cls.dimension = property(_dimension)
        cls.info = property(_info)
        if not cls().is_kcritical:
            cls.make_filtration_non_decreasing = _make_filtration_non_decreasing


_install_python_api()

__all__ = [
    *(cls.__name__ for cls in available_slicers),
    "Slicer_type",
    "available_slicers",
    "available_columns",
    "default_column_type",
    "available_dtype",
    "available_pers_backend",
    "available_filtration_container",
    "from_bitmap",
    "from_function_delaunay",
    "slicer2blocks",
    "to_simplextree",
    "_is_slicer",
    "is_slicer",
    "to_blocks",
    "_signed_measure_from_slicer",
    "_signed_measure_from_scc",
    "get_matrix_slicer",
    "_hilbert_signed_measure",
    "_rank_from_slicer",
]
