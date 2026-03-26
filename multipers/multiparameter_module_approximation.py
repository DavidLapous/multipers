from __future__ import annotations

from functools import reduce
from operator import or_
from typing import Any, Callable, Iterable, Optional, Sequence, Union

import numpy as np
from joblib import Parallel, delayed

import multipers.grids as mpg
import multipers.logs as _mp_logs
from . import _mma_nanobind as _mma
from . import _slicer_nanobind as _nb
from .simplex_tree_multi import SimplexTreeMulti_type, is_simplextree_multi
from .slicer import Slicer_type, is_slicer


available_pymodules = tuple(
    cls for name, cls in sorted(vars(_mma).items()) if name.startswith("PyModule_")
)
available_pysummands = tuple(
    cls for name, cls in sorted(vars(_mma).items()) if name.startswith("PySummand_")
)
available_pyboxes = tuple(
    cls for name, cls in sorted(vars(_mma).items()) if name.startswith("PyBox_")
)

for _cls in (*available_pymodules, *available_pysummands, *available_pyboxes):
    globals()[_cls.__name__] = _cls

for _name, _value in vars(_mma).items():
    if _name.startswith("from_dump_"):
        globals()[_name] = _value

PyModule_type = reduce(or_, available_pymodules) if available_pymodules else Any
PySummand_type = reduce(or_, available_pysummands) if available_pysummands else Any
PyBox_type = reduce(or_, available_pyboxes) if available_pyboxes else Any

_MMA_EMPTY = {np.dtype(cls().dtype): cls for cls in available_pymodules}

AVAILABLE_MMA_FLOAT_DTYPES = tuple(
    dtype for dtype in _MMA_EMPTY if np.issubdtype(dtype, np.floating)
)


def _reconstruct_module(loader_name: str, state):
    return getattr(_mma, loader_name)(state)


def _module_bc_to_full(bcs, basepoint, direction=None):
    basepoint = np.asarray(basepoint)[None, None, :]
    direction = 1 if direction is None else np.asarray(direction)[None, None, :]
    return tuple(bc[:, :, None] * direction + basepoint for bc in bcs)


def _module_threshold_bc(bc):
    return tuple(
        np.fromiter(
            (a for a in stuff if a[0] < np.inf), dtype=np.dtype((bc[0].dtype, 2))
        )
        for stuff in bc
    )


def _module_representation(
    self,
    degrees=None,
    bandwidth: float = 0.1,
    resolution: Sequence[int] | int = 50,
    kernel: str | Callable = "gaussian",
    signed: bool = False,
    normalize: bool = False,
    plot: bool = False,
    save: bool = False,
    dpi: int = 200,
    p: float = 2.0,
    box=None,
    flatten: bool = False,
    n_jobs: int = 0,
    grid=None,
):
    import matplotlib.pyplot as plt
    import multipers.plots

    if box is None:
        box = self.get_box()
    num_parameters = self.num_parameters
    if degrees is None:
        degrees = np.arange(self.max_degree + 1)
    if np.isscalar(resolution):
        resolution = [int(resolution)] * num_parameters
    else:
        resolution = list(resolution)

    if grid is None:
        grid = [
            np.linspace(*np.asarray(box)[:, parameter], num=res)
            for parameter, res in zip(range(num_parameters), resolution)
        ]
    else:
        resolution = tuple(len(g) for g in grid)
    coordinates = mpg.todense(grid)

    if kernel == "linear":
        concatenated_images = np.asarray(
            self._compute_pixels(
                coordinates, degrees, box, bandwidth, p, normalize, n_jobs
            )
        )
    else:
        if kernel == "linear2":
            assert not signed, "This kernel is not compatible with signed."

            def todo(mod_degree):
                x = mod_degree.distance_to(coordinates, signed=signed, n_jobs=n_jobs)
                w = mod_degree.get_interleavings(box)[None] ** p
                x = np.abs(x)
                return (
                    np.where(x < bandwidth, (bandwidth - x) / bandwidth, 0) * w
                ).sum(1)

        elif kernel == "gaussian":

            def todo(mod_degree):
                x = mod_degree.distance_to(coordinates, signed=signed, n_jobs=n_jobs)
                w = mod_degree.get_interleavings(box)[None] ** p
                s = np.where(x >= 0, 1, -1) if signed else 1
                return (s * np.exp(-0.5 * ((x / bandwidth) ** 2)) * w).sum(1)

        elif kernel == "exponential":

            def todo(mod_degree):
                x = mod_degree.distance_to(coordinates, signed=signed, n_jobs=n_jobs)
                w = mod_degree.get_interleavings(box)[None] ** p
                s = np.where(x >= 0, 1, -1) if signed else 1
                return (s * np.exp(-(np.abs(x) / bandwidth)) * w).sum(1)

        else:
            assert callable(kernel), (
                "Kernel should be gaussian, linear, linear2, exponential or callable with signature "
                "(distance_matrix, summand_weights) -> representation."
            )

            def todo(mod_degree):
                x = mod_degree.distance_to(coordinates, signed=signed, n_jobs=n_jobs)
                w = mod_degree.get_interleavings(box)[None] ** p
                return kernel(x / bandwidth, w)

        concatenated_images = np.stack(
            Parallel(n_jobs=n_jobs if n_jobs else -1, backend="threading")(
                delayed(todo)(self.get_module_of_degree(degree)) for degree in degrees
            )
        )

    if normalize and concatenated_images.size:
        max_abs = np.max(np.abs(concatenated_images))
        if max_abs > 0:
            concatenated_images = concatenated_images / max_abs

    if flatten:
        image_vector = concatenated_images.reshape((len(degrees), -1))
        if plot:
            raise ValueError("Unflatten to plot.")
        return image_vector

    image_vector = concatenated_images.reshape((len(degrees), *resolution))
    if plot:
        assert num_parameters == 2, "Plot only available for 2-parameter modules"
        n_plots = len(image_vector)
        scale = 4
        if n_plots > 1:
            fig, axs = plt.subplots(1, n_plots, figsize=(n_plots * scale, scale))
        else:
            fig = plt.gcf()
            axs = [plt.gca()]
        for image, degree, idx in zip(image_vector, degrees, range(len(degrees))):
            ax = axs[idx]
            temp = multipers.plots.plot_surface(grid, image, ax=ax)
            plt.colorbar(temp, ax=ax)
            ax.set_title(rf"$H_{degree}$ $2$-persistence image")
        if save:
            plt.savefig(save, dpi=dpi)
    return image_vector


def _module_landscapes(
    self,
    degree: int,
    ks: list | np.ndarray = [0],
    box=None,
    resolution: list | np.ndarray = [100, 100],
    grid=None,
    n_jobs: int = 0,
    plot: bool = False,
):
    import matplotlib.pyplot as plt

    ks = np.asarray(ks, dtype=np.int32).reshape(-1)
    if grid is not None:
        out = np.asarray(
            self._compute_landscapes_grid(
                int(degree),
                ks.tolist(),
                [np.asarray(g, dtype=self.dtype).tolist() for g in grid],
                int(n_jobs),
            )
        )
        return out

    if box is None:
        box = self.get_box()
    try:
        int(resolution)
        resolution = [int(resolution)] * self.num_parameters
    except Exception:
        resolution = list(resolution)

    out = np.asarray(
        self._compute_landscapes_box(
            int(degree),
            ks.tolist(),
            np.asarray(box, dtype=self.dtype).tolist(),
            [int(r) for r in resolution],
            int(n_jobs),
        )
    )
    if plot:
        to_plot = np.sum(out, axis=0)
        aspect = (box[1][0] - box[0][0]) / (box[1][1] - box[0][1])
        extent = [box[0][0], box[1][0], box[0][1], box[1][1]]
        plt.imshow(to_plot.T, origin="lower", extent=extent, aspect=aspect)
    return out


def _module_barcode2(
    self,
    basepoint,
    direction=None,
    degree: int = -1,
    *,
    threshold: bool = False,
    keep_inf: bool = True,
    full: bool = False,
):
    basepoint = np.ascontiguousarray(basepoint, dtype=self.dtype)
    if direction is not None:
        direction = np.ascontiguousarray(direction, dtype=self.dtype)
    bc = tuple(
        np.asarray(x).reshape(-1, 2)
        for x in self._get_barcode_from_line(basepoint, direction, int(degree))
    )
    if not keep_inf:
        bc = type(self)._threshold_bc(bc)
    if full:
        bc = type(self)._bc_to_full(bc, basepoint, direction)
    return bc


def _module_plot(self, degree: int = -1, **kwargs):
    from multipers.plots import plot2d_PyModule
    import matplotlib.pyplot as plt

    box = kwargs.pop("box", self.get_box())
    if len(box[0]) != 2:
        print("Filtration size :", len(box[0]), " != 2")
        return None

    if degree < 0:
        dims = np.unique(self.get_dimensions())
        separated = kwargs.pop("separated", False)
        ndim = len(dims)
        scale = kwargs.pop("scale", 4)
        if separated:
            fig = None
            axes = None
        elif ndim > 1:
            fig, axes = plt.subplots(1, ndim, figsize=(ndim * scale, scale))
        else:
            fig = plt.gcf()
            axes = [plt.gca()]
        for dim_idx, dim in enumerate(dims):
            if not separated:
                plt.sca(axes[dim_idx])
            self.plot(dim, box=box, separated=separated, **kwargs)
        return None

    mod = self.get_module_of_degree(degree)
    corners = []
    for summand in mod:
        corners.append(
            (np.asarray(summand.get_birth_list()), np.asarray(summand.get_death_list()))
        )
    interleavings = mod.get_interleavings(box)
    plot2d_PyModule(
        corners,
        box=box,
        dimension=degree,
        interleavings=interleavings,
        **kwargs,
    )
    return None


def is_mma(stuff):
    return any(isinstance(stuff, cls) for cls in available_pymodules)


def _install_python_api():
    for cls in available_pymodules:
        cls._bc_to_full = staticmethod(_module_bc_to_full)
        cls._threshold_bc = staticmethod(_module_threshold_bc)
        if np.issubdtype(np.dtype(cls().dtype), np.floating):
            cls.plot = _module_plot
            cls.landscapes = _module_landscapes
            cls.representation = _module_representation
            cls.barcode2 = _module_barcode2


_install_python_api()


def module_approximation_from_slicer(
    slicer: Slicer_type,
    box: Optional[np.ndarray] = None,
    max_error=-1,
    complete: bool = True,
    threshold: bool = False,
    verbose: bool = False,
    direction: list[float] | np.ndarray = [],
    warnings: bool = True,
    unsqueeze_grid=None,
    n_jobs: int = -1,
) -> PyModule_type:
    if not slicer.is_vine:
        if warnings:
            _mp_logs.warn_copy(
                r"Got a non-vine slicer as an input. Use `vineyard=True` to remove this copy."
            )
        from multipers._slicer_meta import Slicer

        slicer = Slicer(slicer, vineyard=True, backend="matrix")

    direction_ = np.ascontiguousarray(direction, dtype=slicer.dtype)
    if box is None:
        box = slicer.filtration_bounds()

    dtype = np.dtype(slicer.dtype)
    if dtype not in AVAILABLE_MMA_FLOAT_DTYPES:
        supported = tuple(dt.name for dt in AVAILABLE_MMA_FLOAT_DTYPES)
        raise ValueError(
            f"Slicer must be float-like and enabled in options.py. Got {slicer.dtype}. Supported dtypes: {supported}."
        )

    approx_mod = _nb._compute_module_approximation_from_slicer(
        slicer,
        direction_,
        max_error,
        np.asarray(box, dtype=dtype),
        threshold,
        complete,
        verbose,
        n_jobs,
    )

    if unsqueeze_grid is not None:
        if verbose:
            print("Reevaluating module in filtration grid...", end="", flush=True)
        approx_mod.evaluate_in_grid(unsqueeze_grid)
        from multipers.grids import compute_bounding_box

        if len(approx_mod):
            approx_mod.set_box(compute_bounding_box(approx_mod))
        if verbose:
            print("Done.", flush=True)

    return approx_mod


def module_approximation(
    input: Union[SimplexTreeMulti_type, Slicer_type, tuple],
    box: Optional[np.ndarray] = None,
    max_error: float = -1,
    nlines: int = 557,
    from_coordinates: bool = False,
    complete: bool = True,
    threshold: bool = False,
    verbose: bool = False,
    ignore_warnings: bool = False,
    direction: Iterable[float] = (),
    swap_box_coords: Iterable[int] = (),
    *,
    n_jobs: int = -1,
) -> PyModule_type:
    if isinstance(input, tuple) or isinstance(input, list):
        dtype = next((np.dtype(s.dtype) for s in input if hasattr(s, "dtype")), None)
    else:
        dtype = np.dtype(input.dtype) if hasattr(input, "dtype") else None
    constructor = _MMA_EMPTY.get(dtype, None)

    if isinstance(input, tuple) or isinstance(input, list):
        if not all(is_slicer(s) and (s.is_minpres or len(s) == 0) for s in input):
            raise ValueError(
                "Modules cannot be merged unless they are minimal presentations."
            )
        if not (
            np.unique([s.minpres_degree for s in input if len(s)], return_counts=True)[
                1
            ]
            <= 1
        ).all():
            raise ValueError(
                "Multiple modules are at the same degree, cannot merge modules"
            )
        if len(input) == 0:
            return (
                constructor() if constructor is not None else available_pymodules[0]()
            )
        modules = tuple(
            Parallel(n_jobs=n_jobs, prefer="threads")(
                delayed(module_approximation)(
                    input=slicer,
                    box=box,
                    max_error=max_error,
                    nlines=nlines,
                    from_coordinates=from_coordinates,
                    complete=complete,
                    threshold=threshold,
                    verbose=verbose,
                    ignore_warnings=ignore_warnings,
                    direction=direction,
                    swap_box_coords=swap_box_coords,
                    n_jobs=n_jobs,
                )
                for slicer in input
            )
        )
        non_empty_modules = tuple(m for m in modules if len(m))
        if len(non_empty_modules) == 0:
            return (
                constructor() if constructor is not None else available_pymodules[0]()
            )
        box = np.array(
            [
                np.min([m.get_box()[0] for m in non_empty_modules], axis=0),
                np.max([m.get_box()[1] for m in non_empty_modules], axis=0),
            ]
        )
        if constructor is None:
            raise ValueError(f"Unsupported module dtype {dtype} for module merge.")
        mod = constructor().set_box(box)
        for i, m in enumerate(modules):
            mod.merge(m, input[i].minpres_degree)
        return mod

    if len(input) == 0:
        if verbose:
            print("Empty input, returning the trivial module.")
        return constructor() if constructor is not None else available_pymodules[0]()

    direction = np.asarray(direction, dtype=np.float64)
    swap_box_coords = np.asarray(tuple(swap_box_coords), dtype=np.int32)
    if box is None:
        box = np.empty((0, 0), dtype=np.float64)
    else:
        box = np.asarray(box, dtype=np.float64)

    return _nb._module_approximation_single_input(
        input=input,
        box=box,
        max_error=max_error,
        nlines=nlines,
        from_coordinates=from_coordinates,
        complete=complete,
        threshold=threshold,
        verbose=verbose,
        ignore_warnings=ignore_warnings,
        direction=direction,
        swap_box_coords=swap_box_coords,
        n_jobs=n_jobs,
    )


__all__ = [
    *(cls.__name__ for cls in available_pymodules),
    *(cls.__name__ for cls in available_pysummands),
    *(cls.__name__ for cls in available_pyboxes),
    *(name for name in vars(_mma) if name.startswith("from_dump_")),
    "available_pymodules",
    "PyModule_type",
    "PySummand_type",
    "PyBox_type",
    "is_mma",
    "module_approximation",
    "module_approximation_from_slicer",
]
