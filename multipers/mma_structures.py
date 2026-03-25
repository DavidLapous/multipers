from __future__ import annotations

import pickle
from functools import reduce
from operator import or_
from typing import Any, Callable, Iterable, Sequence

import numpy as np
from joblib import Parallel, delayed

import multipers.grids as mpg
from multipers import _mma_nanobind as _nb


available_pymodules = tuple(
    cls for name, cls in sorted(vars(_nb).items()) if name.startswith("PyModule_")
)
available_pysummands = tuple(
    cls for name, cls in sorted(vars(_nb).items()) if name.startswith("PySummand_")
)
available_pyboxes = tuple(
    cls for name, cls in sorted(vars(_nb).items()) if name.startswith("PyBox_")
)

for _cls in (*available_pymodules, *available_pysummands, *available_pyboxes):
    globals()[_cls.__name__] = _cls

for _name, _value in vars(_nb).items():
    if _name.startswith("from_dump_"):
        globals()[_name] = _value

PyModule_type = reduce(or_, available_pymodules) if available_pymodules else Any
PySummand_type = reduce(or_, available_pysummands) if available_pysummands else Any
PyBox_type = reduce(or_, available_pyboxes) if available_pyboxes else Any


def _module_getitem(self, i: int):
    if isinstance(i, slice) and i == slice(None):
        return self
    return self._get_summand(i)


def _module_iter(self):
    for i in range(len(self)):
        yield self[i]


def _module_dump(self, path: str | None = None):
    dump = self._get_dump()
    if path is not None:
        with open(path, "wb") as handle:
            pickle.dump(dump, handle)
    return dump


def _module_getstate(self):
    return self.dump()


def _module_setstate(self, dump):
    if isinstance(dump, str):
        with open(dump, "rb") as handle:
            dump = pickle.load(handle)
    self._load_dump(dump)


def _module_reconstruct(cls, state):
    return cls()._load_dump(state)


def _module_reduce(self):
    return (_module_reconstruct, (type(self), self.__getstate__()))


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
        cls.__getitem__ = _module_getitem
        cls.__iter__ = _module_iter
        cls.dump = _module_dump
        cls.__getstate__ = _module_getstate
        cls.__setstate__ = _module_setstate
        cls.__reduce__ = _module_reduce
        cls._bc_to_full = staticmethod(_module_bc_to_full)
        cls._threshold_bc = staticmethod(_module_threshold_bc)
        if np.issubdtype(np.dtype(cls().dtype), np.floating):
            cls.plot = _module_plot
            cls.landscapes = _module_landscapes
            cls.representation = _module_representation
            cls.barcode2 = _module_barcode2


_install_python_api()

__all__ = [
    *(cls.__name__ for cls in available_pymodules),
    *(cls.__name__ for cls in available_pysummands),
    *(cls.__name__ for cls in available_pyboxes),
    *(name for name in vars(_nb) if name.startswith("from_dump_")),
    "available_pymodules",
    "PyModule_type",
    "PySummand_type",
    "PyBox_type",
    "is_mma",
]
