from collections.abc import Iterable, Sequence
from itertools import product
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, TransformerMixin
from tqdm import tqdm

import multipers as mp
from multipers.array_api import api_from_tensor, api_from_tensors
from multipers.filtrations.density import available_kernels, convolution_signed_measures
from multipers.grids import compute_grid, todense
from multipers.point_measure import rank_decomposition_by_rectangles, signed_betti


def batch_signed_measure_convolutions(
    signed_measures,  # array of shape (num_data,num_pts,D)
    x,  # array of shape (num_x, D) or (num_data, num_x, D)
    bandwidth,  # either float or matrix if multivariate kernel
    kernel: available_kernels,
    api=None,
):
    """
    Input
    -----
     - signed_measures: unragged, of shape (num_data, num_pts, D+1)
       where last coord is weights, (0 for dummy points)
     - x : the points to convolve (num_x,D)
     - bandwidth : the bandwidths or covariance matrix inverse or ... of the kernel
     - kernel : "gaussian", "multivariate_gaussian", "exponential", or Callable (x_i, y_i, bandwidth)->float

    Output
    ------
    Array of shape (num_convolutions, (num_axis), num_data,
    Array of shape (num_convolutions, (num_axis), num_data, max_x_size)
    """
    from multipers.filtrations.density import _kernel

    if api is None:
        api = api_from_tensors(signed_measures, x)
    if signed_measures.ndim == 2:
        signed_measures = signed_measures[None, :, :]
    sms = signed_measures[..., :-1]
    weights = signed_measures[..., -1]
    _sms = api.LazyTensor(api.ascontiguous(sms[..., None, :]))
    _x = api.ascontiguous(x[..., None, :, :])

    sms_kernel = _kernel(kernel)(_sms, _x, bandwidth)
    out = (sms_kernel * api.ascontiguous(weights[..., None, None])).sum(
        signed_measures.ndim - 2
    )
    assert out.shape[-1] == 1, "Pykeops bug fixed, TODO : refix this "
    out = out[..., 0]  ## pykeops bug + ensures its a tensor
    # assert out.shape == (x.shape[0], x.shape[1]), f"{x.shape=}, {out.shape=}"
    return out


def sm2deep(signed_measure, api=None):
    if api is None:
        api = api_from_tensor(signed_measure[0])
    dirac_positions, dirac_signs = signed_measure
    dtype = dirac_positions.dtype
    new_shape = list(dirac_positions.shape)
    new_shape[1] += 1
    c = api.empty(new_shape, dtype=dtype)
    c[:, :-1] = dirac_positions
    c[:, -1] = api.astensor(dirac_signs)
    return c


def deep_unrag(sms, api=None):
    if api is None:
        api = api_from_tensor(sms[0][0])
    num_sm = len(sms)
    if num_sm == 0:
        return api.tensor([])
    first = sms[0][0]
    num_parameters = first.shape[1]
    dtype = first.dtype
    deep_sms = tuple(sm2deep(sm, api=api) for sm in sms)
    max_num_pts = np.max([sm[0].shape[0] for sm in sms])
    unragged_sms = api.zeros((num_sm, max_num_pts, num_parameters + 1), dtype=dtype)

    for data in range(num_sm):
        sm = deep_sms[data]
        a, b = sm.shape
        unragged_sms[data, :a, :b] = sm
    return unragged_sms


def sm_convolution(
    sms,
    grid,
    bandwidth,
    kernel: available_kernels = "gaussian",
    plot: bool = False,
    **plt_kwargs,
):
    dense_grid = todense(grid)
    api = api_from_tensors(sms[0][0], dense_grid)
    sms = deep_unrag(sms, api=api)
    convs = batch_signed_measure_convolutions(
        sms, dense_grid, bandwidth, kernel, api=api
    ).reshape(sms.shape[0], *(len(g) for g in grid))
    if plot:
        from multipers.plots import plot_surfaces
        plot_surfaces((grid, convs), **plt_kwargs)
    return convs


class FilteredComplex2SignedMeasure(BaseEstimator, TransformerMixin):
    """
    Input
    -----
    Iterable[SimplexTreeMulti]

    Output
    ------
    Iterable[ list[signed_measure for degree] ]

    signed measure is either
        - (points : (n x num_parameters) array, weights : (n) int array ) if sparse,
        - else an integer matrix.

    Parameters
    ----------
        - degrees : list of degrees to compute. None correspond to the euler characteristic
        - filtration grid : the grid on which to compute.
        If None, the fit will infer it from
        - fit_fraction : the fraction of data to consider for the fit, seed is controlled by the seed parameter
        - resolution : the resolution of this grid
        - filtration_quantile : filtrations values quantile to ignore
        - grid_strategy:str : 'regular' or 'quantile' or 'exact'
        - normalize filtration : if sparse, will normalize all filtrations.
        - expand : expands the simplextree to compute correctly the degree, for
        flag complexes
        - invariant : the topological invariant to produce the signed measure.
        Choices are "hilbert" or "euler". Will add rank invariant later.
        - num_collapse : Either an int or "full". Collapse the complex before
        doing computation.
        - _möbius_inversion : if False, will not do the mobius inversion. output
        has to be a matrix then.
        - enforce_null_mass : Returns a zero mass measure, by thresholding the
        module if True.
    """

    def __init__(
        self,
        # homological degrees + None for euler
        degrees: list[int | None] = [],
        rank_degrees: list[int] = [],  # same for rank invariant
        filtration_grid: (
            Sequence[Sequence[np.ndarray]]
            # filtration values to consider. Format : [ filtration values of Fi for Fi:filtration values of parameter i]
            | None
        ) = None,
        progress=False,  # tqdm
        num_collapses: int | str = 0,  # edge collapses before computing
        n_jobs=None,
        resolution: (
            Iterable[int] | int | None
        ) = None,  # when filtration grid is not given, the resolution of the filtration grid to infer
        # sparse=True, # sparse output # DEPRECATED TO Ssigned measure formatter
        plot: bool = False,
        filtration_quantile: float = 0.0,  # quantile for inferring filtration grid
        # wether or not to do the möbius inversion (not recommended to touch)
        # _möbius_inversion: bool = True,
        expand=False,  # expand the simplextree befoe computing the homology
        normalize_filtrations: bool = False,
        # exact_computation:bool=False, # compute the exact signed measure.
        grid_strategy: str = "exact",
        seed: int = 0,  # if fit_fraction is not 1, the seed sampling
        fit_fraction=1,  # the fraction of the data on which to fit
        out_resolution: Iterable[int] | int | None = None,
        individual_grid: Optional[
            bool
        ] = None,  # Can be significantly faster for some grid strategies, but can drop statistical performance
        enforce_null_mass: bool = False,
        flatten=True,
        backend: Optional[str] = None,
    ):
        super().__init__()
        self.degrees = degrees
        self.rank_degrees = rank_degrees
        self.filtration_grid = filtration_grid
        self.progress = progress
        self.num_collapses = num_collapses
        self.n_jobs = n_jobs
        self.resolution = resolution
        self.plot = plot
        self.backend = backend
        # self.sparse=sparse # TODO : deprecate
        self.filtration_quantile = filtration_quantile
        # Will only work for non sparse output. (discrete matrices cannot be "rescaled")
        self.normalize_filtrations = normalize_filtrations
        self.grid_strategy = grid_strategy
        # self._möbius_inversion = _möbius_inversion
        self._reconversion_grid = None
        self.expand = expand
        # will only refit the grid if filtration_grid has never been given.
        self._refit_grid = None
        self.seed = seed
        self.fit_fraction = fit_fraction
        self._transform_st = None
        self.out_resolution = out_resolution
        self.individual_grid = individual_grid
        self.enforce_null_mass = enforce_null_mass
        self._default_mass_location = None
        self.flatten = flatten
        self.num_parameters: int = 0

        return

    @staticmethod
    def _is_filtered_complex(input):
        return mp.simplex_tree_multi.is_simplextree_multi(input) or mp.slicer.is_slicer(
            input, allow_minpres=True
        )

    def _input_checks(self, X):
        assert len(X) > 0, "No filtered complex found. Cannot fit."
        assert self._is_filtered_complex(
            X[0][0]
        ), f"X[0] is not a known filtered complex, {X[0]=}, nor X[0][0]."
        self._num_axis = len(X[0])
        first = X[0][0]
        assert (
            not mp.slicer.is_slicer(first) or not self.expand
        ), "Cannot expand slicers."
        assert not mp.slicer.is_slicer(first) or not (
            isinstance(first, Union[tuple, list]) and first[0].is_minpres
        ), "Multi-degree minpres are not supported yet as an input. This can still be computed by providing a backend."

    def _infer_filtration(self, X):
        self.num_parameters = X[0][0].num_parameters
        indices = np.random.choice(
            len(X), min(int(self.fit_fraction * len(X)) + 1, len(X)), replace=False
        )
        ## ax, num_x
        filtrations = tuple(
            tuple(
                compute_grid(x, strategy="exact")
                for x in (X[idx][ax] for idx in indices)
            )
            for ax in range(self._num_axis)
        )
        num_parameters = len(filtrations[0][0])
        assert (
            num_parameters == self.num_parameters
        ), f"Internal error, got {num_parameters=} and {self.num_parameters=}"

        filtrations_values = [
            [
                np.unique(np.concatenate([x[i] for x in filtrations[ax]]))
                for i in range(num_parameters)
            ]
            for ax in range(self._num_axis)
        ]
        ## ax, param, gridsize
        filtration_grid = tuple(
            compute_grid(
                filtrations_values[ax],
                resolution=self.resolution,
                strategy=self.grid_strategy,
            )
            for ax in range(self._num_axis)
        )  # TODO :use more parameters
        self.filtration_grid = filtration_grid
        return filtration_grid

    def _params_check(self):
        assert (
            self.resolution is not None
            or self.filtration_grid is not None
            or self.grid_strategy == "exact"
            or self.individual_grid
        ), "For non exact filtrations, a resolution has to be specified."

    def fit(self, X, y=None):  # Todo : infer filtration grid ? quantiles ?
        self._params_check()
        self._input_checks(X)

        if isinstance(self.resolution, int):
            self.resolution = [self.resolution] * self.num_parameters

        self.individual_grid = (
            self.individual_grid
            if self.individual_grid is not None
            else self.grid_strategy
            in ["regular_closest", "exact", "quantile", "partition"]
        )

        if (
            not self.enforce_null_mass
            and self.individual_grid
            or self.filtration_grid is not None
        ):
            self._refit_grid = False
        else:
            self._refit_grid = True

        if self._refit_grid:
            self._infer_filtration(X=X)
        if self.out_resolution is None:
            self.out_resolution = self.resolution
        # elif isinstance(self.out_resolution, int):
        #     self.out_resolution = [self.out_resolution] * self.num_parameters
        if self.normalize_filtrations and not self.individual_grid:
            # self._reconversion_grid = [np.linspace(0,1, num=len(f), dtype=float) for f in self.filtration_grid] ## This will not work for non-regular grids...
            self._reconversion_grid = [
                [(f - np.min(f)) / np.std(f) for f in F] for F in self.filtration_grid
            ]  # not the best, but better than some weird magic
        # elif not self.sparse: # It actually renormalizes the filtration !!
        # self._reconversion_grid = [np.linspace(0,r, num=r, dtype=int) for r in self.out_resolution]
        else:
            self._reconversion_grid = self.filtration_grid
        ## ax, num_param
        self._default_mass_location = (
            np.asarray([[g[-1] for g in F] for F in self.filtration_grid])
            if self.enforce_null_mass
            else None
        )
        return self

    def transform1(
        self,
        simplextree,
        ax,
        # _reconversion_grid,
        thread_id: str = "",
    ):
        # st = mp.SimplexTreeMulti(st, num_parameters=st.num_parameters)  # COPY
        if self.individual_grid:
            filtration_grid = compute_grid(
                simplextree, strategy=self.grid_strategy, resolution=self.resolution
            )
            mass_default = (
                self._default_mass_location[ax] if self.enforce_null_mass else None
            )
            if self.enforce_null_mass:
                filtration_grid = [
                    np.concatenate([f, [d]], axis=0)
                    for f, d in zip(filtration_grid, mass_default)
                ]
            _reconversion_grid = filtration_grid
        else:
            filtration_grid = self.filtration_grid[ax]
            _reconversion_grid = self._reconversion_grid[ax]
            mass_default = (
                self._default_mass_location[ax] if self.enforce_null_mass else None
            )

        st = simplextree.grid_squeeze(filtration_grid=filtration_grid)
        if st.num_parameters == 2 and mp.simplex_tree_multi.is_simplextree_multi(st):
            st.collapse_edges(num=self.num_collapses, max_dimension=1)
        int_degrees = np.asarray([d for d in self.degrees if d is not None], dtype=int)
        # EULER. First as there is prune above dimension below
        if self.expand and None in self.degrees:
            st.expansion(st.num_vertices)
        signed_measures_euler = (
            mp.signed_measure(
                st,
                degrees=[None],
                plot=self.plot,
                mass_default=mass_default,
                invariant="euler",
                # thread_id=thread_id,
                backend=self.backend,
                grid=_reconversion_grid,
            )[0]
            if None in self.degrees
            else []
        )

        if self.expand and len(int_degrees) > 0:
            st.expansion(np.max(int_degrees) + 1)
        if len(int_degrees) > 0:
            st.prune_above_dimension(
                np.max(np.concatenate([int_degrees, self.rank_degrees])) + 1
            )  # no need to compute homology beyond this
        signed_measures_pers = (
            mp.signed_measure(
                st,
                degrees=int_degrees,
                mass_default=mass_default,
                plot=self.plot,
                invariant="hilbert",
                thread_id=thread_id,
                backend=self.backend,
                grid=_reconversion_grid,
            )
            if len(int_degrees) > 0
            else []
        )
        if self.plot:
            plt.show()
        if self.expand and len(self.rank_degrees) > 0:
            st.expansion(np.max(self.rank_degrees) + 1)
        if len(self.rank_degrees) > 0:
            st.prune_above_dimension(
                np.max(self.rank_degrees) + 1
            )  # no need to compute homology beyond this
        signed_measures_rank = (
            mp.signed_measure(
                st,
                degrees=self.rank_degrees,
                mass_default=mass_default,
                plot=self.plot,
                invariant="rank",
                thread_id=thread_id,
                backend=self.backend,
                grid=_reconversion_grid,
            )
            if len(self.rank_degrees) > 0
            else []
        )
        if self.plot:
            plt.show()

        count = 0
        signed_measures = []
        for d in self.degrees:
            if d is None:
                signed_measures.append(signed_measures_euler)
            else:
                signed_measures.append(signed_measures_pers[count])
                count += 1
        signed_measures += signed_measures_rank
        return signed_measures

    def transform(self, X):
        ## X of shape (num_x, num_axis, filtered_complex
        assert (
            self.filtration_grid is not None and self._reconversion_grid is not None
        ) or self.individual_grid, "Fit first"

        def todo_x(x):
            return tuple(self.transform1(x_axis, j) for j, x_axis in enumerate(x))

        ## out shape num_x, num_axis, degree, sm
        out = tuple(
            Parallel(n_jobs=self.n_jobs, backend="threading")(
                delayed(todo_x)(x) for x in X
            )
        )
        # out = Parallel(n_jobs=self.n_jobs, backend="threading")(
        #     delayed(self.transform1)(to_st, thread_id=str(thread_id))
        #     for thread_id, to_st in tqdm(
        #         enumerate(X),
        #         disable=not self.progress,
        #         desc="Computing signed measure decompositions",
        #     )
        # )
        return out


class SimplexTree2SignedMeasure(FilteredComplex2SignedMeasure):
    def __init__(
        self,
        # homological degrees + None for euler
        degrees: list[int | None] = [],
        rank_degrees: list[int] = [],  # same for rank invariant
        filtration_grid: (
            Sequence[Sequence[np.ndarray]]
            # filtration values to consider. Format : [ filtration values of Fi for Fi:filtration values of parameter i]
            | None
        ) = None,
        progress=False,  # tqdm
        num_collapses: int | str = 0,  # edge collapses before computing
        n_jobs=None,
        resolution: (
            Iterable[int] | int | None
        ) = None,  # when filtration grid is not given, the resolution of the filtration grid to infer
        # sparse=True, # sparse output # DEPRECATED TO Ssigned measure formatter
        plot: bool = False,
        filtration_quantile: float = 0.0,  # quantile for inferring filtration grid
        # wether or not to do the möbius inversion (not recommended to touch)
        # _möbius_inversion: bool = True,
        expand=False,  # expand the simplextree befoe computing the homology
        normalize_filtrations: bool = False,
        # exact_computation:bool=False, # compute the exact signed measure.
        grid_strategy: str = "exact",
        seed: int = 0,  # if fit_fraction is not 1, the seed sampling
        fit_fraction=1,  # the fraction of the data on which to fit
        out_resolution: Iterable[int] | int | None = None,
        individual_grid: Optional[
            bool
        ] = None,  # Can be significantly faster for some grid strategies, but can drop statistical performance
        enforce_null_mass: bool = False,
        flatten=True,
        backend: Optional[str] = None,
    ):
        stuff = locals()
        stuff.pop("self")
        keys = list(stuff.keys())
        for key in keys:
            if key.startswith("__"):
                stuff.pop(key)
        super().__init__(**stuff)
        from warnings import warn

        warn("This class is deprecated, use FilteredComplex2SignedMeasure instead.")


# class SimplexTrees2SignedMeasures(SimplexTree2SignedMeasure):
#     """
#     Input
#     -----
#
#     (data) x (axis, e.g. different bandwidths for simplextrees) x (simplextree)
#
#     Output
#     ------
#     (data) x (axis) x (degree) x (signed measure)
#     """
#
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#         self._num_st_per_data = None
#         # self._super_model=SimplexTree2SignedMeasure(**kwargs)
#         self._filtration_grids = None
#         return
#
#     def fit(self, X, y=None):
#         if len(X) == 0:
#             return self
#         try:
#             self._num_st_per_data = len(X[0])
#         except:
#             raise Exception(
#                 "Shape has to be (num_data, num_axis), dtype=SimplexTreeMulti"
#             )
#         self._filtration_grids = []
#         for axis in range(self._num_st_per_data):
#             self._filtration_grids.append(
#                 super().fit([x[axis] for x in X]).filtration_grid
#             )
#             # self._super_fits.append(truc)
#         # self._super_fits_params = [super().fit([x[axis] for x in X]).get_params() for axis in range(self._num_st_per_data)]
#         return self
#
#     def transform(self, X):
#         if self.normalize_filtrations:
#             _reconversion_grids = [
#                 [np.linspace(0, 1, num=len(f), dtype=float) for f in F]
#                 for F in self._filtration_grids
#             ]
#         else:
#             _reconversion_grids = self._filtration_grids
#
#         def todo(x):
#             # return [SimplexTree2SignedMeasure().set_params(**transformer_params).transform1(x[axis]) for axis,transformer_params in enumerate(self._super_fits_params)]
#             out = [
#                 self.transform1(
#                     x[axis],
#                     filtration_grid=filtration_grid,
#                     _reconversion_grid=_reconversion_grid,
#                 )
#                 for axis, filtration_grid, _reconversion_grid in zip(
#                     range(self._num_st_per_data),
#                     self._filtration_grids,
#                     _reconversion_grids,
#                 )
#             ]
#             return out
#
#         return Parallel(n_jobs=self.n_jobs, backend="threading")(
#             delayed(todo)(x)
#             for x in tqdm(
#                 X,
#                 disable=not self.progress,
#                 desc="Computing Signed Measures from simplextrees.",
#             )
#         )


def rescale_sparse_signed_measure(
    signed_measure, filtration_weights, normalize_scales=None
):
    # from copy import deepcopy
    #
    # out = deepcopy(signed_measure)

    if filtration_weights is None and normalize_scales is None:
        return signed_measure

    # if normalize_scales is None:
    #     out = tuple(
    #         (
    #             _cat(
    #                 tuple(
    #                     signed_measure[degree][0][:, parameter]
    #                     * filtration_weights[parameter]
    #                     for parameter in range(num_parameters)
    #                 ),
    #                 axis=1,
    #             ),
    #             signed_measure[degree][1],
    #         )
    #         for degree in range(len(signed_measure))
    #     )
    # for degree in range(len(signed_measure)):  # degree
    #     for parameter in range(len(filtration_weights)):
    #         signed_measure[degree][0][:, parameter] *= filtration_weights[parameter]
    #         # TODO Broadcast w.r.t. the parameter
    # out = tuple(
    #     _cat(
    #         tuple(
    #             signed_measure[degree][0][:, [parameter]]
    #             * filtration_weights[parameter]
    #             / (
    #                 normalize_scales[degree][parameter]
    #                 if normalize_scales is not None
    #                 else 1
    #             )
    #             for parameter in range(num_parameters)
    #         ),
    #         axis=1,
    #     )
    #     for degree in range(len(signed_measure))
    # )
    out = tuple(
        (
            signed_measure[degree][0]
            * (1 if filtration_weights is None else filtration_weights.reshape(1, -1))
            / (
                normalize_scales[degree].reshape(1, -1)
                if normalize_scales is not None
                else 1
            ),
            signed_measure[degree][1],
        )
        for degree in range(len(signed_measure))
    )
    # for degree in range(len(out)):
    #     for parameter in range(len(filtration_weights)):
    #         out[degree][0][:, parameter] *= (
    #             filtration_weights[parameter] / normalize_scales[degree][parameter]
    #         )
    return out


class SignedMeasureFormatter(BaseEstimator, TransformerMixin):
    """
    Input
    -----

    (data) x (degree) x (signed measure) or (data) x (axis) x (degree) x (signed measure)

    Iterable[list[signed_measure_matrix of degree]] or Iterable[previous].

    The second is meant to use multiple choices for signed measure input. An example of usage : they come from a Rips + Density with different bandwidth.
    It is controlled by the axis parameter.

    Output
    ------

    Iterable[list[(reweighted)_sparse_signed_measure of degree]]

    or (deep format)

    Tensor of shape (num_axis*num_degrees, data, max_num_pts, num_parameters)
    """

    def __init__(
        self,
        filtrations_weights: Optional[Iterable[float]] = None,
        normalize=False,
        plot: bool = False,
        unsparse: bool = False,
        axis: int = -1,
        resolution: int | Iterable[int] = 50,
        flatten: bool = False,
        deep_format: bool = False,
        unrag: bool = True,
        n_jobs: int = 1,
        verbose: bool = False,
        integrate: bool = False,
        grid_strategy="regular",
    ):
        super().__init__()
        self.filtrations_weights = filtrations_weights
        self.num_parameters: int = 0
        self.plot = plot
        self.unsparse = unsparse
        self.n_jobs = n_jobs
        self.axis = axis
        self._num_axis = 0
        self.resolution = resolution
        self._filtrations_bounds = None
        self.flatten = flatten
        self.normalize = normalize
        self._normalization_factors = None
        self.deep_format = deep_format
        self.unrag = unrag
        assert (
            not self.deep_format or not self.unsparse or not self.integrate
        ), "One post processing at the time."
        self.verbose = verbose
        self._num_degrees = 0
        self.integrate = integrate
        self.grid_strategy = grid_strategy
        self._infered_grids = None
        self._axis_iterator = None
        self._backend = None
        return

    def _get_filtration_bounds(self, X, axis):
        stuff = [
            self._backend.cat(
                [sm[axis][degree][0] for sm in X],
                axis=0,
            )
            for degree in range(self._num_degrees)
        ]
        sizes_ = np.array([len(x) == 0 for x in stuff])
        assert np.all(~sizes_), f"Degree axis {np.where(sizes_)} is/are trivial !"

        filtrations_bounds = self._backend.asnumpy(
            self._backend.stack(
                [
                    self._backend.stack(
                        [
                            self._backend.minvalues(f, axis=0),
                            self._backend.maxvalues(f, axis=0),
                        ]
                    )
                    for f in stuff
                ]
            )
        )  ## don't want to rescale gradient of normalization
        normalization_factors = (
            filtrations_bounds[:, 1] - filtrations_bounds[:, 0]
            if self.normalize
            else None
        )
        # print("Normalization factors : ",self._normalization_factors)
        if (normalization_factors == 0).any():
            indices = normalization_factors == 0
            # warn(f"Constant filtration encountered, at degree, parameter {indices} and axis {self.axis}.")
            normalization_factors[indices] = 1
        return filtrations_bounds, normalization_factors

    def _plot_signed_measures(self, sms: Iterable[np.ndarray], size=4):
        from multipers.plots import plot_signed_measure

        num_degrees = len(sms[0])
        num_imgs = len(sms)
        fig, axes = plt.subplots(
            ncols=num_degrees,
            nrows=num_imgs,
            figsize=(size * num_degrees, size * num_imgs),
        )
        axes = np.asarray(axes).reshape(num_imgs, num_degrees)
        # assert axes.ndim==2, "Internal error"
        for i, sm in enumerate(sms):
            for j, sm_of_degree in enumerate(sm):
                plot_signed_measure(sm_of_degree, ax=axes[i, j])

    @staticmethod
    def _check_sm(sm) -> bool:
        return (
            isinstance(sm, tuple)
            and hasattr(sm[0], "ndim")
            and sm[0].ndim == 2
            and len(sm) == 2
        )

    def _check_axis(self, X):
        # axes should be (num_data, num_axis, num_degrees, (signed_measure))
        if len(X) == 0:
            return
        if len(X[0]) == 0:
            return
        if self._check_sm(X[0][0]):
            self._has_axis = False
            self._num_axis = 1
            self._axis_iterator = [slice(None)]
            return
        assert self._check_sm(  ## vaguely checks that its a signed measure
            _sm := X[0][0][0]
        ), f"Cannot take this input. # data, axis, degrees, sm.\n Got {_sm} of type {type(_sm)}"

        self._has_axis = True
        self._num_axis = len(X[0])
        self._axis_iterator = range(self._num_axis) if self.axis == -1 else [self.axis]

    def _check_backend(self, X):
        if self._has_axis:
            # data, axis, degrees, (pts, weights)
            first_sm = X[0][0][0][0]
        else:
            first_sm = X[0][0][0]
        self._backend = api_from_tensor(first_sm, verbose=self.verbose)

    def _check_measures(self, X):
        if self._has_axis:
            first_sm = X[0][0]
        else:
            first_sm = X[0]
        self._num_degrees = len(first_sm)
        self.num_parameters = first_sm[0][0].shape[1]

    def _check_resolution(self):
        assert self.num_parameters > 0, "Num parameters hasn't been initialized."
        if isinstance(self.resolution, int):
            self.resolution = [self.resolution] * self.num_parameters
        self.resolution = np.asarray(self.resolution, dtype=int)
        assert (
            self.resolution.shape[0] == self.num_parameters
        ), "Resolution doesn't have a proper size."

    def _check_weights(self):
        if self.filtrations_weights is None:
            return
        assert (
            self.filtrations_weights.shape[0] == self.num_parameters
        ), "Filtration weights don't have a proper size"

    def _infer_grids(self, X):
        # Computes normalization factors
        if self.normalize:
            # if self._has_axis and self.axis == -1:
            self._filtrations_bounds = []
            self._normalization_factors = []
            for ax in self._axis_iterator:
                (
                    filtration_bounds,
                    normalization_factors,
                ) = self._get_filtration_bounds(X, axis=ax)
                self._filtrations_bounds.append(filtration_bounds)
                self._normalization_factors.append(normalization_factors)
            self._filtrations_bounds = self._backend.astensor(self._filtrations_bounds)
            self._normalization_factors = self._backend.astensor(
                self._normalization_factors
            )
            # else:
            #     (
            #         self._filtrations_bounds,
            #         self._normalization_factors,
            #     ) = self._get_filtration_bounds(
            #         X, axis=self._axis_iterator[0]
            #     )  ## axis = slice(None)
        elif self.integrate or self.unsparse or self.deep_format:
            filtration_values = [
                np.concatenate(
                    [
                        (
                            stuff
                            if isinstance(stuff := x[ax][degree][0], np.ndarray)
                            else stuff.detach().numpy()
                        )
                        for x in X
                        for degree in range(self._num_degrees)
                    ]
                )
                for ax in self._axis_iterator
            ]
            # axis, filtration_values
            filtration_values = [
                self._backend.astensor(
                    compute_grid(
                        f_ax.T, resolution=self.resolution, strategy=self.grid_strategy
                    )
                )
                for f_ax in filtration_values
            ]
            self._infered_grids = filtration_values

    def _print_stats(self, X):
        print("------------SignedMeasureFormatter------------")
        print("---- Parameters")
        print(f"Number of axis : {self._num_axis}")
        print(f"Number of degrees : {self._num_degrees}")
        print(f"Filtration bounds : \n{self._filtrations_bounds}")
        print(f"Normalization factor : \n{self._normalization_factors}")
        if self._infered_grids is not None:
            print(
                f"Filtration grid shape : \n \
                {tuple(tuple(len(f) for f in F) for F in self._infered_grids)}"
            )
        print("---- SM stats")
        print("In axis :", self._num_axis)
        sizes = [
            [[len(xd[1]) for xd in x[ax]] for x in X] for ax in self._axis_iterator
        ]
        print(f"Size means (axis) x (degree): {np.mean(sizes, axis=(1))}")
        print(f"Size std : {np.std(sizes, axis=(1))}")
        print("----------------------------------------------")

    def fit(self, X, y=None):
        # Gets a grid. This will be the max in each coord+1
        if (
            len(X) == 0
            or len(X[0]) == 0
            or (self.axis is not None and len(X[0][0][0]) == 0)
        ):
            return self

        self._check_axis(X)
        self._check_backend(X)
        self._check_measures(X)
        self._check_resolution()
        self._check_weights()
        # if not sparse : not recommended.

        self._infer_grids(X)
        if self.verbose:
            self._print_stats(X)
        return self

    def unsparse_signed_measure(self, sparse_signed_measure):
        filtrations = self._infered_grids  # ax, filtration
        out = []
        for filtrations_of_ax, ax in zip(filtrations, self._axis_iterator, strict=True):
            sparse_signed_measure_of_ax = sparse_signed_measure[ax]
            measure_of_ax = []
            for pts, weights in sparse_signed_measure_of_ax:  # over degree
                signed_measure, _ = np.histogramdd(
                    pts, bins=filtrations_of_ax, weights=weights
                )
                if self.flatten:
                    signed_measure = signed_measure.flatten()
                measure_of_ax.append(signed_measure)
            out.append(np.asarray(measure_of_ax))

        if self.flatten:
            out = np.concatenate(out).flatten()
        elif self.axis == -1:
            return np.asarray(out)
        else:
            return np.asarray(out)[0]

    @staticmethod
    def _integrate_measure(sm, filtrations):
        from multipers.point_measure import integrate_measure

        return integrate_measure(sm[0], sm[1], filtrations)

    def _rescale_measures(self, X):
        def rescale_from_sparse(sparse_signed_measure):
            if self.axis == -1 and self._has_axis:
                return tuple(
                    rescale_sparse_signed_measure(
                        sparse_signed_measure[ax],
                        filtration_weights=self.filtrations_weights,
                        normalize_scales=n,
                    )
                    for ax, n in zip(
                        self._axis_iterator, self._normalization_factors, strict=True
                    )
                )
            return rescale_sparse_signed_measure(  ## axis iterator is of size 1 here
                sparse_signed_measure,
                filtration_weights=self.filtrations_weights,
                normalize_scales=self._normalization_factors[0],
            )

        out = tuple(rescale_from_sparse(x) for x in X)
        return out

    def transform(self, X):
        if not self._has_axis or self.axis == -1:
            out = X
        else:
            out = tuple(x[self.axis] for x in X)
            # same format for everyone

        if self._normalization_factors is not None:
            out = self._rescale_measures(out)

        if self.plot:
            # assert ax != -1, "Not implemented"
            self._plot_signed_measures(out)
        if self.integrate:
            filtrations = self._infered_grids
            # if self.axis != -1:
            ax = 0  # if self.axis is None else self.axis # TODO deal with axis -1

            assert ax != -1, "Not implemented. Can only integrate with axis"
            # try:
            out = np.asarray(
                [
                    [
                        self._integrate_measure(x[degree], filtrations=filtrations[ax])
                        for degree in range(self._num_degrees)
                    ]
                    for x in out
                ]
            )
            # except:
            # print(self.axis, ax, filtrations)
            if self.flatten:
                out = out.reshape((len(X), -1))
            # else:
            # out = [[[self._integrate_measure(x[axis][degree],filtrations=filtrations[degree].T) for degree in range(self._num_degrees)] for axis in range(self._num_axis)] for x in out]
        elif self.unsparse:
            out = [self.unsparse_signed_measure(x) for x in out]
        elif self.deep_format:
            num_degrees = self._num_degrees
            out = tuple(
                tuple(sm2deep(sm[axis][degree]) for sm in out)
                for degree in range(num_degrees)
                for axis in self._axis_iterator
            )
            if self.unrag:
                max_num_pts = np.max(
                    [sm.shape[0] for sm_of_axis in out for sm in sm_of_axis]
                )
                num_axis_degree = len(out)
                num_data = len(out[0])
                assert num_axis_degree == num_degrees * (
                    self._num_axis if self._has_axis else 1
                ), f"Bad axis/degree count. Got {num_axis_degree} (Internal error)"
                num_parameters = out[0][0].shape[1]
                dtype = out[0][0].dtype
                unragged_tensor = self._backend.zeros(
                    (
                        num_axis_degree,
                        num_data,
                        max_num_pts,
                        num_parameters,
                    ),
                    dtype=dtype,
                )
                for ax in range(num_axis_degree):
                    for data in range(num_data):
                        sm = out[ax][data]
                        a, b = sm.shape
                        unragged_tensor[ax, data, :a, :b] = sm
                out = unragged_tensor
        return out


class SignedMeasure2Convolution(BaseEstimator, TransformerMixin):
    """
    Discrete convolution of a signed measure

    Input
    -----

    (data) x (degree) x (signed measure)

    Parameters
    ----------
     - filtration_grid : Iterable[array] For each filtration, the filtration values on which to evaluate the grid
     - resolution : int or (num_parameters) : If filtration grid is not given, will infer a grid, with this resolution
     - grid_strategy : the strategy to generate the grid. Available ones are regular, quantile, exact
     - flatten : if true, the output will be flattened
     - kernel : kernel to used to convolve the images.
     - flatten : flatten the images if True
     - progress : progress bar if True
     - backend : sklearn, pykeops or numba.
     - plot : Creates a plot Figure.

    Output
    ------

    (data) x (concatenation of imgs of degree)
    """

    def __init__(
        self,
        filtration_grid: Iterable[np.ndarray] = None,
        kernel: available_kernels = "gaussian",
        bandwidth: float | Iterable[float] = 1.0,
        flatten: bool = False,
        n_jobs: int = 1,
        resolution: int | None = None,
        grid_strategy: str = "regular",
        progress: bool = False,
        backend: str = "pykeops",
        plot: bool = False,
        log_density: bool = False,
        **kde_kwargs,
        #   **kwargs ## DANGEROUS
    ):
        super().__init__()
        self.kernel: available_kernels = kernel
        self.bandwidth = bandwidth
        # self.more_kde_kwargs=kwargs
        self.filtration_grid = filtration_grid
        self.flatten = flatten
        self.progress = progress
        self.n_jobs = n_jobs
        self.resolution = resolution
        self.grid_strategy = grid_strategy
        self._is_input_sparse = None
        self._refit = filtration_grid is None
        self._input_resolution = None
        self._bandwidths = None
        self.diameter = None
        self.backend = backend
        self.plot = plot
        self.log_density = log_density
        self.kde_kwargs = kde_kwargs
        self._api = None
        return

    def fit(self, X, y=None):
        # Infers if the input is sparse given X
        if len(X) == 0:
            return self
        if isinstance(X[0][0], tuple):
            self._is_input_sparse = True

            self._api = api_from_tensor(X[0][0][0], verbose=self.progress)
        else:
            self._is_input_sparse = False

            self._api = api_from_tensor(X, verbose=self.progress)
        # print(f"IMG output is set to {'sparse' if self.sparse else 'matrix'}")
        if not self._is_input_sparse:
            self._input_resolution = X[0][0].shape
            try:
                b = float(self.bandwidth)
                self._bandwidths = [
                    b if b > 0 else -b * s for s in self._input_resolution
                ]
            except:
                self._bandwidths = [
                    b if b > 0 else -b * s
                    for s, b in zip(self._input_resolution, self.bandwidth)
                ]
            return self  # in that case, singed measures are matrices, and the grid is already given

        if self.filtration_grid is None and self.resolution is None:
            raise Exception(
                "Cannot infer filtration grid. Provide either a filtration grid or a resolution."
            )
        # If not sparse : a grid has to be defined
        if self._refit:
            # print("Fitting a grid...", end="")
            pts = self._api.cat(
                [sm[0] for signed_measures in X for sm in signed_measures]
            ).T
            self.filtration_grid = compute_grid(
                pts,
                strategy=self.grid_strategy,
                resolution=self.resolution,
            )
            # print('Done.')
        if self.filtration_grid is not None:
            self.diameter = self._api.norm(
                self._api.astensor([f[-1] - f[0] for f in self.filtration_grid])
            )
            if self.progress:
                print(f"Computed a diameter of {self.diameter}")
        return self

    def _sm2smi(self, signed_measures):
        # print(self._input_resolution, self.bandwidths, _bandwidths)
        from scipy.ndimage import gaussian_filter

        return np.concatenate(
            [
                gaussian_filter(
                    input=signed_measure,
                    sigma=self._bandwidths,
                    mode="constant",
                    cval=0,
                )
                for signed_measure in signed_measures
            ],
            axis=0,
        )

    def _transform_from_sparse(self, X):
        bandwidth = (
            self.bandwidth if self.bandwidth > 0 else -self.bandwidth * self.diameter
        )
        # COMPILE KEOPS FIRST
        dummyx = [X[0]]
        dummyf = [f[:2] for f in self.filtration_grid]
        convolution_signed_measures(
            dummyx,
            filtrations=dummyf,
            bandwidth=bandwidth,
            flatten=self.flatten,
            n_jobs=1,
            kernel=self.kernel,
            backend=self.backend,
        )

        return convolution_signed_measures(
            X,
            filtrations=self.filtration_grid,
            bandwidth=bandwidth,
            flatten=self.flatten,
            n_jobs=self.n_jobs,
            kernel=self.kernel,
            backend=self.backend,
            **self.kde_kwargs,
        )

    def _plot_imgs(self, imgs: Iterable[np.ndarray], size=4):
        from multipers.plots import plot_surface

        imgs = self._api.asnumpy(imgs)
        num_degrees = imgs[0].shape[0]
        num_imgs = len(imgs)
        fig, axes = plt.subplots(
            ncols=num_degrees,
            nrows=num_imgs,
            figsize=(size * num_degrees, size * num_imgs),
        )
        axes = np.asarray(axes).reshape(num_imgs, num_degrees)
        # assert axes.ndim==2, "Internal error"
        for i, img in enumerate(imgs):
            for j, img_of_degree in enumerate(img):
                plot_surface(
                    [self._api.asnumpy(f) for f in self.filtration_grid],
                    img_of_degree,
                    ax=axes[i, j],
                    cmap="Spectral",
                )

    def transform(self, X):
        if self._is_input_sparse is None:
            raise Exception("Fit first")
        if self._is_input_sparse:
            out = self._transform_from_sparse(X)
        else:
            todo = SignedMeasure2Convolution._sm2smi
            out = Parallel(n_jobs=self.n_jobs, backend="threading")(
                delayed(todo)(self, signed_measures)
                for signed_measures in tqdm(
                    X, desc="Computing images", disable=not self.progress
                )
            )
            out = self._api.cat([x[None] for x in out])
        if self.plot and not self.flatten:
            if self.progress:
                print("Plotting convolutions...", end="")
            self._plot_imgs(out)
            if self.progress:
                print("Done !")
        if self.flatten and not self._is_input_sparse:
            out = self._api.cat([x.ravel()[None] for x in out])
        return out


class SignedMeasure2SlicedWassersteinDistance(BaseEstimator, TransformerMixin):
    """
    Transformer from signed measure to distance matrix.

    Input
    -----

    (data) x (degree) x (signed measure)

    Format
    ------
    - a signed measure : tuple of array. (point position) : npts x (num_paramters) and weigths : npts
    - each data is a list of signed measure (for e.g. multiple degrees)

    Output
    ------
    - (degree) x (distance matrix)
    """

    def __init__(
        self,
        n_jobs=None,
        num_directions: int = 10,
        _sliced: bool = True,
        epsilon=-1,
        ground_norm=1,
        progress=False,
        grid_reconversion=None,
        scales=None,
    ):
        super().__init__()
        self.n_jobs = n_jobs
        self._SWD_list = None
        self._sliced = _sliced
        self.epsilon = epsilon
        self.ground_norm = ground_norm
        self.num_directions = num_directions
        self.progress = progress
        self.grid_reconversion = grid_reconversion
        self.scales = scales
        return

    def fit(self, X, y=None):
        from multipers.ml.sliced_wasserstein import (
            SlicedWassersteinDistance,
            WassersteinDistance,
        )

        # _DISTANCE = lambda : SlicedWassersteinDistance(num_directions=self.num_directions) if self._sliced else WassersteinDistance(epsilon=self.epsilon, ground_norm=self.ground_norm) # WARNING if _sliced is false, this distance is not CNSD
        if len(X) == 0:
            return self
        num_degrees = len(X[0])
        self._SWD_list = [
            (
                SlicedWassersteinDistance(
                    num_directions=self.num_directions,
                    n_jobs=self.n_jobs,
                    scales=self.scales,
                )
                if self._sliced
                else WassersteinDistance(
                    epsilon=self.epsilon,
                    ground_norm=self.ground_norm,
                    n_jobs=self.n_jobs,
                )
            )
            for _ in range(num_degrees)
        ]
        for degree, swd in enumerate(self._SWD_list):
            signed_measures_of_degree = [x[degree] for x in X]
            swd.fit(signed_measures_of_degree)
        return self

    def transform(self, X):
        assert self._SWD_list is not None, "Fit first"
        # out = []
        # for degree, swd in tqdm(enumerate(self._SWD_list), desc="Computing distance matrices", total=len(self._SWD_list), disable= not self.progress):
        with tqdm(
            enumerate(self._SWD_list),
            desc="Computing distance matrices",
            total=len(self._SWD_list),
            disable=not self.progress,
        ) as SWD_it:
            # signed_measures_of_degree = [x[degree] for x in X]
            # out.append(swd.transform(signed_measures_of_degree))
            def todo(swd, X_of_degree):
                return swd.transform(X_of_degree)

            out = Parallel(n_jobs=self.n_jobs, prefer="threads")(
                delayed(todo)(swd, [x[degree] for x in X]) for degree, swd in SWD_it
            )
            return np.asarray(out)

    def predict(self, X):
        return self.transform(X)


class SignedMeasures2SlicedWassersteinDistances(BaseEstimator, TransformerMixin):
    """
    Transformer from signed measure to distance matrix.
    Input
    -----
    (data) x opt (axis) x (degree) x (signed measure)

    Format
    ------
    - a signed measure : tuple of array. (point position) : npts x (num_paramters) and weigths : npts
    - each data is a list of signed measure (for e.g. multiple degrees)

    Output
    ------
    - (axis) x (degree) x (distance matrix)
    """

    def __init__(
        self,
        progress=False,
        n_jobs: int = 1,
        scales: Iterable[Iterable[float]] | None = None,
        **kwargs,
    ):  # same init
        self._init_child = SignedMeasure2SlicedWassersteinDistance(
            progress=False, scales=None, n_jobs=-1, **kwargs
        )
        self._axe_iterator = None
        self._childs_to_fit = None
        self.scales = scales
        self.progress = progress
        self.n_jobs = n_jobs
        return

    def fit(self, X, y=None):
        from sklearn.base import clone

        if len(X) == 0:
            return self
        if isinstance(X[0][0], tuple):  # Meaning that there are no axes
            self._axe_iterator = [slice(None)]
        else:
            self._axe_iterator = range(len(X[0]))
        if self.scales is None:
            self.scales = [None]
        else:
            self.scales = np.asarray(self.scales)
            if self.scales.ndim == 1:
                self.scales = np.asarray([self.scales])
        assert (
            self.scales[0] is None or self.scales.ndim == 2
        ), "Scales have to be either None or a list of scales !"
        self._childs_to_fit = [
            clone(self._init_child).set_params(scales=scales).fit([x[axis] for x in X])
            for axis, scales in product(self._axe_iterator, self.scales)
        ]
        print("New axes : ", list(product(self._axe_iterator, self.scales)))
        return self

    def transform(self, X):
        return Parallel(n_jobs=self.n_jobs, prefer="processes")(
            delayed(self._childs_to_fit[child_id].transform)([x[axis] for x in X])
            for child_id, (axis, _) in tqdm(
                enumerate(product(self._axe_iterator, self.scales)),
                desc=f"Computing distances matrices of axis, and scales",
                disable=not self.progress,
                total=len(self._childs_to_fit),
            )
        )
        # [
        # child.transform([x[axis // len(self.scales)] for x in X])
        # for axis, child in tqdm(enumerate(self._childs_to_fit),
        # desc=f"Computing distances of axis", disable=not self.progress, total=len(self._childs_to_fit)
        # )
        # ]


class SimplexTree2RectangleDecomposition(BaseEstimator, TransformerMixin):
    """
    Transformer. 2 parameter SimplexTrees to their respective rectangle decomposition.
    """

    def __init__(
        self,
        filtration_grid: np.ndarray,
        degrees: Iterable[int],
        plot=False,
        reconvert_grid=True,
        num_collapses: int = 0,
    ):
        super().__init__()
        self.filtration_grid = filtration_grid
        self.degrees = degrees
        self.plot = plot
        self.reconvert_grid = reconvert_grid
        self.num_collapses = num_collapses
        return

    def fit(self, X, y=None):
        """
        TODO : infer grid from multiple simplextrees
        """
        return self

    def transform(self, X: Iterable[mp.simplex_tree_multi.SimplexTreeMulti_type]):
        rectangle_decompositions = [
            [
                _st2ranktensor(
                    simplextree,
                    filtration_grid=self.filtration_grid,
                    degree=degree,
                    plot=self.plot,
                    reconvert_grid=self.reconvert_grid,
                    num_collapse=self.num_collapses,
                )
                for degree in self.degrees
            ]
            for simplextree in X
        ]
        # TODO : return iterator ?
        return rectangle_decompositions


def _st2ranktensor(
    st: mp.simplex_tree_multi.SimplexTreeMulti_type,
    filtration_grid: np.ndarray,
    degree: int,
    plot: bool,
    reconvert_grid: bool,
    num_collapse: int | str = 0,
):
    """
    TODO
    """
    # Copy (the squeeze change the filtration values)
    # stcpy = mp.SimplexTreeMulti(st)
    # turns the simplextree into a coordinate simplex tree
    stcpy = st.grid_squeeze(filtration_grid=filtration_grid, coordinate_values=True)
    # stcpy.collapse_edges(num=100, strong = True, ignore_warning=True)
    if num_collapse == "full":
        stcpy.collapse_edges(full=True, ignore_warning=True, max_dimension=degree + 1)
    elif isinstance(num_collapse, int):
        stcpy.collapse_edges(
            num=num_collapse, ignore_warning=True, max_dimension=degree + 1
        )
    else:
        raise TypeError(
            f"Invalid num_collapse=\
            {num_collapse} type. Either full, or an integer."
        )
    # computes the rank invariant tensor
    rank_tensor = mp.rank_invariant2d(
        stcpy, degree=degree, grid_shape=[len(f) for f in filtration_grid]
    )
    # refactor this tensor into the rectangle decomposition of the signed betti
    grid_conversion = filtration_grid if reconvert_grid else None
    rank_decomposition = rank_decomposition_by_rectangles(
        rank_tensor,
        threshold=True,
    )
    rectangle_decomposition = tensor_möbius_inversion(
        tensor=rank_decomposition,
        grid_conversion=grid_conversion,
        plot=plot,
        num_parameters=st.num_parameters,
    )
    return rectangle_decomposition


class DegreeRips2SignedMeasure(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        degrees: Iterable[int],
        min_rips_value: float,
        max_rips_value,
        max_normalized_degree: float,
        min_normalized_degree: float,
        grid_granularity: int,
        progress: bool = False,
        n_jobs=1,
        sparse: bool = False,
        _möbius_inversion=True,
        fit_fraction=1,
    ) -> None:
        super().__init__()
        self.min_rips_value = min_rips_value
        self.max_rips_value = max_rips_value
        self.min_normalized_degree = min_normalized_degree
        self.max_normalized_degree = max_normalized_degree
        self._max_rips_value = None
        self.grid_granularity = grid_granularity
        self.progress = progress
        self.n_jobs = n_jobs
        self.degrees = degrees
        self.sparse = sparse
        self._möbius_inversion = _möbius_inversion
        self.fit_fraction = fit_fraction
        return

    def fit(self, X: np.ndarray | list, y=None):
        if self.max_rips_value < 0:
            print("Estimating scale...", flush=True, end="")
            indices = np.random.choice(
                len(X), min(len(X), int(self.fit_fraction * len(X)) + 1), replace=False
            )
            diameters = np.max(
                [distance_matrix(x, x).max() for x in (X[i] for i in indices)]
            )
            print(f"Done. {diameters}", flush=True)
        self._max_rips_value = (
            -self.max_rips_value * diameters
            if self.max_rips_value < 0
            else self.max_rips_value
        )
        return self

    def _transform1(self, data: np.ndarray):
        _distance_matrix = distance_matrix(data, data)
        signed_measures = []
        (
            rips_values,
            normalized_degree_values,
            hilbert_functions,
            minimal_presentations,
        ) = hf_degree_rips(
            _distance_matrix,
            min_rips_value=self.min_rips_value,
            max_rips_value=self._max_rips_value,
            min_normalized_degree=self.min_normalized_degree,
            max_normalized_degree=self.max_normalized_degree,
            grid_granularity=self.grid_granularity,
            max_homological_dimension=np.max(self.degrees),
        )
        for degree in self.degrees:
            hilbert_function = hilbert_functions[degree]
            signed_measure = (
                signed_betti(hilbert_function, threshold=True)
                if self._möbius_inversion
                else hilbert_function
            )
            if self.sparse:
                signed_measure = tensor_möbius_inversion(
                    tensor=signed_measure,
                    num_parameters=2,
                    grid_conversion=[rips_values, normalized_degree_values],
                )
            if not self._möbius_inversion:
                signed_measure = signed_measure.flatten()
            signed_measures.append(signed_measure)
        return signed_measures

    def transform(self, X):
        return Parallel(n_jobs=self.n_jobs)(
            delayed(self._transform1)(data)
            for data in tqdm(X, desc=f"Computing DegreeRips, of degrees {self.degrees}")
        )


def tensor_möbius_inversion(
    tensor,
    grid_conversion: Iterable[np.ndarray] | None = None,
    plot: bool = False,
    raw: bool = False,
    num_parameters: int | None = None,
):
    from torch import Tensor

    betti_sparse = Tensor(tensor.copy()).to_sparse()  # Copy necessary in some cases :(
    num_indices, num_pts = betti_sparse.indices().shape
    num_parameters = num_indices if num_parameters is None else num_parameters
    if num_indices == num_parameters:  # either hilbert or rank invariant
        rank_invariant = False
    elif 2 * num_parameters == num_indices:
        rank_invariant = True
    else:
        raise TypeError(
            f"Unsupported betti shape. {num_indices}\
            has to be either {num_parameters} or \
            {2*num_parameters}."
        )
    points_filtration = np.asarray(betti_sparse.indices().T, dtype=int)
    weights = np.asarray(betti_sparse.values(), dtype=int)

    if grid_conversion is not None:
        coords = np.empty(shape=(num_pts, num_indices), dtype=float)
        for i in range(num_indices):
            coords[:, i] = grid_conversion[i % num_parameters][points_filtration[:, i]]
    else:
        coords = points_filtration
    if (not rank_invariant) and plot:
        plt.figure()
        color_weights = np.empty(weights.shape)
        color_weights[weights > 0] = np.log10(weights[weights > 0]) + 2
        color_weights[weights < 0] = -np.log10(-weights[weights < 0]) - 2
        plt.scatter(
            points_filtration[:, 0],
            points_filtration[:, 1],
            c=color_weights,
            cmap="coolwarm",
        )
    if (not rank_invariant) or raw:
        return coords, weights

    def _is_trivial(rectangle: np.ndarray):
        birth = rectangle[:num_parameters]
        death = rectangle[num_parameters:]
        return np.all(birth <= death)  # and not np.array_equal(birth,death)

    correct_indices = np.array([_is_trivial(rectangle) for rectangle in coords])
    if len(correct_indices) == 0:
        return np.empty((0, num_indices)), np.empty((0))
    signed_measure = np.asarray(coords[correct_indices])
    weights = weights[correct_indices]
    if plot:
        # plot only the rank decompo for the moment
        assert signed_measure.shape[1] == 4

        def _plot_rectangle(rectangle: np.ndarray, weight: float):
            x_axis = rectangle[[0, 2]]
            y_axis = rectangle[[1, 3]]
            color = "blue" if weight > 0 else "red"
            plt.plot(x_axis, y_axis, c=color)

        for rectangle, weight in zip(signed_measure, weights):
            _plot_rectangle(rectangle=rectangle, weight=weight)
    return signed_measure, weights
