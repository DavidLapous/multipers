from typing import Callable, Iterable, List, Optional

import numpy as np
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, TransformerMixin
from tqdm import tqdm

import multipers as mp
import multipers.simplex_tree_multi
from multipers.grids import compute_grid as reduce_grid
from multipers.ml.tools import filtration_grid_to_coordinates
from multipers.mma_structures import PyBox_f64, PyModule_type


class SimplexTree2MMA(BaseEstimator, TransformerMixin):
    """
    Turns a list of simplextrees to MMA approximations
    """

    def __init__(
        self,
        n_jobs: int = 1,
        expand_dim: Optional[int] = None,
        prune_degrees_above: Optional[int] = None,
        progress=False,
        **persistence_kwargs,
    ) -> None:
        super().__init__()
        self.persistence_args = persistence_kwargs
        self.n_jobs = n_jobs
        self._has_axis = None
        self._num_axis = None
        self.prune_degrees_above = prune_degrees_above
        self.progress = progress
        self.expand_dim = expand_dim
        self._boxes = None
        return

    def fit(self, X, y=None):
        if len(X) == 0:
            return self
        self._has_axis = not mp.simplex_tree_multi.is_simplextree_multi(X[0])
        if self._has_axis:
            try:
                X[0][0]
            except IndexError:
                print(f"IndexError, {X[0]=}")
                if len(X[0]) == 0:
                    print(
                        "No simplextree found, maybe you forgot to give a filtration parameter to the previous pipeline"
                    )
                raise IndexError
            assert mp.simplex_tree_multi.is_simplextree_multi(X[0][0]), f"X[0] is not a simplextre, {X[0]=}, and X[0][0] neither."
            self._num_axis = len(X[0])
            filtration_values = np.asarray(
                [
                    [x[axis].filtration_bounds() for x in X]
                    for axis in range(self._num_axis)
                ]
            )
            num_parameters = filtration_values.shape[-1]
            # Output : axis, data, min/max, num_parameters
            # print("TEST : NUM PARAMETERS ", num_parameters)
            m = np.asarray(
                [
                    [
                        filtration_values[axis, :, 0, parameter].min()
                        for parameter in range(num_parameters)
                    ]
                    for axis in range(self._num_axis)
                ]
            )
            M = np.asarray(
                [
                    [
                        filtration_values[axis, :, 1, parameter].max()
                        for parameter in range(num_parameters)
                    ]
                    for axis in range(self._num_axis)
                ]
            )
            # shape of m/M axis,num_parameters
            self._boxes = [
                np.array([m_of_axis, M_of_axis]) for m_of_axis, M_of_axis in zip(m, M)
            ]
        else:
            filtration_values = np.asarray([x.filtration_bounds() for x in X])
            num_parameters = filtration_values.shape[-1]
            # print("TEST : NUM PARAMETERS ", num_parameters)
            m = np.asarray(
                [
                    filtration_values[:, 0, parameter].min()
                    for parameter in range(num_parameters)
                ]
            )
            M = np.asarray(
                [
                    filtration_values[:, 1, parameter].max()
                    for parameter in range(num_parameters)
                ]
            )
            self._boxes = [m, M]
        return self

    def transform(self, X):
        if self.prune_degrees_above is not None:
            for x in X:
                if self._has_axis:
                    for x_ in x:
                        x_.prune_above_dimension(
                            self.prune_degrees_above
                        )  # we only do for H0 for computational ease
                else:
                    x.prune_above_dimension(
                        self.prune_degrees_above
                    )  # we only do for H0 for computational ease

        def todo1(x: mp.simplex_tree_multi.SimplexTreeMulti_type, box):
            # print(x.get_filtration_grid(resolution=3, grid_strategy="regular"))
            # print("TEST BOX",box)
            if self.expand_dim is not None:
                x.expansion(self.expand_dim)
            return x.persistence_approximation(
                box=box, verbose=False, **self.persistence_args
            )

        def todo(sts: List[mp.simplex_tree_multi.SimplexTreeMulti_type] | mp.simplex_tree_multi.SimplexTreeMulti_type):
            if self._has_axis:
                assert not mp.simplex_tree_multi.is_simplextree_multi(sts)
                return [todo1(st, box) for st, box in zip(sts, self._boxes)]
            assert mp.simplex_tree_multi.is_simplextree_multi(sts)
            return todo1(sts, self._boxes)

        return Parallel(n_jobs=self.n_jobs, backend="threading")(
            delayed(todo)(x)
            for x in tqdm(X, desc="Computing modules", disable=not self.progress)
        )


class MMAFormatter(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        degrees: Optional[list[int]] = None,
        axis=None,
        verbose: bool = False,
        normalize: bool = False,
        weights=None,
        quantiles=None,
        dump=False,
        from_dump=False,
    ):
        self._module_bounds = None
        self.verbose = verbose
        self.axis = axis
        self._axis = []
        self._has_axis = None
        self._num_axis = 0
        self.degrees = degrees
        self._degrees = None
        self.normalize = normalize
        self._num_parameters = None
        self.weights = weights
        self.quantiles = quantiles
        self.dump = dump
        self.from_dump = from_dump

    @staticmethod
    def _maybe_from_dump(X_in):
        if len(X_in) == 0:
            return X_in
        import pickle

        if isinstance(X_in[0], bytes):
            X = [pickle.loads(mods) for mods in X_in]
        else:
            X = X_in
        return X
        # return [[mp.multiparameter_module_approximation.from_dump(mod) for mod in mods] for mods in dumped_modules]

    @staticmethod
    def _get_module_bound(x, degree):
        """
        Output format : (2,num_parameters)
        """
        # l,L = x.get_box()
        filtration_values = x.get_module_of_degree(degree).get_filtration_values(
            unique=True
        )
        out = np.array([[f[0], f[-1]] for f in filtration_values if len(f) > 0]).T
        if len(out) != 2:
            print(f"Missing degree {degree} here !")
            m = M = [np.nan for _ in range(x.num_parameters)]
        else:
            m, M = out
        # m = np.where(m<np.inf, m, l)
        # M = np.where(M>-np.inf, M,L)
        return m, M

    @staticmethod
    def _infer_axis(X):
        has_axis = not isinstance(X[0], PyModule_type)
        assert not has_axis or isinstance(X[0][0], PyModule_type)
        return has_axis

    @staticmethod
    def _infer_num_parameters(X, ax=slice(None)):
        return X[0][ax].num_parameters

    @staticmethod
    def _infer_bounds(X, degrees=None, axis=[slice(None)], quantiles=None):
        """
        Compute bounds of filtration values of a list of modules.

        Output Format
        -------------
        m,M of shape : (num_axis,num_degrees,2,num_parameters)
        """
        if degrees is None:
            degrees = np.arange(X[0][axis[0]].max_degree + 1)
        bounds = np.array(
            [
                [
                    [
                        MMAFormatter._get_module_bound(x[ax], degree)
                        for degree in degrees
                    ]
                    for ax in axis
                ]
                for x in X
            ]
        )
        if quantiles is not None:
            qm, qM = quantiles
            # TODO per axis, degree !!
            # m = np.quantile(bounds[:,:,:,0,:], q=qm,axis=0)
            # M = np.quantile(bounds[:,:,:,1,:], q=1-qM,axis=0)
            num_pts, num_axis, num_degrees, _, num_parameters = bounds.shape
            m = [
                [
                    [
                        np.nanquantile(
                            bounds[:, ax, degree, 0, parameter], axis=0, q=qm
                        )
                        for parameter in range(num_parameters)
                    ]
                    for degree in range(num_degrees)
                ]
                for ax in range(num_axis)
            ]
            m = np.asarray(m)
            M = [
                [
                    [
                        np.nanquantile(
                            bounds[:, ax, degree, 1, parameter], axis=0, q=1 - qM
                        )
                        for parameter in range(num_parameters)
                    ]
                    for degree in range(num_degrees)
                ]
                for ax in range(num_axis)
            ]
            M = np.asarray(M)
        else:
            num_pts, num_axis, num_degrees, _, num_parameters = bounds.shape
            m = [
                [
                    [
                        np.nanmin(bounds[:, ax, degree, 0, parameter], axis=0)
                        for parameter in range(num_parameters)
                    ]
                    for degree in range(num_degrees)
                ]
                for ax in range(num_axis)
            ]
            m = np.asarray(m)
            M = [
                [
                    [
                        np.nanmax(bounds[:, ax, degree, 1, parameter], axis=0)
                        for parameter in range(num_parameters)
                    ]
                    for degree in range(num_degrees)
                ]
                for ax in range(num_axis)
            ]
            M = np.asarray(M)
            # m = bounds[:,:,:,0,:].min(axis=0)
            # M = bounds[:,:,:,1,:].max(axis=0)
        return (m, M)

    @staticmethod
    def _infer_grid(X: List[PyModule_type], strategy: str, resolution: int, degrees=None):
        """
        Given a list of PyModules, computes a multiparameter discrete grid,
        with a given strategy,
        from the filtration values of the summands of the modules.
        """
        num_parameters = X[0].num_parameters
        if degrees is None:
            # Format here : ((filtration values of parameter) for parameter)
            filtration_values = tuple(
                mod.get_filtration_values(unique=True) for mod in X
            )
        else:
            filtration_values = tuple(
                mod.get_module_of_degrees(degrees).get_filtration_values(unique=True)
                for mod in X
            )

        if "_mean" in strategy:
            substrategy = strategy.split("_")[0]
            processed_filtration_values = [
                reduce_grid(f, resolution, substrategy, unique=False)
                for f in filtration_values
            ]
            reduced_grid = np.mean(processed_filtration_values, axis=0)
        # elif "_quantile" in strategy:
        # substrategy = strategy.split("_")[0]
        # processed_filtration_values = [reduce_grid(f, resolution, substrategy, unique=False) for f in filtration_values]
        # reduced_grid = np.qu(processed_filtration_values, axis=0)
        else:
            filtration_values = [
                np.unique(
                    np.concatenate([f[parameter] for f in filtration_values], axis=0)
                )
                for parameter in range(num_parameters)
            ]
            reduced_grid = reduce_grid(
                filtration_values, resolution, strategy, unique=True
            )

        coordinates, new_resolution = filtration_grid_to_coordinates(
            reduced_grid, return_resolution=True
        )
        return coordinates, new_resolution

    def _infer_degrees(self, X):
        if self.degrees is None:
            max_degrees = [x[ax].max_degree
                        for i, ax in enumerate(self._axis)
                        for x in X
                ] + [0]
            self._degrees = np.arange(np.max(max_degrees) + 1)
        else:
            self._degrees = self.degrees
        
    def fit(self, X_in, y=None):
        X = self._maybe_from_dump(X_in)
        if len(X) == 0:
            return self
        self._has_axis = self._infer_axis(X)
        # assert not self._has_axis or isinstance(X[0][0], mp.PyModule)
        if self.axis is None and self._has_axis:
            self.axis = -1
        if self.axis is not None and not (self._has_axis):
            raise Exception(f"SMF didn't find an axis, but requested axis {self.axis}")
        if self._has_axis:
            self._num_axis = len(X[0])
        if self.verbose:
            print("-----------MMAFormatter-----------")
            print("---- Infered stats")
            print(f"Found axis : {self._has_axis}, num : {self._num_axis}")
            print(f"Number of parameters : {self._num_parameters}")
        self._axis = (
            [slice(None)]
            if self.axis is None
            else range(self._num_axis)
            if self.axis == -1
            else [self.axis]
        )
        self._infer_degrees(X)

        self._num_parameters = self._infer_num_parameters(X, ax=self._axis[0])
        if self.normalize:
            # print(self._axis)
            self._module_bounds = self._infer_bounds(
                X, self._degrees, self._axis, self.quantiles
            )
        else:
            m = np.zeros((self._num_axis, len(self._degrees), self._num_parameters))
            M = m + 1
            self._module_bounds = (m, M)
        assert self._num_parameters == self._module_bounds[0].shape[-1]
        if self.verbose:
            print("---- Bounds (only computed if normalize):")
            if self._has_axis and self._num_axis > 1:
                print("(axis) x (degree) x (parameter)")
            else:
                print("(degree) x (parameter)")
            m, M = self._module_bounds
            print("-- Lower bound : ", m.shape)
            print(m)
            print("-- Upper bound :", M.shape)
            print(M)
        w = 1 if self.weights is None else np.asarray(self.weights)
        m, M = self._module_bounds
        normalizer = M - m
        zero_normalizer = normalizer == 0
        if np.any(zero_normalizer):
            from warnings import warn

            warn(f"Encountered empty bounds. Please fix me. \n M-m = {normalizer}")
        normalizer[zero_normalizer] = 1
        self._normalization_factors = w / normalizer
        if self.verbose:
            print("-- Normalization factors:", self._normalization_factors.shape)
            print(self._normalization_factors)

        if self.verbose:
            print("---- Module size :")
            for ax in self._axis:
                print(f"- Axis {ax}")
                for degree in self._degrees:
                    sizes = [len(x[ax].get_module_of_degree(degree)) for x in X]
                    print(
                        f" - Degree {degree} size \
                        {np.mean(sizes).round(decimals=2)}\
                        ±{np.std(sizes).round(decimals=2)}"
                    )
            print("----------------------------------")
        return self

    @staticmethod
    def copy_transform(mod, degrees, translation, rescale_factors, new_box):
        copy = mod.get_module_of_degrees(
            degrees
        )  # and only returns the specific degrees
        for j, degree in enumerate(degrees):
            copy.translate(translation[j], degree=degree)
            copy.rescale(rescale_factors[j], degree=degree)
        copy.set_box(new_box)
        return copy

    def transform(self, X_in):
        X = self._maybe_from_dump(X_in)
        if np.any(self._normalization_factors != 1):
            if self.verbose:
                print("Normalizing...", end="")
            w = (
                [1] * self._num_parameters
                if self.weights is None
                else np.asarray(self.weights)
            )
            standard_box = PyBox_f64([0] * self._num_parameters, w)

            X_copy = [
                [
                    self.copy_transform(
                        mod=x[ax],
                        degrees=self._degrees,
                        translation=-self._module_bounds[0][i],
                        rescale_factors=self._normalization_factors[i],
                        new_box=standard_box,
                    )
                    for i, ax in enumerate(self._axis)
                ]
                for x in X
            ]
            if self.verbose:
                print("Done.")
            return X_copy
        if self.axis != -1:
            X = [x[self.axis] for x in X]
        if self.dump:
            import pickle

            X = [pickle.dumps(mods) for mods in X]
        return X
        # return [todo(x) for x in X]


class MMA2IMG(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        degrees: list,
        bandwidth: float = 0.1,
        power: float = 1,
        normalize: bool = False,
        resolution: list | int = 50,
        plot: bool = False,
        box=None,
        n_jobs=-1,
        flatten=False,
        progress=False,
        grid_strategy="regular",
        kernel="linear",
        signed:bool=False,
    ):
        self.bandwidth = bandwidth
        self.degrees = degrees
        self.resolution = resolution
        self.box = box
        self.plot = plot
        self._box = None
        self.normalize = normalize
        self.power = power
        self._has_axis = None
        self._num_parameters = None
        self.n_jobs = n_jobs
        self.flatten = flatten
        self.progress = progress
        self.grid_strategy = grid_strategy
        self._num_axis = None
        self._coords_to_compute = None
        self._new_resolutions = None
        self.kernel=kernel
        self.signed = signed

    def fit(self, X, y=None):
        # TODO infer box
        # TODO rescale module
        self._has_axis = MMAFormatter._infer_axis(X)
        if self._has_axis:
            self._num_axis = len(X[0])
        if self.box is None:
            self._box = [[0,0], [1, 1]]
        else:
            self._box = self.box
        if self._has_axis:
            its = (tuple(x[axis] for x in X) for axis in range(self._num_axis))
            crs = tuple(
                MMAFormatter._infer_grid(
                    X_axis, self.grid_strategy, self.resolution, degrees=self.degrees
                )
                for X_axis in its
            )
            self._coords_to_compute = [
                c for c, _ in crs
            ]  # not the same resolutions, so cannot be put in an array
            self._new_resolutions = np.asarray([r for _, r in crs])
        else:
            coords, new_resolution = MMAFormatter._infer_grid(
                X, self.grid_strategy, self.resolution, degrees=self.degrees
            )
            self._coords_to_compute = coords
            self._new_resolutions = new_resolution
        return self

    def transform(self, X):
        img_args = {
            "bandwidth": self.bandwidth,
            "p": self.power,
            "normalize": self.normalize,
            # "plot":self.plot,
            # "cb":1, # colorbar
            # "resolution" : self.resolution, # info in coordinates
            "box": self.box,
            "degrees": self.degrees,
            # num_jobs is better for parallel over modules.
            "n_jobs": self.n_jobs,
            "kernel":self.kernel,
            "signed":self.signed,
            "flatten":True, # custom coordinates
        }
        if self._has_axis:

            def todo1(x, c):
                return x.representation(coordinates=c, **img_args)
        else:

            def todo1(x):
                return x.representation(coordinates = self._coords_to_compute, **img_args)[
                    None, :
                ]  # shape same as has_axis

        if self._has_axis:
            def todo2(mods):
                return tuple(todo1(mod, c) for mod, c in zip(mods, self._coords_to_compute))
        else:
            todo2 = todo1

        if self.flatten:

            def todo(mods):
                return np.concatenate(todo2(mods), axis=1).flatten()
        else:

            def todo(mods):
                return tuple(
                    img.reshape(len(img_args["degrees"]), *r)
                    for img, r in zip(todo2(mods), self._new_resolutions)
                )

        return Parallel(n_jobs=self.n_jobs, backend="threading")(
            delayed(todo)(x)
            for x in tqdm(X, desc="Computing images", disable=not self.progress)
        )  # res depends on ax (infer_grid)


class MMA2Landscape(BaseEstimator, TransformerMixin):
    """
    Turns a list of MMA approximations into Landscapes vectorisations
    """

    def __init__(
        self,
        resolution=[100, 100],
        degrees: list[int] | None = [0, 1],
        ks: Iterable[int] = range(5),
        phi: Callable = np.sum,
        box=None,
        plot: bool = False,
        n_jobs=-1,
        filtration_quantile: float = 0.01,
    ) -> None:
        super().__init__()
        self.resolution: list[int] = resolution
        self.degrees = degrees
        self.ks = ks
        self.phi = phi  # Has to have a axis=0 !
        self.box = box
        self.plot = plot
        self.n_jobs = n_jobs
        self.filtration_quantile = filtration_quantile
        return

    def fit(self, X, y=None):
        if len(X) <= 0:
            return
        assert (
            X[0].num_parameters == 2
        ), f"Number of parameters {X[0].num_parameters} has to be 2."
        if self.box is None:

            def _bottom(mod):
                return mod.get_bottom()

            def _top(mod):
                return mod.get_top()

            m = np.quantile(
                Parallel(n_jobs=self.n_jobs, backend="threading")(
                    delayed(_bottom)(mod) for mod in X
                ),
                q=self.filtration_quantile,
                axis=0,
            )
            M = np.quantile(
                Parallel(n_jobs=self.n_jobs, backend="threading")(
                    delayed(_top)(mod) for mod in X
                ),
                q=1 - self.filtration_quantile,
                axis=0,
            )
            self.box = [m, M]
        return self

    def transform(self, X) -> list[np.ndarray]:
        if len(X) <= 0:
            return

        def todo(mod):
            return np.concatenate(
                [
                    self.phi(
                        mod.landscapes(
                            ks=self.ks,
                            resolution=self.resolution,
                            degree=degree,
                            plot=self.plot,
                        ),
                        axis=0,
                    ).flatten()
                    for degree in self.degrees
                ]
            ).flatten()

        return Parallel(n_jobs=self.n_jobs, backend="threading")(
            delayed(todo)(x) for x in X
        )
