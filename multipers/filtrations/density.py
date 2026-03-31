from collections.abc import Callable, Iterable
from typing import Any, Literal, Union

import numpy as np
import multipers.array_api.numpy as _npapi
from multipers.array_api import api_from_tensor

global available_kernels
available_kernels = Union[
    Literal[
        "gaussian", "exponential", "exponential_kernel", "multivariate_gaussian", "sinc"
    ],
    Callable,
]


def convolution_signed_measures(
    iterable_of_signed_measures,
    filtrations,
    bandwidth,
    flatten: bool = True,
    n_jobs: int = 1,
    backend="pykeops",
    kernel: available_kernels = "gaussian",
    **kwargs,
):
    """
    Evaluates the convolution of the signed measures Iterable(pts, weights) with a gaussian measure of bandwidth bandwidth, on a grid given by the filtrations

    Parameters
    ----------

     - iterable_of_signed_measures : (num_signed_measure) x [ (npts,num_parameters), (npts)]
     - filtrations : (num_parameter) x (filtration values)
     - flatten : bool
     - n_jobs : int

    Outputs
    -------

    The concatenated images, for each signed measure (num_signed_measures) x (len(f) for f in filtration_values)
    """
    from multipers.grids import todense

    grid_iterator = todense(filtrations)
    api = api_from_tensor(iterable_of_signed_measures[0][0][0])
    match backend:
        case "sklearn":
            if api is not _npapi:
                raise ValueError(
                    f"The sklearn backend only supports numpy. Got {api=}."
                )

            def convolution_signed_measures_on_grid(
                signed_measures,
            ):
                return api.cat(
                    [
                        _pts_convolution_sparse_old(
                            pts=pts,
                            pts_weights=weights,
                            grid_iterator=grid_iterator,
                            bandwidth=bandwidth,
                            kernel=kernel,
                            **kwargs,
                        )
                        for pts, weights in signed_measures
                    ],
                    axis=0,
                )

        case "pykeops":

            def convolution_signed_measures_on_grid(
                signed_measures: Iterable[tuple[np.ndarray, np.ndarray]],
            ) -> np.ndarray:
                return api.cat(
                    [
                        _pts_convolution_pykeops(
                            pts=pts,
                            pts_weights=weights,
                            grid_iterator=grid_iterator,
                            bandwidth=bandwidth,
                            kernel=kernel,
                            api=api,
                            **kwargs,
                        )
                        for pts, weights in signed_measures
                    ],
                    axis=0,
                )

            # compiles first once
            pts, weights = iterable_of_signed_measures[0][0]
            small_pts, small_weights = pts[:2], weights[:2]

            _pts_convolution_pykeops(
                small_pts,
                small_weights,
                grid_iterator=grid_iterator,
                bandwidth=bandwidth,
                kernel=kernel,
                api=api,
                **kwargs,
            )

        case "dense" | "jax":
            if backend == "jax" and not _is_jax_api(api):
                raise ValueError("The jax backend requires JAX arrays.")

            def convolution_signed_measures_on_grid(
                signed_measures: Iterable[tuple[np.ndarray, np.ndarray]],
            ) -> np.ndarray:
                return api.cat(
                    [
                        _pts_convolution_dense(
                            pts=pts,
                            pts_weights=weights,
                            grid_iterator=grid_iterator,
                            bandwidth=bandwidth,
                            kernel=kernel,
                            api=api,
                            **kwargs,
                        )
                        for pts, weights in signed_measures
                    ],
                    axis=0,
                )

            pts, weights = iterable_of_signed_measures[0][0]
            small_pts, small_weights = pts[:2], weights[:2]
            _pts_convolution_dense(
                small_pts,
                small_weights,
                grid_iterator=grid_iterator,
                bandwidth=bandwidth,
                kernel=kernel,
                api=api,
                **kwargs,
            )

        case _:
            raise ValueError(f"Unknown convolution backend {backend}.")

    if n_jobs > 1 or n_jobs == -1:
        prefer = "processes" if backend == "sklearn" else "threads"
        from joblib import Parallel, delayed

        convolutions = Parallel(n_jobs=n_jobs, prefer=prefer)(
            delayed(convolution_signed_measures_on_grid)(sms)
            for sms in iterable_of_signed_measures
        )
    else:
        convolutions = [
            convolution_signed_measures_on_grid(sms)
            for sms in iterable_of_signed_measures
        ]
    if not flatten:
        out_shape = [-1] + [len(f) for f in filtrations]  # Degree
        convolutions = [x.reshape(out_shape) for x in convolutions]
    return api.cat([x[None] for x in convolutions])


# def _test(r=1000, b=0.5, plot=True, kernel=0):
# 	import matplotlib.pyplot  as plt
# 	pts, weigths = np.array([[1.,1.], [1.1,1.1]]), np.array([1,-1])
# 	pt_list = np.array(list(product(*[np.linspace(0,2,r)]*2)))
# 	img = _pts_convolution_sparse_pts(pts,weigths, pt_list,b,kernel=kernel)
# 	if plot:
# 		plt.imshow(img.reshape(r,-1).T, origin="lower")
# 		plt.show()


def _pts_convolution_sparse_old(
    pts: np.ndarray,
    pts_weights: np.ndarray,
    grid_iterator,
    kernel: available_kernels = "gaussian",
    bandwidth=0.1,
    **more_kde_args,
):
    """
    Old version of `convolution_signed_measures`. Scikitlearn's convolution is slower than the code above.
    """
    from sklearn.neighbors import KernelDensity

    if len(pts) == 0:
        # warn("Found a trivial signed measure !")
        return np.zeros(len(grid_iterator))
    if kernel == "multivariate_gaussian":
        kernel = "gaussian"
    if kernel == "sinc":
        raise ValueError("Sinc kernel is not supported by sklearn backend.")
    kde = KernelDensity(
        kernel=kernel, bandwidth=bandwidth, **more_kde_args
    )  # TODO : check rtol
    pos_indices = pts_weights > 0
    neg_indices = pts_weights < 0
    normalizer = len(pts)
    img_pos = (
        np.zeros(len(grid_iterator))
        if pos_indices.sum() == 0
        else kde.fit(
            pts[pos_indices], sample_weight=pts_weights[pos_indices]
        ).score_samples(grid_iterator)
    )
    img_neg = (
        np.zeros(len(grid_iterator))
        if neg_indices.sum() == 0
        else kde.fit(
            pts[neg_indices], sample_weight=-pts_weights[neg_indices]
        ).score_samples(grid_iterator)
    )
    pos_scale = (
        pts_weights[pos_indices].sum() / normalizer if pos_indices.any() else 0.0
    )
    neg_scale = (
        (-pts_weights[neg_indices]).sum() / normalizer if neg_indices.any() else 0.0
    )
    return np.exp(img_pos) * pos_scale - np.exp(img_neg) * neg_scale


def _pts_convolution_pykeops(
    pts,
    pts_weights,
    grid_iterator,
    kernel: available_kernels = "gaussian",
    bandwidth=0.1,
    api=None,
    **more_kde_args,
):
    """
    KDE-based convolution. Uses PyKeOps when available for the input backend.
    """
    if api is None:
        api = api_from_tensor(pts)
    pts = api.astensor(pts)
    dtype = pts.dtype
    kde = KDE(kernel=kernel, bandwidth=bandwidth, **more_kde_args)
    return kde.fit(
        pts, sample_weights=api.astype(pts_weights, dtype), api=api
    ).score_samples(api.astype(grid_iterator, dtype))


def _pts_convolution_dense(
    pts,
    pts_weights,
    grid_iterator,
    kernel: available_kernels = "gaussian",
    bandwidth=0.1,
    api=None,
    **more_kde_args,
):
    """
    Exact dense convolution using the current array backend.
    """
    if api is None:
        api = api_from_tensor(pts)
    pts = api.astensor(pts)
    dtype = pts.dtype
    kde = KDE(
        kernel=kernel,
        bandwidth=bandwidth,
        implementation="dense",
        **more_kde_args,
    )
    return kde.fit(
        pts, sample_weights=api.astype(pts_weights, dtype), api=api
    ).score_samples(api.astype(grid_iterator, dtype))


def gaussian_kernel(x_i, y_j, bandwidth, api=_npapi, **kwargs):
    D = x_i.shape[-1]
    bandwidth = api.astensor(bandwidth).reshape(1)
    exponent = -(((x_i - y_j) / bandwidth) ** 2).sum(dim=-1) / 2
    # float is necessary for some reason (pykeops fails)
    kernel = (exponent).exp() / api.astensor((bandwidth * (2 * np.pi) ** 0.5) ** D)
    return kernel


def multivariate_gaussian_kernel(x_i, y_j, covariance_matrix_inverse, api=_npapi):
    # 1 / \sqrt(2 \pi^dim * \Sigma.det()) * exp( -(x-y).T @ \Sigma ^{-1} @ (x-y))
    # CF https://www.kernel-operations.io/keops/_auto_examples/pytorch/plot_anisotropic_kernels.html#sphx-glr-auto-examples-pytorch-plot-anisotropic-kernels-py
    #    and https://www.kernel-operations.io/keops/api/math-operations.html
    dim = x_i.shape[-1]
    z = x_i - y_j
    exponent = -(z.weightedsqnorm(covariance_matrix_inverse.flatten()) / 2)
    return (
        api.astensor((2 * np.pi) ** (-dim / 2), dtype=x_i.dtype).reshape(1)
        * (api.sqrt(api.det(covariance_matrix_inverse)))
        * exponent.exp()
    )


def exponential_kernel(x_i, y_j, bandwidth, api=_npapi, **kwargs):
    # 1 / \sigma * exp( norm(x-y, dim=-1))
    bandwidth = api.astensor(bandwidth).reshape(1)
    exponent = -((((x_i - y_j) ** 2).sum(dim=-1).sqrt()) / bandwidth)
    kernel = exponent.exp() / bandwidth
    return kernel


def sinc_kernel(x_i, y_j, bandwidth, api=_npapi):
    bandwidth = api.astensor(bandwidth).reshape(1)
    norm = (((x_i - y_j) ** 2).sum(dim=-1).sqrt()) / bandwidth
    sinc = type(x_i).sinc
    kernel = 2 * sinc(2 * norm) - sinc(norm)
    return kernel


def _is_jax_api(api) -> bool:
    return getattr(api, "__name__", "") == "multipers.array_api.jax"


def _dense_kernel(
    X,
    Y,
    *,
    kernel: available_kernels,
    bandwidth,
    api,
    **kwargs,
):
    if callable(kernel):
        raise NotImplementedError(
            "Custom kernels are only supported with the PyKeOps implementation."
        )

    X = api.astensor(X)
    Y = api.astype(Y, X.dtype)
    dim = X.shape[-1]

    match kernel:
        case "gaussian":
            pairwise_distances = api.cdist(X, Y)
            bandwidth = api.astensor(bandwidth, dtype=X.dtype)
            exponent = -((pairwise_distances / bandwidth) ** 2) / 2
            normalization = api.astensor(
                (bandwidth * (2 * np.pi) ** 0.5) ** dim, dtype=X.dtype
            )
            return api.exp(exponent) / normalization
        case "exponential" | "exponential_kernel":
            pairwise_distances = api.cdist(X, Y)
            bandwidth = api.astensor(bandwidth, dtype=X.dtype)
            return api.exp(-(pairwise_distances / bandwidth)) / bandwidth
        case "multivariate_gaussian":
            covariance_matrix_inverse = api.astype(bandwidth, X.dtype)
            z = X[:, None, :] - Y[None, :, :]
            quad_form = api.einsum("...i,ij,...j->...", z, covariance_matrix_inverse, z)
            normalization = api.astensor(
                (2 * np.pi) ** (-dim / 2), dtype=X.dtype
            ) * api.sqrt(api.det(covariance_matrix_inverse))
            return normalization * api.exp(-(quad_form / 2))
        case "sinc":
            pairwise_distances = api.cdist(X, Y)
            bandwidth = api.astensor(bandwidth, dtype=X.dtype)
            norm = pairwise_distances / bandwidth
            sinc = api.sinc
            return 2 * sinc(2 * norm) - sinc(norm)
        case _:
            raise ValueError(f"Unknown kernel {kernel}.")


def _kernel(
    kernel: available_kernels = "gaussian",
):
    match kernel:
        case "gaussian":
            return gaussian_kernel
        case "exponential" | "exponential_kernel":
            return exponential_kernel
        case "multivariate_gaussian":
            return multivariate_gaussian_kernel
        case "sinc":
            return sinc_kernel
        case _:
            assert callable(kernel), f"""
            --------------------------
            Unknown kernel {kernel}.
            -------------------------- 
            Custom kernel has to be callable, 
            (x:LazyTensor(n,1,D),y:LazyTensor(1,m,D),bandwidth:float) ---> kernel matrix

            Valid operations are given here:
            https://www.kernel-operations.io/keops/python/api/index.html
            """
            return kernel


# TODO : multiple bandwidths at once with lazy tensors
class KDE:
    """
    Fast, scikit-style, differentiable kernel density estimation.

    Uses PyKeOps when available for NumPy and Torch inputs, and falls back to an
    exact dense implementation otherwise.
    """

    def __init__(
        self,
        bandwidth: Any = 1,
        kernel: available_kernels = "gaussian",
        return_log: bool = False,
        implementation: Literal["auto", "pykeops", "dense"] = "auto",
        chunk_size: int = 2048,
        **kwargs,
    ):
        """
        bandwidth : numeric
                bandwidth for Gaussian kernel
        """
        self.X = None
        self.bandwidth = bandwidth
        self.kernel: available_kernels = kernel
        self._kernel = None
        self.api = None
        self._sample_weights = None
        self.return_log = return_log
        self.implementation = implementation
        self.chunk_size = chunk_size
        self._implementation = None
        self._dense_score_samples = None
        self.kwargs = kwargs

    def fit(self, X, sample_weights=None, y=None, api=None):
        self.api = api_from_tensor(X) if api is None else api
        self.X = self.api.astensor(X)
        self._sample_weights = sample_weights
        self._kernel = _kernel(self.kernel)
        self._implementation = self._resolve_implementation()
        self._dense_score_samples = None
        return self

    def _resolve_implementation(self) -> Literal["pykeops", "dense"]:
        if self.implementation == "auto":
            if getattr(self.api, "check_keops", lambda: False)():
                return "pykeops"
            return "dense"
        if self.implementation == "pykeops":
            if not getattr(self.api, "check_keops", lambda: False)():
                raise ValueError("PyKeOps is unavailable for this backend.")
            return "pykeops"
        return "dense"

    def _make_dense_score_samples(self):
        assert self.api is not None and self.X is not None
        X = self.X
        weights = None
        if self._sample_weights is not None:
            weights = self.api.astype(self._sample_weights, X.dtype)

        def score_chunk(X_chunk, Y_chunk):
            kernel_values = _dense_kernel(
                X_chunk,
                Y_chunk,
                kernel=self.kernel,
                bandwidth=self.bandwidth,
                api=self.api,
                **self.kwargs,
            )
            if weights is not None:
                kernel_values = kernel_values * weights[:, None]
            return self.api.sum(kernel_values, axis=0) / kernel_values.shape[0]

        if _is_jax_api(self.api):
            return self.api.jit(score_chunk)
        return score_chunk

    def _score_samples_dense(self, Y, return_kernel=False):
        assert self.api is not None and self.X is not None
        X = self.X
        if X.shape[0] == 0:
            return self.api.zeros((Y.shape[0]), dtype=X.dtype)
        if self._dense_score_samples is None:
            self._dense_score_samples = self._make_dense_score_samples()

        Y = self.api.astype(Y, X.dtype)
        if return_kernel:
            kernel_values = _dense_kernel(
                X,
                Y,
                kernel=self.kernel,
                bandwidth=self.bandwidth,
                api=self.api,
                **self.kwargs,
            )
            if self._sample_weights is not None:
                kernel_values = (
                    kernel_values
                    * self.api.astype(self._sample_weights, X.dtype)[:, None]
                )
            return kernel_values

        outputs = []
        for start in range(0, Y.shape[0], self.chunk_size):
            stop = min(start + self.chunk_size, Y.shape[0])
            outputs.append(self._dense_score_samples(X, Y[start:stop]))
        density_estimation = outputs[0] if len(outputs) == 1 else self.api.cat(outputs)
        return (
            self.api.log(density_estimation) if self.return_log else density_estimation
        )

    @staticmethod
    def to_lazy(X, Y, x_weights):
        if isinstance(X, np.ndarray):
            from pykeops.numpy import LazyTensor

            lazy_x = LazyTensor(
                X.reshape((X.shape[0], 1, X.shape[1]))
            )  # numpts, 1, dim
            lazy_y = LazyTensor(
                Y.reshape((1, Y.shape[0], Y.shape[1])).astype(X.dtype)
            )  # 1, numpts, dim
            if x_weights is not None:
                w = LazyTensor(np.asarray(x_weights, dtype=X.dtype)[:, None], axis=0)
                return lazy_x, lazy_y, w
            return lazy_x, lazy_y, None
        import torch

        if isinstance(X, torch.Tensor):
            from pykeops.torch import LazyTensor

            lazy_x = LazyTensor(X.reshape(X.shape[0], 1, X.shape[1]))
            lazy_y = LazyTensor(Y.type(X.dtype).reshape(1, Y.shape[0], Y.shape[1]))
            if x_weights is not None:
                if isinstance(x_weights, np.ndarray):
                    x_weights = torch.from_numpy(x_weights)
                w = LazyTensor(x_weights.reshape(-1, 1).type(X.dtype), axis=0)
                return lazy_x, lazy_y, w
            return lazy_x, lazy_y, None
        raise Exception("Bad tensor type.")

    def score_samples(self, Y, X=None, return_kernel=False):
        """Returns the kernel density estimates of each point in `Y`.

        Parameters
        ----------
        Y : tensor (m, d)
                `m` points with `d` dimensions for which the probability density will
                be calculated
        X : tensor (n, d), optional
                `n` points with `d` dimensions to which KDE will be fit. Provided to
                allow batch calculations in `log_prob`. By default, `X` is None and
                all points used to initialize KernelDensityEstimator are included.


        Returns
        -------
        log_probs : tensor (m)
                log probability densities for each of the queried points in `Y`
        """
        assert self.api is not None and self._kernel is not None, "Fit first."
        X = self.X if X is None else self.api.astensor(X)
        Y = self.api.astensor(Y)
        if X.shape[0] == 0:
            return self.api.zeros((Y.shape[0]), dtype=X.dtype)
        assert Y.shape[1] == X.shape[1] and X.ndim == Y.ndim == 2
        if self._implementation == "dense":
            return self._score_samples_dense(Y, return_kernel=return_kernel)
        lazy_x, lazy_y, w = self.to_lazy(X, Y, x_weights=self._sample_weights)
        kernel = self._kernel(
            lazy_x,
            lazy_y,
            self.api.astensor(self.bandwidth, dtype=X.dtype),
            api=self.api,
            **self.kwargs,
        )
        if w is not None:
            kernel *= w
        if return_kernel:
            return kernel
        density_estimation = kernel.sum(dim=0).squeeze() / kernel.shape[0]  # mean
        return (
            self.api.log(density_estimation) if self.return_log else density_estimation
        )


class DTM:
    """
    Distance To Measure
    """

    def __init__(self, masses, metric: str = "euclidean", **_kdtree_kwargs):
        """
        mass : float in [0,1]
                The mass threshold
        metric :
                The distance between points to consider
        """
        self.masses = masses
        self.metric = metric
        self._kdtree_kwargs = _kdtree_kwargs
        self._ks = None
        self._kdtree = None
        self._X = None
        self.api = None

    def fit(self, X, sample_weights=None, y=None):
        if len(self.masses) == 0:
            return self
        assert np.max(self.masses) <= 1, "All masses should be in (0,1]."
        from sklearn.neighbors import KDTree

        self.api = api_from_tensor(X)
        _X = self.api.asnumpy(X)
        self._ks = np.array([int(mass * X.shape[0]) + 1 for mass in self.masses])
        self._kdtree = KDTree(_X, metric=self.metric, **self._kdtree_kwargs)
        self._X = X
        return self

    def score_samples(self, Y, X=None):
        """Returns the kernel density estimates of each point in `Y`.

        Parameters
        ----------
        Y : tensor (m, d)
                `m` points with `d` dimensions for which the probability density will
                be calculated


        Returns
        -------
        the DTMs of Y, for each mass in masses.
        """
        if self.api is None:
            raise ValueError("Fit first")
        if len(self.masses) == 0:
            return self.api.empty((0, len(Y)))
        assert (
            self._ks is not None and self._kdtree is not None and self._X is not None
        ), f"Fit first. Got {self._ks=}, {self._kdtree=}, {self._X=}."
        assert Y.ndim == 2
        if self.api is _npapi:
            _Y = Y
        else:
            _Y = self.api.asnumpy(Y)
        NN_Dist, NN = self._kdtree.query(_Y, self._ks.max(), return_distance=True)
        DTMs = np.array([((NN_Dist**2)[:, :k].mean(1)) ** 0.5 for k in self._ks])
        return self.api.astensor(DTMs)

    def score_samples_diff(self, Y):
        """Returns the kernel density estimates of each point in `Y`.

        Parameters
        ----------
        Y : tensor (m, d)
                `m` points with `d` dimensions for which the probability density will
                be calculated
        X : tensor (n, d), optional
                `n` points with `d` dimensions to which KDE will be fit. Provided to
                allow batch calculations in `log_prob`. By default, `X` is None and
                all points used to initialize KernelDensityEstimator are included.


        Returns
        -------
        log_probs : tensor (m)
                log probability densities for each of the queried points in `Y`
        """
        import torch
        from multipers.array_api import torch

        if len(self.masses) == 0:
            return torch.empty(0, len(Y))

        assert Y.ndim == 2
        assert self.api is torch, "Use the non-diff version with numpy."
        assert (
            self._ks is not None and self._kdtree is not None and self._X is not None
        ), f"Fit first. Got {self._ks=}, {self._kdtree=}, {self._X=}."
        NN = self._kdtree.query(Y.detach(), self._ks.max(), return_distance=False)
        DTMs = tuple(
            (((self._X[NN] - Y[:, None, :]) ** 2)[:, :k].sum(dim=(1, 2)) / k) ** 0.5
            for k in self._ks
        )  # TODO : kdtree already computes distance, find implementation of kdtree that is pytorch differentiable
        return DTMs


## code taken from pykeops doc (https://www.kernel-operations.io/keops/_auto_benchmarks/benchmark_KNN.html)
class KNNmean:
    def __init__(self, k: int, metric: str = "euclidean"):
        self.k = k
        self.metric = metric
        self._KNN_fun = None
        self._x = None

    def fit(self, x):
        if isinstance(x, np.ndarray):
            from pykeops.numpy import Vi, Vj
        else:
            import torch

            assert isinstance(x, torch.Tensor), "Backend has to be numpy or torch"
            from pykeops.torch import Vi, Vj

        D = x.shape[1]
        X_i = Vi(0, D)
        X_j = Vj(1, D)

        # Symbolic distance matrix:
        if self.metric == "euclidean":
            D_ij = ((X_i - X_j) ** 2).sum(-1) ** (1 / 2)
        elif self.metric == "manhattan":
            D_ij = (X_i - X_j).abs().sum(-1)
        elif self.metric == "angular":
            D_ij = -(X_i | X_j)
        elif self.metric == "hyperbolic":
            D_ij = ((X_i - X_j) ** 2).sum(-1) / (X_i[0] * X_j[0])
        else:
            raise NotImplementedError(f"The '{self.metric}' distance is not supported.")

        self._x = x
        self._KNN_fun = D_ij.Kmin(self.k, dim=1)
        return self

    def score_samples(self, x):
        assert self._x is not None and self._KNN_fun is not None, "Fit first."
        return self._KNN_fun(x, self._x).sum(axis=1) / self.k


# def _pts_convolution_sparse(pts:np.ndarray, pts_weights:np.ndarray, filtration_grid:Iterable[np.ndarray], kernel="gaussian", bandwidth=0.1, **more_kde_args):
# 	"""
# 	Old version of `convolution_signed_measures`. Scikitlearn's convolution is slower than the code above.
# 	"""
# 	from sklearn.neighbors import KernelDensity
# 	grid_iterator = np.asarray(list(product(*filtration_grid)))
# 	grid_shape = [len(f) for f in filtration_grid]
# 	if len(pts) == 0:
# 		# warn("Found a trivial signed measure !")
# 		return np.zeros(shape=grid_shape)
# 	kde = KernelDensity(kernel=kernel, bandwidth=bandwidth, rtol = 1e-4, **more_kde_args) # TODO : check rtol

# 	pos_indices = pts_weights>0
# 	neg_indices = pts_weights<0
# 	img_pos = kde.fit(pts[pos_indices], sample_weight=pts_weights[pos_indices]).score_samples(grid_iterator).reshape(grid_shape)
# 	img_neg = kde.fit(pts[neg_indices], sample_weight=-pts_weights[neg_indices]).score_samples(grid_iterator).reshape(grid_shape)
# 	return np.exp(img_pos) - np.exp(img_neg)


# Precompiles the convolution
# _test(r=2,b=.5, plot=False)
