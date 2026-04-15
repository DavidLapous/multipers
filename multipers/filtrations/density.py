from collections.abc import Callable, Iterable
from contextlib import contextmanager
from time import perf_counter
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
    backend="auto",
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
    if backend in {"auto", "pykeops"}:
        implementation = "auto"
    elif backend == "dense":
        implementation = "dense"
    else:
        raise ValueError(f"Unknown convolution backend {backend}.")

    def score_measure(pts, weights):
        pts = api.astensor(pts)
        dtype = pts.dtype
        return (
            KDE(
                kernel=kernel,
                bandwidth=bandwidth,
                implementation=implementation,
                api=api,
                **kwargs,
            )
            .fit(pts, sample_weights=api.astype(weights, dtype))
            .score_samples(api.astype(grid_iterator, dtype))
        )

    def convolution_signed_measures_on_grid(
        signed_measures: Iterable[tuple[np.ndarray, np.ndarray]],
    ) -> np.ndarray:
        return api.cat([score_measure(pts, weights) for pts, weights in signed_measures], axis=0)

    # compiles first once
    pts, weights = iterable_of_signed_measures[0][0]
    score_measure(pts[:2], weights[:2])

    if n_jobs > 1 or n_jobs == -1:
        from joblib import Parallel, delayed

        convolutions = Parallel(n_jobs=n_jobs, prefer="threads")(
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
def gaussian_kernel(x_i, y_j, bandwidth, api=_npapi):
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


def exponential_kernel(x_i, y_j, bandwidth, api=_npapi):
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


def _uses_logsumexp_dense_path(kernel: available_kernels) -> bool:
    return not callable(kernel) and kernel in {
        "gaussian",
        "exponential",
        "exponential_kernel",
    }


def _dense_log_kernel(
    X,
    Y,
    *,
    kernel: available_kernels,
    bandwidth,
    api,
):
    X = api.astensor(X)
    Y = api.astype(Y, X.dtype)
    pairwise_distances = api.cdist(X, Y)
    bandwidth = api.astensor(bandwidth, dtype=X.dtype)
    if bandwidth.ndim != 0:
        bandwidth = bandwidth.reshape(())

    match kernel:
        case "gaussian":
            dim = X.shape[-1]
            log_normalization = api.astensor(
                dim * 0.5 * np.log(2 * np.pi), dtype=X.dtype
            ) + dim * api.log(bandwidth)
            return -((pairwise_distances / bandwidth) ** 2) / 2 - log_normalization
        case "exponential" | "exponential_kernel":
            return -(pairwise_distances / bandwidth) - api.log(bandwidth)
        case _:
            raise ValueError(
                f"Log-sum-exp dense scoring is unsupported for kernel {kernel}."
            )


def _dense_kernel(
    X,
    Y,
    *,
    kernel: available_kernels,
    bandwidth,
    api,
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
        api=None,
        verbose: bool = False,
    ):
        """
        bandwidth : numeric
                bandwidth for Gaussian kernel
        """
        self.X = None
        self.bandwidth = bandwidth
        self.kernel: available_kernels = kernel
        self._kernel = None
        self.api = api
        self._api = None
        self._sample_weights = None
        self.return_log = return_log
        self.implementation = implementation
        self.chunk_size = chunk_size
        self.verbose = verbose
        self._implementation = None
        self._dense_score_samples = None

    def fit(self, X, sample_weights=None, y=None):
        self._api = (
            api_from_tensor(
                X,
                jit_promote=self.implementation == "dense"
                or (self.implementation == "auto" and not _npapi.check_keops()),
            )
            if self.api is None
            else self.api
        )
        self.X = self._api.astensor(X)
        self._sample_weights = sample_weights
        self._kernel = _kernel(self.kernel)
        self._implementation = self._resolve_implementation()
        self._log(
            "fit:resolving_api",
            (
                f"Using {self._implementation} implementation "
                f"(requested {self.implementation}) on {self._api.name} backend."
            ),
        )
        self._dense_score_samples = None
        return self

    def _resolve_implementation(self) -> Literal["pykeops", "dense"]:
        if self.implementation == "pykeops":
            if not self._api.check_keops():
                raise ValueError(f"PyKeOps is unavailable for backend `{self._api}`.")
            return "pykeops"
        if self.implementation == "auto" and self._api.check_keops():
            return "pykeops"
        return "dense"

    def _log(self, scope: str, message: str) -> None:
        if self.verbose:
            print(f"[KDE][{scope}] {message}", flush=True)

    @contextmanager
    def _timed(self, scope: str, message: str):
        if not self.verbose:
            yield
            return
        self._log(scope, f"{message}...")
        start = perf_counter()
        try:
            yield
        finally:
            self._log(scope, f"Done in {perf_counter() - start:.3f}s.")

    def _make_dense_score_samples(self):
        assert self._api is not None and self.X is not None
        api = self._api
        X = self.X
        if _uses_logsumexp_dense_path(self.kernel):
            log_normalizer = api.log(api.astensor(X.shape[0], dtype=X.dtype))
            weights = None
            positive_log_weights = None
            negative_log_weights = None
            positive_only = True
            if self._sample_weights is not None:
                weights = api.astensor(self._sample_weights, dtype=X.dtype)
                neg_inf = api.astensor(-np.inf, dtype=X.dtype)
                abs_weights = api.abs(weights)
                safe_abs_weights = api.where(
                    abs_weights > 0, abs_weights, abs_weights * 0 + 1
                )
                log_abs_weights = api.where(abs_weights > 0, api.log(safe_abs_weights), neg_inf)
                positive_log_weights = api.where(
                    weights > 0, log_abs_weights, neg_inf
                )
                negative_log_weights = api.where(
                    weights < 0, log_abs_weights, neg_inf
                )
                positive_only = bool(
                    float(api.asnumpy(api.minvalues(weights))) >= 0.0
                )

            def score_chunk(X_chunk, Y_chunk):
                log_kernel_values = _dense_log_kernel(
                    X_chunk,
                    Y_chunk,
                    kernel=self.kernel,
                    bandwidth=self.bandwidth,
                    api=api,
                )
                if weights is None:
                    log_density = api.logsumexp(log_kernel_values, axis=0) - log_normalizer
                    return log_density if self.return_log else api.exp(log_density)

                positive_log_density = (
                    api.logsumexp(log_kernel_values + positive_log_weights[:, None], axis=0)
                    - log_normalizer
                )
                if positive_only:
                    return positive_log_density if self.return_log else api.exp(positive_log_density)

                negative_log_density = (
                    api.logsumexp(log_kernel_values + negative_log_weights[:, None], axis=0)
                    - log_normalizer
                )
                density = api.exp(positive_log_density) - api.exp(negative_log_density)
                return api.log(density) if self.return_log else density

            return api.jit(score_chunk)

        weights = None
        if self._sample_weights is not None:
            weights = api.astensor(self._sample_weights, dtype=X.dtype)

        def score_chunk(X_chunk, Y_chunk):
            kernel_values = _dense_kernel(
                X_chunk,
                Y_chunk,
                kernel=self.kernel,
                bandwidth=self.bandwidth,
                api=api,
            )
            if weights is not None:
                kernel_values = kernel_values * weights[:, None]
            return api.sum(kernel_values, axis=0) / kernel_values.shape[0]

        return api.jit(score_chunk)

    def _score_samples_dense(self, Y, return_kernel=False):
        assert self._api is not None and self.X is not None
        api = self._api
        X = self.X
        if X.shape[0] == 0:
            return api.zeros((Y.shape[0]), dtype=X.dtype)
        if self._dense_score_samples is None:
            with self._timed("score_samples:dense:setup", "Building dense scorer"):
                self._dense_score_samples = self._make_dense_score_samples()

        Y = api.astype(Y, X.dtype)
        if return_kernel:
            kernel_values = _dense_kernel(
                X,
                Y,
                kernel=self.kernel,
                bandwidth=self.bandwidth,
                api=api,
            )
            if self._sample_weights is not None:
                kernel_values = (
                    kernel_values
                    * api.astensor(self._sample_weights, dtype=X.dtype)[:, None]
                )
            return kernel_values

        outputs = []
        for start in range(0, Y.shape[0], self.chunk_size):
            stop = min(start + self.chunk_size, Y.shape[0])
            outputs.append(self._dense_score_samples(X, Y[start:stop]))
        density_estimation = outputs[0] if len(outputs) == 1 else api.cat(outputs)
        if _uses_logsumexp_dense_path(self.kernel):
            return density_estimation
        return api.log(density_estimation) if self.return_log else density_estimation

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
        assert self._api is not None and self._kernel is not None, "Fit first."
        api = self._api
        X = self.X if X is None else api.astensor(X)
        Y = api.astensor(Y)
        if X.shape[0] == 0:
            return api.zeros((Y.shape[0]), dtype=X.dtype)
        assert Y.shape[1] == X.shape[1] and X.ndim == Y.ndim == 2
        operation = (
            f"Computing kernel matrix for {Y.shape[0]} query points"
            if return_kernel
            else f"Scoring {Y.shape[0]} query points"
        )
        with self._timed(f"score_samples:{self._implementation}", operation):
            if self._implementation == "dense":
                return self._score_samples_dense(Y, return_kernel=return_kernel)
            lazy_x, lazy_y, w = self.to_lazy(X, Y, x_weights=self._sample_weights)
            kernel = self._kernel(
                lazy_x,
                lazy_y,
                api.astensor(self.bandwidth, dtype=X.dtype),
                api=api,
            )
            if w is not None:
                kernel *= w
            if return_kernel:
                return kernel
            density_estimation = kernel.sum(dim=0).squeeze() / kernel.shape[0]  # mean
            return api.log(density_estimation) if self.return_log else density_estimation


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
    def __init__(self, k: int, metric: str = "euclidean", api=None):
        self.k = k
        self.metric = metric
        self.api = api
        self._api = None
        self._implementation = None
        self._KNN_fun = None
        self._x = None
        self._dense_score_samples = None

    def fit(self, x):
        self._api = (
            api_from_tensor(x, jit_promote=not _npapi.check_keops())
            if self.api is None
            else self.api
        )
        self._x = self._api.astensor(x)
        self._dense_score_samples = None
        self._implementation = "pykeops" if self._api.check_keops() else "dense"

        if self._implementation == "dense":
            self._KNN_fun = None
            return self

        D = self._x.shape[1]
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

        self._KNN_fun = D_ij.Kmin(self.k, dim=1)
        return self

    def _make_dense_score_samples(self):
        assert self._api is not None and self._x is not None
        api = self._api
        x = self._x
        if self.metric != "euclidean":
            raise NotImplementedError(
                "Dense JAX KNNmean currently only supports the euclidean metric."
            )

        def score_chunk(y):
            distances = api.cdist(api.astype(y, x.dtype), x, p=2)
            knn_distances = api.sort(distances, axis=1)[:, : self.k]
            return api.mean(knn_distances, axis=1)

        return api.jit(score_chunk)

    def score_samples(self, x):
        from sklearn.exceptions import NotFittedError

        if self._x is None or self._api is None or self._implementation is None:
            raise NotFittedError(
                "This KNNmean instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator."
            )
        if self._implementation == "pykeops":
            if self._KNN_fun is None:
                raise NotFittedError(
                    "This KNNmean instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator."
                )
            return self._KNN_fun(x, self._x).sum(axis=1) / self.k
        if self._dense_score_samples is None:
            self._dense_score_samples = self._make_dense_score_samples()
        return self._dense_score_samples(self._api.astensor(x))
