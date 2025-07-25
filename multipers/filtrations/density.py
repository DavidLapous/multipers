from collections.abc import Callable, Iterable
from typing import Any, Literal, Union

import numpy as np

from multipers.array_api import api_from_tensor, api_from_tensors

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

     - iterable_of_signed_measures : (num_signed_measure) x [ (npts) x (num_parameters), (npts)]
     - filtrations : (num_parameter) x (filtration values)
     - flatten : bool
     - n_jobs : int

    Outputs
    -------

    The concatenated images, for each signed measure (num_signed_measures) x (len(f) for f in filtration_values)
    """
    from multipers.grids import todense

    grid_iterator = todense(filtrations, product_order=True)
    api = api_from_tensor(iterable_of_signed_measures[0][0][0])
    match backend:
        case "sklearn":

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
                **kwargs,
            )

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
    kde = KernelDensity(
        kernel=kernel, bandwidth=bandwidth, rtol=1e-4, **more_kde_args
    )  # TODO : check rtol
    pos_indices = pts_weights > 0
    neg_indices = pts_weights < 0
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
    return np.exp(img_pos) - np.exp(img_neg)


def _pts_convolution_pykeops(
    pts: np.ndarray,
    pts_weights: np.ndarray,
    grid_iterator,
    kernel: available_kernels = "gaussian",
    bandwidth=0.1,
    **more_kde_args,
):
    """
    Pykeops convolution
    """
    if isinstance(pts, np.ndarray):
        _asarray_weights = lambda x: np.asarray(x, dtype=pts.dtype)
        _asarray_grid = _asarray_weights
    else:
        import torch

        _asarray_weights = lambda x: torch.from_numpy(x).type(pts.dtype)
        _asarray_grid = lambda x: x.type(pts.dtype)
    kde = KDE(kernel=kernel, bandwidth=bandwidth, **more_kde_args)
    return kde.fit(pts, sample_weights=_asarray_weights(pts_weights)).score_samples(
        _asarray_grid(grid_iterator)
    )


def gaussian_kernel(x_i, y_j, bandwidth):
    D = x_i.shape[-1]
    exponent = -(((x_i - y_j) / bandwidth) ** 2).sum(dim=-1) / 2
    # float is necessary for some reason (pykeops fails)
    kernel = (exponent).exp() / float((bandwidth * np.sqrt(2 * np.pi)) ** D)
    return kernel


def multivariate_gaussian_kernel(x_i, y_j, covariance_matrix_inverse):
    # 1 / \sqrt(2 \pi^dim * \Sigma.det()) * exp( -(x-y).T @ \Sigma ^{-1} @ (x-y))
    # CF https://www.kernel-operations.io/keops/_auto_examples/pytorch/plot_anisotropic_kernels.html#sphx-glr-auto-examples-pytorch-plot-anisotropic-kernels-py
    #    and https://www.kernel-operations.io/keops/api/math-operations.html
    dim = x_i.shape[-1]
    z = x_i - y_j
    exponent = -(z.weightedsqnorm(covariance_matrix_inverse.flatten()) / 2)
    return (
        float((2 * np.pi) ** (-dim / 2))
        * (covariance_matrix_inverse.det().sqrt())
        * exponent.exp()
    )


def exponential_kernel(x_i, y_j, bandwidth):
    # 1 / \sigma * exp( norm(x-y, dim=-1))
    exponent = -(((((x_i - y_j) ** 2)).sum(dim=-1) ** 1 / 2) / bandwidth)
    kernel = exponent.exp() / bandwidth
    return kernel


def sinc_kernel(x_i, y_j, bandwidth):
    norm = ((((x_i - y_j) ** 2)).sum(dim=-1) ** 1 / 2) / bandwidth
    sinc = type(x_i).sinc
    kernel = 2 * sinc(2 * norm) - sinc(norm)
    return kernel


def _kernel(
    kernel: available_kernels = "gaussian",
):
    match kernel:
        case "gaussian":
            return gaussian_kernel
        case "exponential":
            return exponential_kernel
        case "multivariate_gaussian":
            return multivariate_gaussian_kernel
        case "sinc":
            return sinc_kernel
        case _:
            assert callable(
                kernel
            ), f"""
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
    Fast, scikit-style, and differentiable kernel density estimation, using PyKeops.
    """

    def __init__(
        self,
        bandwidth: Any = 1,
        kernel: available_kernels = "gaussian",
        return_log: bool = False,
    ):
        """
        bandwidth : numeric
                bandwidth for Gaussian kernel
        """
        self.X = None
        self.bandwidth = bandwidth
        self.kernel: available_kernels = kernel
        self._kernel = None
        self._backend = None
        self._sample_weights = None
        self.return_log = return_log

    def fit(self, X, sample_weights=None, y=None):
        self.X = X
        self._sample_weights = sample_weights
        if isinstance(X, np.ndarray):
            self._backend = np
        else:
            import torch

            if isinstance(X, torch.Tensor):
                self._backend = torch
            else:
                raise Exception("Unsupported backend.")
        self._kernel = _kernel(self.kernel)
        return self

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

            lazy_x = LazyTensor(X.view(X.shape[0], 1, X.shape[1]))
            lazy_y = LazyTensor(Y.type(X.dtype).view(1, Y.shape[0], Y.shape[1]))
            if x_weights is not None:
                if isinstance(x_weights, np.ndarray):
                    x_weights = torch.from_numpy(x_weights)
                w = LazyTensor(x_weights[:, None].type(X.dtype), axis=0)
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
        assert self._backend is not None and self._kernel is not None, "Fit first."
        X = self.X if X is None else X
        if X.shape[0] == 0:
            return self._backend.zeros((Y.shape[0]))
        assert Y.shape[1] == X.shape[1] and X.ndim == Y.ndim == 2
        lazy_x, lazy_y, w = self.to_lazy(X, Y, x_weights=self._sample_weights)
        kernel = self._kernel(lazy_x, lazy_y, self.bandwidth)
        if w is not None:
            kernel *= w
        if return_kernel:
            return kernel
        density_estimation = kernel.sum(dim=0).squeeze() / kernel.shape[0]  # mean
        return (
            self._backend.log(density_estimation)
            if self.return_log
            else density_estimation
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
        self._backend = None

    def fit(self, X, sample_weights=None, y=None):
        if len(self.masses) == 0:
            return self
        assert np.max(self.masses) <= 1, "All masses should be in (0,1]."
        from sklearn.neighbors import KDTree

        if not isinstance(X, np.ndarray):
            import torch

            assert isinstance(X, torch.Tensor), "Backend has to be numpy of torch"
            _X = X.detach()
            self._backend = "torch"
        else:
            _X = X
            self._backend = "numpy"
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
        if len(self.masses) == 0:
            return np.empty((0, len(Y)))
        assert (
            self._ks is not None and self._kdtree is not None and self._X is not None
        ), f"Fit first. Got {self._ks=}, {self._kdtree=}, {self._X=}."
        assert Y.ndim == 2
        if self._backend == "torch":
            _Y = Y.detach().numpy()
        else:
            _Y = Y
        NN_Dist, NN = self._kdtree.query(_Y, self._ks.max(), return_distance=True)
        DTMs = np.array([((NN_Dist**2)[:, :k].mean(1)) ** 0.5 for k in self._ks])
        return DTMs

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

        if len(self.masses) == 0:
            return torch.empty(0, len(Y))

        assert Y.ndim == 2
        assert self._backend == "torch", "Use the non-diff version with numpy."
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
