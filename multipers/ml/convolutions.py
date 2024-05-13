from collections.abc import Callable
from itertools import product
from typing import Any, Iterable, Literal

import numpy as np


def convolution_signed_measures(
    iterable_of_signed_measures,
    filtrations,
    bandwidth,
    flatten: bool = True,
    n_jobs: int = 1,
    backend="pykeops",
    kernel="gaussian",
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
    grid_iterator = np.array(list(product(*filtrations)), dtype=float)
    match backend:
        case "sklearn":

            def convolution_signed_measures_on_grid(
                signed_measures: Iterable[tuple[np.ndarray, np.ndarray]],
            ):
                return np.concatenate(
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
            ):
                return np.concatenate(
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
    return np.asarray(convolutions, dtype=float)


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
    kernel="gaussian",
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
    kernel="gaussian",
    bandwidth=0.1,
    **more_kde_args,
):
    """
    Pykeops convolution
    """
    kde = KDE(kernel=kernel, bandwidth=bandwidth, **more_kde_args)
    return kde.fit(
        pts, sample_weights=np.asarray(pts_weights, dtype=pts.dtype)
    ).score_samples(np.asarray(grid_iterator, dtype=pts.dtype))


def gaussian_kernel(x_i, y_j, bandwidth):
    exponent = -(((x_i - y_j) / bandwidth) ** 2).sum(dim=-1) / 2
    # float is necessary for some reason (pykeops fails)
    kernel = (exponent).exp() / (bandwidth * float(np.sqrt(2 * np.pi)))
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
    exponent = -(((((x_i - y_j) ** 2).sum()) ** 1 / 2) / bandwidth).sum(dim=-1)
    kernel = exponent.exp() / bandwidth
    return kernel


def _kernel(
    kernel: (
        Literal["gaussian", "exponential", "multivariate_gaussian"] | Callable
    ) = "gaussian",
):
    match kernel:
        case "gaussian":
            return gaussian_kernel
        case "exponential":
            return exponential_kernel
        case "multivariate_gaussian":
            return multivariate_gaussian_kernel
        case _:
            assert callable(
                kernel
            ), f"--------------------------\nUnknown kernel {kernel}.\n--------------------------\n Custom kernel has to be callable, (x:LazyTensor(n,1,D),y:LazyTensor(1,m,D),bandwidth:float) ---> kernel matrix"
            return kernel


# TODO : multiple bandwidths at once with lazy tensors
class KDE:
    """
    Fast, scikit-style, and differentiable kernel density estimation, using PyKeops.
    """

    def __init__(
        self,
        bandwidth: Any = 1,
        kernel: (
            Literal["m_gaussian", "gaussian", "exponential"] | Callable
        ) = "gaussian",
        return_log=False,
    ):
        """
        bandwidth : numeric
                bandwidth for Gaussian kernel
        """
        self.X = None
        self.bandwidth = bandwidth
        self.kernel = kernel
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
        match self.kernel:
            case "gaussian":
                self._kernel = self.gaussian_kernel
            case "m_gaussian":
                self._kernel = self.multivariate_gaussian_kernel
            case "exponential":
                self._kernel = self.exponential_kernel
            case _:
                assert callable(
                    self.kernel
                ), f"""
--------------------------
Unknown kernel {self.kernel}.
--------------------------
Custom kernel has to be callable,
(x:LazyTensor(n,1,D),y:LazyTensor(1,m,D),bandwidth:float) ---> kernel matrix
"""
                self._kernel = self.kernel
        return self

    @staticmethod
    def gaussian_kernel(x_i, y_j, bandwidth):
        exponent = -(((x_i - y_j) / bandwidth) ** 2).sum(dim=2) / 2
        # float is necessary for some reason (pykeops fails)
        kernel = (exponent).exp() / (bandwidth * float(np.sqrt(2 * np.pi)))
        return kernel

    @staticmethod
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

    @staticmethod
    def exponential_kernel(x_i, y_j, bandwidth):
        exponent = -(((((x_i - y_j) ** 2).sum()) ** 1 / 2) / bandwidth).sum(dim=2)
        kernel = exponent.exp() / bandwidth
        return kernel

    @staticmethod
    def to_lazy(X, Y, x_weights):
        if isinstance(X, np.ndarray):
            from pykeops.numpy import LazyTensor

            lazy_x = LazyTensor(
                X.reshape((X.shape[0], 1, X.shape[1]))
            )  # numpts, 1, dim
            lazy_y = LazyTensor(
                Y.reshape((1, Y.shape[0], Y.shape[1]))
            )  # 1, numpts, dim
            if x_weights is not None:
                w = LazyTensor(x_weights[:, None], axis=0)
                return lazy_x, lazy_y, w
            return lazy_x, lazy_y, None
        import torch

        if isinstance(X, torch.Tensor):
            from pykeops.torch import LazyTensor

            lazy_x = LazyTensor(X.view(X.shape[0], 1, X.shape[1]))
            lazy_y = LazyTensor(Y.view(1, Y.shape[0], Y.shape[1]))
            if x_weights is not None:
                w = LazyTensor(x_weights[:, None], axis=0)
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
        density_estimation = kernel.sum(dim=0).flatten() / kernel.shape[0]  # mean
        return (
            self._backend.log(density_estimation)
            if self.return_log
            else density_estimation
        )


def batch_signed_measure_convolutions(
    signed_measures,  # array of shape (num_data,num_pts,D)
    x,  # array of shape (num_x, D) or (num_data, num_x, D)
    bandwidth,  # either float or matrix if multivariate kernel
    kernel,
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
    if signed_measures.ndim == 2:
        signed_measures = signed_measures[None, :, :]
    sms = signed_measures[..., :-1]
    weights = signed_measures[..., -1]
    if isinstance(signed_measures, np.ndarray):
        from pykeops.numpy import LazyTensor
    else:
        import torch

        assert isinstance(signed_measures, torch.Tensor)
        from pykeops.torch import LazyTensor

    _sms = LazyTensor(sms[..., None, :].contiguous())
    _x = x[..., None, :, :].contiguous()

    sms_kernel = _kernel(kernel)(_sms, _x, bandwidth)
    out = (sms_kernel * weights[..., None, None].contiguous()).sum(
        signed_measures.ndim - 2
    )
    assert out.shape[-1] == 1, "Pykeops bug fixed, TODO : refix this "
    out = out[..., 0]  ## pykeops bug + ensures its a tensor
    # assert out.shape == (x.shape[0], x.shape[1]), f"{x.shape=}, {out.shape=}"
    return out


class DTM:
    """
    Distance To Measure
    """

    def __init__(self, masses=[0.1], metric: str = "euclidean", **_kdtree_kwargs):
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

        assert Y.ndim == 2
        assert self._backend == "torch", "Use the non-diff version with numpy."
        if len(self.masses) == 0:
            return torch.empty(0, len(Y))
        NN = self._kdtree.query(Y.detach(), self._ks.max(), return_distance=False)
        DTMs = tuple(
            (((self._X[NN] - Y[:, None, :]) ** 2)[:, :k].sum(dim=(1, 2)) / k) ** 0.5
            for k in self._ks
        )  # TODO : kdtree already computes distance, find implementation of kdtree that is pytorch differentiable
        return DTMs


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
