# This code was written by Mathieu Carrière.

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import pairwise_kernels

from multipers.array_api import api_from_tensor
from multipers.distances import (
    _compute_signed_measure_projections,
    _infer_api,
    _repeat_signed_points,
    _sliced_wasserstein_distance_on_projections,
    sm_distance,
    pairwise_distances,
)


def _build_transport_pair(meas1, meas2, api):
    meas1_plus, meas1_minus = meas1
    meas2_plus, meas2_minus = meas2
    return api.cat([meas1_plus, meas2_minus], 0), api.cat([meas2_plus, meas1_minus], 0)


def _ot_distance(meas1, meas2, ground_norm=1, epsilon=1.0):
    api = api_from_tensor(meas1[0])
    meas_t1, meas_t2 = _build_transport_pair(meas1, meas2, api)
    num_pts = len(meas_t1)
    import ot

    weights = api.ones(num_pts, dtype=meas_t1.dtype) / num_pts
    if ground_norm < 1:
        raise ValueError("Only lp metrics with p >= 1 are supported.")
    cost = api.cdist(meas_t1, meas_t2, p=ground_norm)
    if epsilon > 0:
        wass = ot.sinkhorn2(weights, weights, cost, epsilon)
        return wass[0]
    return ot.lp.emd2(weights, weights, cost)


def _compute_signed_measure_parts(X):
    """
    This is a function for separating the positive and negative points of a list of signed measures. This function can be used as a preprocessing step in order to speed up the running time for computing all pairwise (sliced) Wasserstein distances on a list of signed measures.

    Parameters:
        X (list of n tuples): list of signed measures.

    Returns:
        list of n pairs of numpy arrays of shape (num x dimension): list of positive and negative signed measures.
    """
    from multipers.distances import _normalize_signed_measures

    measures, api, _, _, _ = _normalize_signed_measures(X)
    return [
        [_repeat_signed_points(C, M, 1, api), _repeat_signed_points(C, M, -1, api)]
        for C, M in measures
    ]


def pairwise_signed_measure_distances(
    X, Y=None, metric="sliced_wasserstein", n_jobs=None, **kwargs
):
    """
    This function computes the distance matrix between two lists of signed measures given as numpy arrays of shape (nxD).

    Parameters:
        X (list of n tuples): first list of signed measures.
        Y (list of m tuples): second list of signed measures (optional). If None, pairwise distances are computed from the first list only.
        metric: distance to use. It can be either a string ("sliced_wasserstein", "wasserstein") or a function taking two tuples as inputs. If it is a function, make sure that it is symmetric and that it outputs 0 if called on the same two tuples.
        n_jobs (int): number of jobs to use for the computation. This uses joblib.Parallel(prefer="threads"), so metrics that do not release the GIL may not scale unless run inside a `joblib.parallel_backend <https://joblib.readthedocs.io/en/latest/parallel.html#joblib.parallel_backend>`_ block.
        **kwargs: optional keyword parameters. Any further parameters are passed directly to the distance function. See the docs of the various distance classes in this module.

    Returns:
        numpy array of shape (nxm): distance matrix
    """
    kwargs = dict(kwargs)
    api = _infer_api(X, Y, kwargs.pop("api", None))
    if metric == "sliced_wasserstein":
        Xproj = _compute_signed_measure_projections(X, api=api, **kwargs)
        Yproj = (
            None
            if Y is None
            else _compute_signed_measure_projections(Y, api=api, **kwargs)
        )
        return pairwise_distances(
            Xproj, Yproj, _sliced_wasserstein_distance_on_projections, n_jobs, api
        )
    elif metric == "wasserstein":
        Xproj = _compute_signed_measure_parts(X)
        Yproj = None if Y is None else _compute_signed_measure_parts(Y)
        return pairwise_distances(
            Xproj, Yproj, _wasserstein_distance_on_parts(**kwargs), n_jobs, api
        )
    metric_fn = lambda a, b: metric(a, b, **kwargs)
    return pairwise_distances(X, Y, metric_fn, n_jobs, api)


def _wasserstein_distance_on_parts(ground_norm=1, epsilon=1.0):
    """
    This is a function for computing the Wasserstein distance between two signed measures that have already been separated into their positive and negative parts.

    Parameters:
        meas1: pair of (n x dimension) numpy.arrays containing the points of the positive and negative parts of the first measure.
        meas2: pair of (m x dimension) numpy.arrays containing the points of the positive and negative parts of the second measure.

    Returns:
        float: the sliced Wasserstein distance between the projected signed measures.
    """

    def metric(meas1, meas2):
        return _ot_distance(meas1, meas2, ground_norm=ground_norm, epsilon=epsilon)

    return metric


def _wasserstein_distance(meas1, meas2, epsilon, ground_norm):
    """
    This is a function for computing the Wasserstein distance from two signed measures.

    Parameters:
        meas1: ((n x D), (n)) tuple with numpy.array encoding the (finite points of the) first measure and their multiplicities. Must not contain essential points (i.e. with infinite coordinate).
        meas2: ((m x D), (m)) tuple encoding the second measure.
        epsilon (float): entropy regularization parameter.
        ground_norm (int): norm to use for ground metric cost.

    Returns:
        float: the Wasserstein distance between signed measures.
    """
    approx1, approx2 = _compute_signed_measure_parts([meas1, meas2])
    return _ot_distance(approx1, approx2, ground_norm=ground_norm, epsilon=epsilon)


class SlicedWassersteinDistance(BaseEstimator, TransformerMixin):
    """
    This is a class for computing the sliced Wasserstein distance matrix from a list of signed measures. The Sliced Wasserstein distance is computed by projecting the signed measures onto lines, comparing the projections with the 1-norm, and finally integrating over all possible lines. See http://proceedings.mlr.press/v70/carriere17a.html for more details.
    """

    def __init__(self, num_directions=10, scales=None, n_jobs=None, seed: int = 42):
        """
        Constructor for the SlicedWassersteinDistance class.

        Parameters:
            num_directions (int): number of lines evenly sampled in order to approximate and speed up the distance computation (default 10).
            scales (array of shape D): scales associated to the dimensions.
            n_jobs (int): number of jobs to use for the computation. See :func:`pairwise_signed_measure_distances` for details.
        """
        self.num_directions = num_directions
        self.scales = scales
        self.n_jobs = n_jobs
        self.seed = seed

    def fit(self, X, y=None):
        """
        Fit the SlicedWassersteinDistance class on a list of signed measures: signed measures are projected onto the different lines. The measures themselves are then stored in numpy arrays, called **measures_**.

        Parameters:
            X (list of tuples): input signed measures.
            y (n x 1 array): signed measure labels (unused).
        """
        self.measures_ = X
        self._api = api_from_tensor(
            X[0][0] if isinstance(X, list) and len(X) > 0 else np.asarray(0.0)
        )
        return self

    def transform(self, X):
        """
        Compute all sliced Wasserstein distances between the signed measures that were stored after calling the fit() method, and a given list of (possibly different) signed measures.

        Parameters:
            X (list of tuples): input signed measures.

        Returns:
            numpy array of shape (number of measures in **measures**) x (number of measures in X): matrix of pairwise sliced Wasserstein distances.
        """
        return pairwise_signed_measure_distances(
            X,
            self.measures_,
            metric="sliced_wasserstein",
            num_directions=self.num_directions,
            scales=self.scales,
            seed=self.seed,
            n_jobs=self.n_jobs,
            api=self._api,
        )

    def __call__(self, meas1, meas2):
        """
        Apply SlicedWassersteinDistance on a single pair of signed measures and outputs the result.

        Parameters:
            meas1: ((n x D), (n)) tuple with numpy.array encoding the (finite points of the) first measure and their multiplicities. Must not contain essential points (i.e. with infinite coordinate).
            meas2: ((m x D), (m)) tuple encoding the second measure.

        Returns:
            float: sliced Wasserstein distance.
        """
        return sm_distance(
            meas1,
            meas2,
            sliced=True,
            num_directions=self.num_directions,
            seed=self.seed,
        )


class WassersteinDistance(BaseEstimator, TransformerMixin):
    """
    This is a class for computing the Wasserstein distance matrix from a list of signed measures.
    """

    def __init__(self, epsilon=1.0, ground_norm=1, n_jobs=None):
        """
        Constructor for the WassersteinDistance class.

        Parameters:
            epsilon (float): entropy regularization parameter.
            ground_norm (int): norm to use for ground metric cost.
            n_jobs (int): number of jobs to use for the computation. See :func:`pairwise_signed_measure_distances` for details.
        """
        self.epsilon = epsilon
        self.ground_norm = ground_norm
        self.n_jobs = n_jobs

    def fit(self, X, y=None):
        """
        Fit the WassersteinDistance class on a list of signed measures. The measures themselves are then stored in numpy arrays, called **measures_**.

        Parameters:
            X (list of tuples): input signed measures.
            y (n x 1 array): signed measure labels (unused).
        """
        self.measures_ = X
        return self

    def transform(self, X):
        """
        Compute all Wasserstein distances between the signed measures that were stored after calling the fit() method, and a given list of (possibly different) signed measures.

        Parameters:
            X (list of tuples): input signed measures.

        Returns:
            numpy array of shape (number of measures in **measures**) x (number of measures in X): matrix of pairwise Wasserstein distances.
        """
        return pairwise_signed_measure_distances(
            X,
            self.measures_,
            metric="wasserstein",
            epsilon=self.epsilon,
            ground_norm=self.ground_norm,
            n_jobs=self.n_jobs,
        )

    def __call__(self, meas1, meas2):
        """
        Apply WassersteinDistance on a single pair of signed measures and outputs the result.

        Parameters:
            meas1: ((n x D), (n)) tuple with numpy.array encoding the (finite points of the) first measure and their multiplicities. Must not contain essential points (i.e. with infinite coordinate).
            meas2: ((m x D), (m)) tuple encoding the second measure.

        Returns:
            float: Wasserstein distance.
        """
        return _wasserstein_distance(
            meas1, meas2, epsilon=self.epsilon, ground_norm=self.ground_norm
        )
