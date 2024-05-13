# This code was written by Mathieu Carrière.

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import pairwise_distances, pairwise_kernels
from joblib import Parallel, delayed


def _pairwise(fallback, skipdiag, X, Y, metric, n_jobs):
    if Y is not None:
        return fallback(X, Y, metric=metric, n_jobs=n_jobs)
    triu = np.triu_indices(len(X), k=skipdiag)
    tril = (triu[1], triu[0])
    par = Parallel(n_jobs=n_jobs, prefer="threads")
    d = par(delayed(metric)([triu[0][i]], [triu[1][i]])
            for i in range(len(triu[0])))
    m = np.empty((len(X), len(X)))
    m[triu] = d
    m[tril] = d
    if skipdiag:
        np.fill_diagonal(m, 0)
    return m


def _sklearn_wrapper(metric, X, Y, **kwargs):
    """
    This function is a wrapper for any metric between two signed measures that takes two numpy arrays of shapes (nxD) and (mxD) as arguments.
    """
    if Y is None:

        def flat_metric(a, b):
            return metric(X[int(a[0])], X[int(b[0])], **kwargs)
    else:

        def flat_metric(a, b):
            return metric(X[int(a[0])], Y[int(b[0])], **kwargs)

    return flat_metric


def _compute_signed_measure_parts(X):
    """
    This is a function for separating the positive and negative points of a list of signed measures. This function can be used as a preprocessing step in order to speed up the running time for computing all pairwise (sliced) Wasserstein distances on a list of signed measures.

    Parameters:
        X (list of n tuples): list of signed measures.

    Returns:
        list of n pairs of numpy arrays of shape (num x dimension): list of positive and negative signed measures.
    """
    XX = []
    for C, M in X:
        pos_idxs = np.argwhere(M > 0).ravel()
        neg_idxs = np.setdiff1d(np.arange(len(M)), pos_idxs)
        XX.append(
            [
                np.repeat(C[pos_idxs], M[pos_idxs], axis=0),
                np.repeat(C[neg_idxs], -M[neg_idxs], axis=0),
            ]
        )
    return XX


def _compute_signed_measure_projections(X, num_directions, scales):
    """
    This is a function for projecting the points of a list of signed measures onto a fixed number of lines sampled uniformly. This function can be used as a preprocessing step in order to speed up the running time for computing all pairwise sliced Wasserstein distances on a list of signed measures.

    Parameters:
        X (list of n tuples): list of signed measures.
        num_directions (int): number of lines evenly sampled from [-pi/2,pi/2] in order to approximate and speed up the distance computation.
        scales (array of shape D): scales associated to the dimensions.

    Returns:
        list of n pairs of numpy arrays of shape (num x num_directions): list of positive and negative projected signed measures.
    """
    dimension = X[0][0].shape[1]
    np.random.seed(42)
    thetas = np.random.normal(0, 1, [num_directions, dimension])
    lines = (thetas / np.linalg.norm(thetas, axis=1)[:, None]).T
    weights = (
        np.linalg.norm(np.multiply(scales[:, None], lines), axis=0)
        if scales is not None
        else np.ones(num_directions)
    )
    XX = []
    for C, M in X:
        pos_idxs = np.argwhere(M > 0).ravel()
        neg_idxs = np.setdiff1d(np.arange(len(M)), pos_idxs)
        XX.append(
            [
                np.matmul(np.repeat(C[pos_idxs], M[pos_idxs], axis=0), lines),
                np.matmul(np.repeat(C[neg_idxs], -M[neg_idxs], axis=0), lines),
                weights,
            ]
        )
    return XX


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
    XX = np.reshape(np.arange(len(X)), [-1, 1])
    YY = None if Y is None or Y is X else np.reshape(
        np.arange(len(Y)), [-1, 1])
    if metric == "sliced_wasserstein":
        Xproj = _compute_signed_measure_projections(X, **kwargs)
        Yproj = None if Y is None else _compute_signed_measure_projections(
            Y, **kwargs)
        return _pairwise(
            pairwise_distances,
            True,
            XX,
            YY,
            metric=_sklearn_wrapper(
                _sliced_wasserstein_distance_on_projections, Xproj, Yproj
            ),
            n_jobs=n_jobs,
        )
    elif metric == "wasserstein":
        Xproj = _compute_signed_measure_parts(X)
        Yproj = None if Y is None else _compute_signed_measure_parts(Y)
        return _pairwise(
            pairwise_distances,
            True,
            XX,
            YY,
            metric=_sklearn_wrapper(
                _wasserstein_distance_on_parts(**kwargs), Xproj, Yproj
            ),
            n_jobs=n_jobs,
        )
    else:
        return _pairwise(
            pairwise_distances,
            True,
            XX,
            YY,
            metric=_sklearn_wrapper(metric, X, Y, **kwargs),
            n_jobs=n_jobs,
        )


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
        meas1_plus, meas1_minus = meas1[0], meas1[1]
        meas2_plus, meas2_minus = meas2[0], meas2[1]
        num_pts = len(meas1_plus) + len(meas2_minus)
        meas_t1 = np.vstack([meas1_plus, meas2_minus])
        meas_t2 = np.vstack([meas2_plus, meas1_minus])
        import ot

        if epsilon > 0:
            wass = ot.sinkhorn2(
                1 / num_pts * np.ones(num_pts),
                1 / num_pts * np.ones(num_pts),
                pairwise_distances(
                    meas_t1, meas_t2, metric="minkowski", p=ground_norm),
                epsilon,
            )
            return wass[0]
        else:
            wass = ot.lp.emd2(
                [],
                [],
                np.ascontiguousarray(
                    pairwise_distances(
                        meas_t1, meas_t2, metric="minkowski", p=ground_norm
                    ),
                    dtype=np.float64,
                ),
            )
            return wass

    return metric


def _sliced_wasserstein_distance_on_projections(meas1, meas2, scales=None):
    """
    This is a function for computing the sliced Wasserstein distance between two signed measures that have already been projected onto some lines. It simply amounts to comparing the sorted projections with the 1-norm, and averaging over the lines. See http://proceedings.mlr.press/v70/carriere17a.html for more details.

    Parameters:
        meas1: pair of (n x number_of_lines) numpy.arrays containing the projected points of the positive and negative parts of the first measure.
        meas2: pair of (m x number_of_lines) numpy.arrays containing the projected points of the positive and negative parts of the second measure.
        scales (array of shape D): scales associated to the dimensions.

    Returns:
        float: the sliced Wasserstein distance between the projected signed measures.
    """
    # assert np.array_equal(  meas1[2], meas2[2]  )
    weights = meas1[2]
    meas1_plus, meas1_minus = meas1[0], meas1[1]
    meas2_plus, meas2_minus = meas2[0], meas2[1]
    A = np.sort(np.vstack([meas1_plus, meas2_minus]), axis=0)
    B = np.sort(np.vstack([meas2_plus, meas1_minus]), axis=0)
    L1 = np.sum(np.abs(A - B), axis=0)
    return np.mean(np.multiply(L1, weights))


def _sliced_wasserstein_distance(meas1, meas2, num_directions, scales=None):
    """
    This is a function for computing the sliced Wasserstein distance from two signed measures. The Sliced Wasserstein distance is computed by projecting the signed measures onto lines, comparing the projections with the 1-norm, and finally averaging over the lines. See http://proceedings.mlr.press/v70/carriere17a.html for more details.

    Parameters:
        meas1: ((n x D), (n)) tuple with numpy.array encoding the (finite points of the) first measure and their multiplicities. Must not contain essential points (i.e. with infinite coordinate).
        meas2: ((m x D), (m)) tuple encoding the second measure.
        num_directions (int): number of lines evenly sampled from [-pi/2,pi/2] in order to approximate and speed up the distance computation.
        scales (array of shape D): scales associated to the dimensions.

    Returns:
        float: the sliced Wasserstein distance between signed measures.
    """
    C1, M1 = meas1[0], meas1[1]
    C2, M2 = meas2[0], meas2[1]
    dimension = C1.shape[1]
    C1_plus_idxs, C2_plus_idxs = (
        np.argwhere(M1 > 0).ravel(),
        np.argwhere(M2 > 0).ravel(),
    )
    C1_minus_idxs, C2_minus_idxs = (
        np.setdiff1d(np.arange(len(M1)), C1_plus_idxs),
        np.setdiff1d(np.arange(len(M2)), C2_plus_idxs),
    )
    np.random.seed(42)
    thetas = np.random.normal(0, 1, [num_directions, dimension])
    lines = (thetas / np.linalg.norm(thetas, axis=1)[:, None]).T
    weights = (
        np.linalg.norm(np.multiply(scales[:, None], lines), axis=0)
        if scales is not None
        else np.ones(num_directions)
    )
    approx1 = np.matmul(
        np.vstack(
            [
                np.repeat(C1[C1_plus_idxs], M1[C1_plus_idxs], axis=0),
                np.repeat(C2[C2_minus_idxs], -M2[C2_minus_idxs], axis=0),
            ]
        ),
        lines,
    )
    approx2 = np.matmul(
        np.vstack(
            [
                np.repeat(C2[C2_plus_idxs], M2[C2_plus_idxs], axis=0),
                np.repeat(C1[C1_minus_idxs], -M1[C1_minus_idxs], axis=0),
            ]
        ),
        lines,
    )
    A = np.sort(approx1, axis=0)
    B = np.sort(approx2, axis=0)
    L1 = np.sum(np.abs(A - B), axis=0)
    return np.mean(np.multiply(L1, weights))


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
    C1, M1 = meas1[0], meas1[1]
    C2, M2 = meas2[0], meas2[1]
    C1_plus_idxs, C2_plus_idxs = (
        np.argwhere(M1 > 0).ravel(),
        np.argwhere(M2 > 0).ravel(),
    )
    C1_minus_idxs, C2_minus_idxs = (
        np.setdiff1d(np.arange(len(M1)), C1_plus_idxs),
        np.setdiff1d(np.arange(len(M2)), C2_plus_idxs),
    )
    approx1 = np.vstack(
        [
            np.repeat(C1[C1_plus_idxs], M1[C1_plus_idxs], axis=0),
            np.repeat(C2[C2_minus_idxs], -M2[C2_minus_idxs], axis=0),
        ]
    )
    approx2 = np.vstack(
        [
            np.repeat(C2[C2_plus_idxs], M2[C2_plus_idxs], axis=0),
            np.repeat(C1[C1_minus_idxs], -M1[C1_minus_idxs], axis=0),
        ]
    )
    num_pts = len(approx1)
    import ot

    if epsilon > 0:
        wass = ot.sinkhorn2(
            1 / num_pts * np.ones(num_pts),
            1 / num_pts * np.ones(num_pts),
            pairwise_distances(
                approx1, approx2, metric="minkowski", p=ground_norm),
            epsilon,
        )
        return wass[0]
    else:
        wass = ot.lp.emd2(
            1 / num_pts * np.ones(num_pts),
            1 / num_pts * np.ones(num_pts),
            pairwise_distances(
                approx1, approx2, metric="minkowski", p=ground_norm),
        )
        return wass


class SlicedWassersteinDistance(BaseEstimator, TransformerMixin):
    """
    This is a class for computing the sliced Wasserstein distance matrix from a list of signed measures. The Sliced Wasserstein distance is computed by projecting the signed measures onto lines, comparing the projections with the 1-norm, and finally integrating over all possible lines. See http://proceedings.mlr.press/v70/carriere17a.html for more details.
    """

    def __init__(self, num_directions=10, scales=None, n_jobs=None):
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

    def fit(self, X, y=None):
        """
        Fit the SlicedWassersteinDistance class on a list of signed measures: signed measures are projected onto the different lines. The measures themselves are then stored in numpy arrays, called **measures_**.

        Parameters:
            X (list of tuples): input signed measures.
            y (n x 1 array): signed measure labels (unused).
        """
        self.measures_ = X
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
            n_jobs=self.n_jobs,
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
        return _sliced_wasserstein_distance(
            meas1, meas2, num_directions=self.num_directions, scales=self.scales
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
