from sklearn.base import BaseEstimator, TransformerMixin, clone
import numpy as np
from typing import Iterable


# To do k folds with a distance matrix, we need to slice it into list of distances.
# k-fold usually shuffles the lists, so we need to add an identifier to each entry,
#
class DistanceMatrix2DistanceList(BaseEstimator, TransformerMixin):
    def __init__(self) -> None:
        super().__init__()

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError("X must be a matrix (ndim == 2).")
        indices = np.arange(len(X))[:, None]
        return np.hstack([indices, X])


class DistanceList2DistanceMatrix(BaseEstimator, TransformerMixin):
    def __init__(self) -> None:
        super().__init__()

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        index_list = (
            np.asarray(X[:, 0], dtype=int) + 1
        )  # shift of 1, because the first index is for indexing the pts
        return X[:, index_list]  # The distance matrix of the index_list


class DistanceMatrices2DistancesList(BaseEstimator, TransformerMixin):
    """
    Input (degree) x (distance matrix) or (axis) x (degree) x (distance matrix D)
    Output _ (D1) x opt (axis) x (degree) x (D2, , with indices first)
    """

    def __init__(self) -> None:
        super().__init__()
        self._axes = None

    def fit(self, X, y=None):
        X = np.asarray(X)
        self._axes = X.ndim == 4
        if not self._axes and X.ndim != 3:
            raise ValueError(
                " Bad input shape. Input is either (degree) x (distance matrix) or (axis) x (degree) x (distance matrix) "
            )

        return self

    def transform(self, X):
        X = np.asarray(X)
        if not ((X.ndim == 3 and not self._axes) or (X.ndim == 4 and self._axes)):
            raise ValueError(f"X shape ({X.shape}) is not valid")
        if self._axes:
            out = np.asarray(
                [
                    [
                        DistanceMatrix2DistanceList().fit_transform(M)
                        for M in matrices_in_axes
                    ]
                    for matrices_in_axes in X
                ]
            )
            return np.moveaxis(out, [2, 0, 1, 3], [0, 1, 2, 3])
        else:
            out = np.array(
                [DistanceMatrix2DistanceList().fit_transform(M) for M in X]
            )  # indices are at [:,0,Any_coord]
            # return np.moveaxis(out, 0, -1) ## indices are at [:,0,any_coord], degree axis is the last
            return np.moveaxis(out, [1, 0, 2], [0, 1, 2])

    def predict(self, X):
        return self.transform(X)


class DistancesLists2DistanceMatrices(BaseEstimator, TransformerMixin):
    """
    Input (D1) x opt (axis) x (degree) x (D2 with indices first)
    Output opt (axis) x (degree) x (distance matrix (D1,D2))
    """

    def __init__(self) -> None:
        super().__init__()
        self.train_indices = None
        self._axes = None

    def fit(self, X: np.ndarray, y=None):
        X = np.asarray(X)
        if X.ndim not in [3, 4]:
            raise ValueError("X must have 3 or 4 dimensions")
        self._axes = X.ndim == 4
        if self._axes:
            self.train_indices = np.asarray(X[:, 0, 0, 0], dtype=int)
        else:
            self.train_indices = np.asarray(X[:, 0, 0], dtype=int)
        return self

    def transform(self, X):
        X = np.asarray(X)
        if X.ndim not in [3, 4]:
            raise ValueError("X must have 3 or 4 dimensions")
        # test_indices = np.asarray(X[:,0,0], dtype=int)
        # print(X.shape, self.train_indices, test_indices, flush=1)
        # First coord of X is test indices by design, train indices have to be selected in the second coord, last one is the degree
        if self._axes:
            Y = X[:, :, :, self.train_indices + 1]
            return np.moveaxis(Y, [0, 1, 2, 3], [2, 0, 1, 3])
        else:
            Y = X[
                :, :, self.train_indices + 1
            ]  # we only keep the good indices # shift of 1, because the first index is for indexing the pts
            return np.moveaxis(
                Y, [0, 1, 2], [1, 0, 2]
            )  # we put back the degree axis first

        # # out = np.moveaxis(Y,-1,0) ## we put back the degree axis first
        # return out


class DistanceMatrix2Kernel(BaseEstimator, TransformerMixin):
    """
    Input : (degree) x (distance matrix) or (axis) x (degree) x (distance matrix) in the second case, axis HAS to be specified (meant for cross validation)
    Output : kernel of the same shape of distance matrix
    """

    def __init__(
        self,
        sigma: float | Iterable[float] = 1,
        axis: int | None = None,
        weights: Iterable[float] | float = 1,
    ) -> None:
        super().__init__()
        self.sigma = sigma
        self.axis = axis
        self.weights = weights
        # self._num_axes=None
        self._num_degrees = None

    def fit(self, X, y=None):
        if len(X) == 0:
            return self
        X_arr = np.asarray(X)
        if X_arr.ndim not in [3, 4]:
            raise ValueError("Bad input. Expected ndim 3 or 4.")
        if self.axis is None:
            if X_arr.ndim != 3 and X_arr.shape[0] != 1:
                raise ValueError("Set an axis for data with axis !")
            if X_arr.shape[0] == 1 and X_arr.ndim == 4:
                self.axis_ = 0
                self._num_degrees = len(X_arr[0])
            else:
                self.axis_ = None
                self._num_degrees = len(X_arr)
        else:
            if X_arr.ndim != 4:
                raise ValueError("Cannot choose axis from data with no axis !")
            self.axis_ = self.axis
            self._num_degrees = len(X_arr[self.axis_])
        if isinstance(self.weights, float) or isinstance(self.weights, int):
            self.weights_ = [self.weights] * self._num_degrees
        else:
            self.weights_ = list(self.weights)
        if len(self.weights_) != self._num_degrees:
            raise ValueError(
                f"Number of weights ({len(self.weights_)}) has to be the same as the number of degrees ({self._num_degrees})"
            )
        return self

    def transform(self, X) -> np.ndarray:
        if getattr(self, "axis_", None) is not None:
            X = X[self.axis_]
        # TODO : pykeops, and full pipeline w/ pykeops
        kernels = np.asarray(
            [
                np.exp(-distance_matrix / (2 * self.sigma**2)) * weight
                for distance_matrix, weight in zip(X, self.weights_)
            ]
        )
        out = np.mean(kernels, axis=0)

        return out
