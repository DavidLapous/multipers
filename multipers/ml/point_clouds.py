from typing import Literal, Optional

import gudhi as gd
import numpy as np
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import pairwise_distances
from tqdm import tqdm

import multipers as mp
import multipers.slicer as mps
from multipers.ml.convolutions import DTM, KDE


def _throw_nofit(any):
    raise Exception("Fit first")


class PointCloud2SimplexTree(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        bandwidths=[],
        masses=[],
        threshold: float = np.inf,
        complex: Literal["alpha", "rips", "delaunay"] = "rips",
        sparse: float | None = None,
        num_collapses: int | Literal["full"] = "full",
        kernel: str = "gaussian",
        log_density: bool = True,
        expand_dim: int = 1,
        progress: bool = False,
        n_jobs: Optional[int] = None,
        fit_fraction: float = 1,
        verbose: bool = False,
        safe_conversion: bool = False,
    ) -> None:
        """
        (Rips or Alpha or Delaunay) + (Density Estimation or DTM) 1-critical 2-filtration.

        Parameters
        ----------
         - bandwidth : real : The kernel density estimation bandwidth, or the DTM mass. If negative, it replaced by abs(bandwidth)*(radius of the dataset)
         - threshold : real,  max edge lenfth of the rips or max alpha square of the alpha
         - sparse : real, sparse rips (c.f. rips doc) WARNING : ONLY FOR RIPS
         - num_collapse : int, Number of edge collapses applied to the simplextrees, WARNING : ONLY FOR RIPS
         - expand_dim : int, expand the rips complex to this dimension. WARNING : ONLY FOR RIPS
         - kernel : the kernel used for density estimation. Available ones are, e.g., "dtm", "gaussian", "exponential".
         - progress : bool, shows the calculus status
         - n_jobs : number of processes
         - fit_fraction : real, the fraction of data on which to fit
         - verbose : bool, Shows more information if true.

        Output
        ------
        A list of SimplexTreeMulti whose first parameter is a rips and the second is the codensity.
        """
        super().__init__()
        self.bandwidths = bandwidths
        self.masses = masses
        self.num_collapses = num_collapses
        self.kernel = kernel
        self.log_density = log_density
        self.progress = progress
        self._bandwidths = np.empty((0,))
        self._threshold = np.inf
        self.n_jobs = n_jobs
        self._scale = np.empty((0,))
        self.fit_fraction = fit_fraction
        self.expand_dim = expand_dim
        self.verbose = verbose
        self.complex = complex
        self.threshold = threshold
        self.sparse = sparse
        self._get_sts = _throw_nofit
        self.safe_conversion = safe_conversion
        return

    def _get_distance_quantiles(self, X, qs):
        if len(qs) == 0:
            self._scale = []
            return []
        if self.progress:
            print("Estimating scale...", flush=True, end="")
        indices = np.random.choice(
            len(X), min(len(X), int(self.fit_fraction * len(X)) + 1), replace=False
        )
        # diameter = np.asarray([distance_matrix(x,x).max() for x in (X[i] for i in indices)]).max()
        diameter = np.max(
            [pairwise_distances(X=x).max() for x in (X[i] for i in indices)]
        )
        self._scale = diameter * np.asarray(qs)
        if self.threshold > 0:
            self._scale[self._scale > self.threshold] = self.threshold
        if self.progress:
            print(f"Done. Chosen scales {qs} are {self._scale}", flush=True)
        return self._scale

    def _get_sts_rips(self, x):
        st_init = gd.RipsComplex(
            points=x, max_edge_length=self._threshold, sparse=self.sparse
        ).create_simplex_tree(max_dimension=1)
        st_init = mp.simplex_tree_multi.SimplexTreeMulti(
            st_init, num_parameters=2, safe_conversion=self.safe_conversion
        )
        codensities = self._get_codensities(x_fit=x, x_sample=x)
        num_axes = codensities.shape[0]
        sts = [st_init] + [st_init.copy() for _ in range(num_axes - 1)]
        # no need to multithread here, most operations are memory
        for codensity, st_copy in zip(codensities, sts):
            # RIPS has contigus vertices, so vertices are ordered.
            st_copy.fill_lowerstar(codensity, parameter=1)

        def collapse_edges(st):
            if self.verbose:
                print("Num simplices :", st.num_simplices)
            if isinstance(self.num_collapses, int):
                st.collapse_edges(num=self.num_collapses)
                if self.verbose:
                    print(", after collapse :", st.num_simplices, end="")
            elif self.num_collapses == "full":
                st.collapse_edges(full=True)
                if self.verbose:
                    print(", after collapse :", st.num_simplices, end="")
            if self.expand_dim > 1:
                st.expansion(self.expand_dim)
                if self.verbose:
                    print(", after expansion :", st.num_simplices, end="")
            if self.verbose:
                print("")
            return st

        return Parallel(backend="threading", n_jobs=self.n_jobs)(
            delayed(collapse_edges)(st) for st in sts
        )

    def _get_sts_alpha(self, x: np.ndarray, return_alpha=False):
        alpha_complex = gd.AlphaComplex(points=x)
        st = alpha_complex.create_simplex_tree(max_alpha_square=self._threshold**2)
        vertices = np.array([i for (i,), _ in st.get_skeleton(0)])
        new_points = np.asarray(
            [alpha_complex.get_point(i) for i in vertices]
        )  # Seems to be unsafe for some reason
        # new_points = x
        st = mp.simplex_tree_multi.SimplexTreeMulti(
            st, num_parameters=2, safe_conversion=self.safe_conversion
        )
        codensities = self._get_codensities(x_fit=x, x_sample=new_points)
        num_axes = codensities.shape[0]
        sts = [st] + [st.copy() for _ in range(num_axes - 1)]
        # no need to multithread here, most operations are memory
        max_vertices = vertices.max() + 2  # +1 to be safe
        for codensity, st_copy in zip(codensities, sts):
            alligned_codensity = np.array([np.nan] * max_vertices)
            alligned_codensity[vertices] = codensity
            # alligned_codensity = np.array([codensity[i] if i in vertices else np.nan for i in range(max_vertices)])
            st_copy.fill_lowerstar(alligned_codensity, parameter=1)
        if return_alpha:
            return alpha_complex, sts
        return sts

    def _get_sts_delaunay(self, x: np.ndarray):
        codensities = self._get_codensities(x_fit=x, x_sample=x)

        def get_st(c):
            slicer = mps.from_function_delaunay(
                x, c, verbose=self.verbose, clear=not self.verbose
            )
            st = mps.to_simplextree(slicer)
            return st

        sts = Parallel(backend="threading", n_jobs=self.n_jobs)(
            delayed(get_st)(c) for c in codensities
        )
        return sts

    def _get_codensities(self, x_fit, x_sample):
        x_fit = np.asarray(x_fit, dtype=np.float32)
        x_sample = np.asarray(x_sample, dtype=np.float32)
        codensities_kde = np.asarray(
            [
                -KDE(
                    bandwidth=bandwidth, kernel=self.kernel, return_log=self.log_density
                )
                .fit(x_fit)
                .score_samples(x_sample)
                for bandwidth in self._bandwidths
            ],
        ).reshape(len(self._bandwidths), len(x_sample))
        codensities_dtm = (
            DTM(masses=self.masses)
            .fit(x_fit)
            .score_samples(x_sample)
            .reshape(len(self.masses), len(x_sample))
        )
        return np.concatenate([codensities_kde, codensities_dtm])

    def fit(self, X: np.ndarray | list, y=None):
        # self.bandwidth = "silverman" ## not good, as is can make bandwidth not constant
        match self.complex:
            case "rips":
                self._get_sts = self._get_sts_rips
            case "alpha":
                self._get_sts = self._get_sts_alpha
            case "delaunay":
                self._get_sts = self._get_sts_delaunay
            case _:
                raise ValueError(
                    f"Invalid complex \
                {self.complex}. Possible choises are rips or alpha."
                )

        qs = [
            q for q in [*-np.asarray(self.bandwidths), -self.threshold] if 0 <= q <= 1
        ]
        self._get_distance_quantiles(X, qs=qs)
        self._bandwidths = np.array(self.bandwidths)
        count = 0
        for i in range(len(self._bandwidths)):
            if self.bandwidths[i] < 0:
                self._bandwidths[i] = self._scale[count]
                count += 1
        self._threshold = self.threshold if self.threshold > 0 else self._scale[-1]

        # PRECOMPILE FIRST
        self._get_codensities(X[0][:4], X[0][:4])
        return self

    def transform(self, X):
        # precompile first
        # self._get_sts(X[0][:5])
        self._get_codensities(X[0][:4], X[0][:4])
        with tqdm(
            X, desc="Filling simplextrees", disable=not self.progress, total=len(X)
        ) as data:
            stss = Parallel(backend="threading", n_jobs=self.n_jobs)(
                delayed(self._get_sts)(x) for x in data
            )
        return stss
