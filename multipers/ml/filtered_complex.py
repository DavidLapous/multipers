from collections.abc import Iterable
from typing import Literal, Optional

import gudhi as gd
import numpy as np
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.spatial.distance import cdist
from tqdm import tqdm

import multipers as mp
import multipers.slicer as mps
from multipers.filtrations.density import DTM, KDE, available_kernels


class PointCloud2FilteredComplex(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        bandwidths=[],
        masses=[],
        threshold: float = -np.inf,
        complex: Literal["alpha", "rips", "delaunay"] = "rips",
        sparse: Optional[float] = None,
        kernel: available_kernels = "gaussian",
        log_density: bool = True,
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
         - kernel : the kernel used for density estimation. Available ones are, e.g., "dtm", "gaussian", "exponential".
         - progress : bool, shows the calculus status
         - n_jobs : number of processes
         - fit_fraction : real, the fraction of data on which to fit
         - verbose : bool, Shows more information if true.

        Output
        ------
        A list of filtered complexes whose first parameter is a rips and the second is the codensity.
        """
        super().__init__()
        self.bandwidths = bandwidths
        self.masses = masses
        self.kernel = kernel
        self.log_density = log_density
        self.progress = progress
        self._bandwidths = np.empty((0,))
        self._threshold = np.inf
        self.n_jobs = n_jobs
        self._scale = np.empty((0,))
        self.fit_fraction = fit_fraction
        self.verbose = verbose
        self.complex = complex
        self.threshold = threshold
        self.sparse = sparse
        self._get_sts = lambda: Exception("Fit first")
        self.safe_conversion = safe_conversion
        return

    def _get_distance_quantiles_and_threshold(self, X, qs):
        ## if we dont need to compute a distance matrix
        if len(qs) == 0 and self.threshold >= 0:
            self._scale = []
            return []
        if self.progress:
            print("Estimating scale...", flush=True, end="")
        ## subsampling
        indices = np.random.choice(
            len(X), min(len(X), int(self.fit_fraction * len(X)) + 1), replace=False
        )

        def compute_max_scale(x):
            from pykeops.numpy import LazyTensor

            a = LazyTensor(x[None, :, :])
            b = LazyTensor(x[:, None, :])
            return np.sqrt(((a - b) ** 2).sum(2).max(1).min(0)[0])

        diameter = np.max([compute_max_scale(x) for x in (X[i] for i in indices)])
        self._scale = diameter * np.array(qs)

        if self.threshold == -np.inf:
            self._threshold = diameter
        elif self.threshold > 0:
            self._threshold = self.threshold
        else:
            self._threshold = -diameter * self.threshold

        if self.threshold > 0:
            self._scale[self._scale > self.threshold] = self.threshold

        if self.progress:
            print(f"Done. Chosen scales {qs} are {self._scale}", flush=True)
        return self._scale

    def _get_sts_rips(self, x):
        if self.sparse is None:
            st_init = gd.SimplexTree.create_from_array(
                cdist(x, x), max_filtration=self._threshold
            )
        else:
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

        return sts

    def _get_sts_alpha(self, x: np.ndarray, return_alpha=False):
        alpha_complex = gd.AlphaComplex(points=x)
        st = alpha_complex.create_simplex_tree(max_alpha_square=self._threshold**2)
        vertices = np.array([i for (i,), _ in st.get_skeleton(0)])
        new_points = np.asarray(
            [alpha_complex.get_point(int(i)) for i in vertices]
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
                x,
                c,
                verbose=self.verbose,
                clear=not self.verbose,
            )
            return slicer

        sts = Parallel(backend="threading", n_jobs=self.n_jobs)(
            delayed(get_st)(c) for c in codensities
        )
        return sts

    def _get_codensities(self, x_fit, x_sample):
        x_fit = np.asarray(x_fit, dtype=np.float64)
        x_sample = np.asarray(x_sample, dtype=np.float64)
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
        if len(self.masses) == 0:
            codensities_dtm = np.empty((0, len(x_sample)))
        else:
            codensities_dtm = (
                DTM(masses=self.masses)
                .fit(x_fit)
                .score_samples(x_sample)
                .reshape(len(self.masses), len(x_sample))
            )
        return np.concatenate([codensities_kde, codensities_dtm])

    def _define_sts(self):
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
                {self.complex}. Possible choises are rips, delaunay, or alpha."
                )

    def _define_bandwidths(self, X):
        qs = [q for q in [*-np.asarray(self.bandwidths)] if 0 <= q <= 1]
        self._get_distance_quantiles_and_threshold(X, qs=qs)
        self._bandwidths = np.array(self.bandwidths)
        count = 0
        for i in range(len(self._bandwidths)):
            if self.bandwidths[i] < 0:
                self._bandwidths[i] = self._scale[count]
                count += 1

    def fit(self, X: np.ndarray | list, y=None):
        # self.bandwidth = "silverman" ## not good, as is can make bandwidth not constant
        self._define_sts()
        self._define_bandwidths(X)
        # PRECOMPILE FIRST
        self._get_codensities(X[0][:2], X[0][:2])
        return self

    def transform(self, X):
        # precompile first
        # self._get_sts(X[0][:5])
        self._get_codensities(X[0][:2], X[0][:2])
        with tqdm(
            X, desc="Filling simplextrees", disable=not self.progress, total=len(X)
        ) as data:
            stss = Parallel(backend="threading", n_jobs=self.n_jobs)(
                delayed(self._get_sts)(x) for x in data
            )
        return stss


class PointCloud2SimplexTree(PointCloud2FilteredComplex):
    def __init__(
        self,
        bandwidths=[],
        masses=[],
        threshold: float = np.inf,
        complex: Literal["alpha", "rips", "delaunay"] = "rips",
        sparse: float | None = None,
        kernel: available_kernels = "gaussian",
        log_density: bool = True,
        progress: bool = False,
        n_jobs: Optional[int] = None,
        fit_fraction: float = 1,
        verbose: bool = False,
        safe_conversion: bool = False,
    ) -> None:
        stuff = locals()
        stuff.pop("self")
        keys = list(stuff.keys())
        for key in keys:
            if key.startswith("__"):
                stuff.pop(key)
        super().__init__(**stuff)
        from warnings import warn

        warn("This class is deprecated, use PointCloud2FilteredComplex instead.")


class FilteredComplexPreprocess(BaseEstimator, TransformerMixin):
    """Apply common preprocessing steps to filtered complexes.

    This transformer expects the nested structure returned by
    ``PointCloud2FilteredComplex`` (lists/tuples of SimplexTreeMulti or Slicer
    instances) and applies:

    1. Optional edge collapses when the item is a ``SimplexTreeMulti``.
    2. Optional minimal presentation reduction (typically for slicers).
    """

    def __init__(
        self,
        num_collapses: int = 0,
        reduce_degrees: Optional[Iterable[int]] = None,
        expand_dim: int | None = None,
        vineyard: Optional[bool] = None,
        pers_backend: Optional[str] = None,
        column_type: Optional[str] = None,
        dtype: Optional[np.dtype] = None,
        kcritical: Optional[bool] = None,
        filtration_container: Optional[str] = None,
        output_type: Optional[Literal["simplextree", "slicer"]] = None,
        n_jobs: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.num_collapses = num_collapses
        self.reduce_degrees = reduce_degrees
        self.expand_dim = expand_dim
        self.vineyard = vineyard
        self.pers_backend = pers_backend
        self.column_type = column_type
        self.dtype = dtype
        self.kcritical = kcritical
        self.filtration_container = filtration_container
        self.output_type = output_type
        self.n_jobs = n_jobs
        self._is_st = None

    def fit(self, X, y=None):
        self._is_st = mp.simplex_tree_multi.is_simplextree_multi(X[0])
        if not self._is_st:
            assert mp.slicer.is_slicer(X[0], allow_minpres=False)

        if self.num_collapses is not None and not self._is_st:
            raise ValueError(
                "Edge collapsing is only supported for SimplexTreeMulti inputs."
            )
        if self.output_type == "simplextree" and self.reduce_degrees is not None:
            raise ValueError(
                "Minimal presentations are not simplicial; cannot return a simplextree."
            )
        return self

    def _process_complex(self, cplx):
        if self._is_st:
            cplx.collapse_edges(num=self.num_collapses)
            if self.expand_dim is not None and self.expand_dim > 1:
                cplx.expansion(self.expand_dim)
        if self.reduce_degrees is not None:
            if self._is_st:
                cplx = mp.Slicer(
                    cplx,
                    vineyard=self.vineyard,
                    backend=self.pers_backend,
                    column_type=self.column_type,
                    dtype=self.dtype,
                    kcritical=self.kcritical,
                    filtration_container=self.filtration_container,
                )
            else:
                cplx.astype(
                    vineyard=self.vineyard,
                    backend=self.pers_backend,
                    column_type=self.column_type,
                    dtype=self.dtype,
                    kcritical=self.kcritical,
                    filtration_container=self.filtration_container,
                )
            cplx = mp.ops.minimal_presentation(
                cplx,
                degrees=self.reduce_degrees,
            )
        if self.output_type == "simplextree":
            if mp.slicer.is_slicer(cplx, allow_minpres=True):
                cplx = mp.slicer.to_simplextree(cplx)
        elif self.output_type == "slicer":
            if not mp.slicer.is_slicer(cplx, allow_minpres=True):
                cplx = mp.Slicer(
                    cplx,
                    vineyard=self.vineyard,
                    backend=self.pers_backend,
                    column_type=self.column_type,
                    dtype=self.dtype,
                    kcritical=self.kcritical,
                    filtration_container=self.filtration_container,
                )
        return cplx

    def _process(self, item):
        if mp.simplex_tree_multi.is_simplextree_multi(item) or mp.slicer.is_slicer(
            item, allow_minpres=False
        ):
            return self._process_complex(item)
        if isinstance(item, tuple):
            return tuple(self._process(sub_item) for sub_item in item)
        if isinstance(item, list):
            return [self._process(sub_item) for sub_item in item]
        return item

    def transform(self, X):
        return Parallel(backend="threading", n_jobs=self.n_jobs)(
            delayed(self._process)(entry) for entry in X
        )
