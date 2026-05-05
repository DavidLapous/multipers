from collections.abc import Iterable
from typing import Literal, Optional

import gudhi as gd
import numpy as np
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.spatial.distance import cdist, pdist
from tqdm import tqdm

import multipers as mp
import multipers.slicer as mps
from multipers.filtrations.density import available_kernels
from multipers.filtrations import (
    CoreDelaunay,
    DegreeRips,
    DelaunayLowerstar,
    RhomboidBifiltration,
    RipsLowerstar,
    _AlphaLowerstar,
)
from multipers.array_api import api_from_tensor


class PointCloud2FilteredComplex(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        bandwidths=None,
        masses=None,
        knns=None,
        threshold: Optional[float] = None,
        complex: Optional[
            Literal[
                "alpha",
                "rips",
                "delaunay",
                "degree-rips",
                "core-delaunay",
                "multicover",
            ]
        ] = None,
        node_degrees: Optional[Iterable[int]] = None,
        core_degrees: Optional[Iterable[int]] = None,
        multicover_k_max: Optional[int] = None,
        multicover_degree: Optional[int] = None,
        sparse: Optional[float] = None,
        kernel: available_kernels = "gaussian",
        log_density: bool = True,
        progress: bool = False,
        n_jobs: Optional[int] = None,
        fit_fraction: float = 1,
        verbose: bool = False,
    ) -> None:
        """
        Bulk computation of: 
         1. (Rips or Alpha or Delaunay) + (Density Estimation, DTM, or kNN) 1-critical 2-filtration.
         2. (Scale) + (multicover-like) multi-critical 2-filtration.

        Parameters
        ----------
         - bandwidth : real : The kernel density estimation bandwidth, or the DTM mass. If negative, it replaced by abs(bandwidth)*(radius of the dataset)
         - knns : integer list : k values for k-nearest-neighbor codensity.
         - threshold : real or None, max edge length of the rips or max alpha square of the alpha. None uses the dataset diameter.
         - complex : "alpha", "rips", "delaunay", "degree-rips", "core-delaunay", "multicover", or None. None uses Delaunay for point clouds of dimension <= 4 and Rips otherwise.
         - node_degrees : integer list : degree thresholds for degree-rips.
         - core_degrees : integer list : k-values for core-delaunay.
         - multicover_k_max : integer : maximum cover count for multicover.
         - multicover_degree : integer : homology degree for multicover.
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
        self.bandwidths = [] if bandwidths is None else bandwidths
        self.masses = [] if masses is None else masses
        self.knns = [] if knns is None else knns
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
        self._complex = complex
        self.node_degrees = node_degrees
        self.core_degrees = core_degrees
        self.multicover_k_max = multicover_k_max
        self.multicover_degree = multicover_degree
        self.threshold = threshold
        self.sparse = sparse
        self._get_sts = lambda x: Exception("Fit first")
        self._api = None
        return

    def _get_distance_quantiles_and_threshold(self, X, qs):
        ## if we dont need to compute a distance matrix
        if self._complex == "delaunay" and self.threshold in (None, -np.inf):
            self._threshold = None
            if len(qs) == 0:
                self._scale = []
                return []

        if len(qs) == 0 and self.threshold is not None and self.threshold >= 0:
            self._scale = []
            return []
        if self.progress:
            print("Estimating scale...", flush=True, end="")
        ## subsampling
        indices = np.random.choice(
            len(X), min(len(X), int(self.fit_fraction * len(X)) + 1), replace=False
        )

        try:
            from pykeops.numpy import LazyTensor
        except ImportError:
            LazyTensor = None

        def compute_max_scale(x):
            x_np = self._api.asnumpy(x)
            if LazyTensor is None:
                distances = pdist(x_np)
                return distances.max() if len(distances) > 0 else 0

            a = LazyTensor(x_np[None, :, :])
            b = LazyTensor(x_np[:, None, :])
            return np.sqrt(((a - b) ** 2).sum(2).max(1).max(0)[0])

        diameter = np.max([compute_max_scale(x) for x in (X[i] for i in indices)])
        self._scale = diameter * np.array(qs)

        if self._complex == "delaunay" and self.threshold in (None, -np.inf):
            self._threshold = None
        elif self.threshold is None or self.threshold == -np.inf:
            self._threshold = diameter
        elif self.threshold > 0:
            self._threshold = self.threshold
        else:
            self._threshold = -diameter * self.threshold

        if self.threshold is not None and self.threshold > 0 and len(self._scale) > 0:
            self._scale[self._scale > self.threshold] = self.threshold

        if self.progress:
            print(f"Done. Chosen scales {qs} are {self._scale}", flush=True)
        return self._scale

    def _is_structural_complex(self):
        return self._complex in {"degree-rips", "core-delaunay", "multicover"}

    def _get_structural_complex(self, x):
        if self._complex == "degree-rips":
            return DegreeRips(
                points=x,
                ks=self.node_degrees,
                threshold_radius=self._threshold,
                verbose=self.verbose,
            )
        if self._complex == "core-delaunay":
            return CoreDelaunay(
                points=x,
                ks=self.core_degrees,
                max_alpha_square=(
                    self._threshold if self.threshold not in (None, -np.inf) else np.inf
                ),
                verbose=self.verbose,
            )
        if self._complex == "multicover":
            return RhomboidBifiltration(
                x=x,
                k_max=self.multicover_k_max,
                degree=self.multicover_degree,
                verbose=self.verbose,
            )
        raise ValueError(f"Invalid structural complex {self._complex}")

    def _get_codensity_complex(self, x, codensity):
        match self._complex:
            case "rips":
                return RipsLowerstar(
                    points=x,
                    function=codensity,
                    threshold_radius=self._threshold,
                    sparse=self.sparse,
                )
            case "alpha":
                return _AlphaLowerstar(
                    points=x,
                    function=codensity,
                    threshold_radius=self._threshold,
                )
            case "delaunay":
                return DelaunayLowerstar(
                    points=x,
                    function=codensity,
                    threshold_radius=self._threshold,
                    verbose=self.verbose,
                )
            case _:
                raise ValueError(f"Invalid complex {self._complex}")

    def _get_sts_all(self, x):
        if self._is_structural_complex():
            return [self._get_structural_complex(x)]
        codensities = self._get_codensities(x, x)
        return [self._get_codensity_complex(x, codensity) for codensity in codensities]

    def _precompile_codensities(self, X):
        if self._is_structural_complex():
            return
        x_precompile = self._precompile_sample(X)
        self._get_codensities(x_precompile, x_precompile)

    def _get_codensities(self, x_fit, x_sample):
        from multipers.filtrations.density import DTM, KDE, KNNmean

        x_fit = self._api.astensor(x_fit)
        x_sample = self._api.astensor(x_sample, dtype=x_fit.dtype)
        if len(self.bandwidths) == 0:
            codensities_kde = self._api.empty((0, len(x_sample)))
        else:
            codensities_kde = self._api.stack(
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
            codensities_dtm = self._api.empty((0, len(x_sample)))
        else:
            codensities_dtm = (
                DTM(masses=self.masses)
                .fit(x_fit)
                .score_samples(x_sample)
                .reshape(len(self.masses), len(x_sample))
            )
        if len(self.knns) == 0:
            codensities_knn = self._api.empty((0, len(x_sample)))
        else:
            codensities_knn = self._api.stack(
                [
                    KNNmean(k=knn, api=self._api).fit(x_fit).score_samples(x_sample)
                    for knn in self.knns
                ]
            )
        return self._api.cat([codensities_kde, codensities_dtm, codensities_knn])

    def _precompile_sample(self, X):
        size = 2
        if len(self.knns) > 0:
            size = max(size, max(self.knns))
        return X[0][: min(len(X[0]), size)]

    def _define_sts(self):
        valid_complexes = {
            "rips",
            "alpha",
            "delaunay",
            "degree-rips",
            "core-delaunay",
            "multicover",
        }
        if self._complex not in valid_complexes:
            raise ValueError(
                f"Invalid complex {self._complex}. "
                f"Possible choices are {sorted(valid_complexes)}."
            )
        self._get_sts = self._get_sts_all

    def _define_bandwidths(self, X):
        qs = [q for q in [*-np.asarray(self.bandwidths)] if 0 <= q <= 1]
        self._get_distance_quantiles_and_threshold(X, qs=qs)
        self._bandwidths = np.array(self.bandwidths)
        count = 0
        for i in range(len(self._bandwidths)):
            if self.bandwidths[i] < 0:
                self._bandwidths[i] = self._scale[count]
                count += 1

    def _validate_complex_parameters(self):
        has_codensity = (
            len(self.bandwidths) > 0 or len(self.masses) > 0 or len(self.knns) > 0
        )
        if self._complex in {"degree-rips", "core-delaunay", "multicover"} and has_codensity:
            raise ValueError(
                f"`{self._complex}` is already a bifiltration and cannot be combined "
                "with `bandwidths`, `masses`, or `knns`."
            )
        if self._complex == "degree-rips" and self.node_degrees is None:
            raise ValueError("`node_degrees` is required for complex='degree-rips'.")
        if self._complex == "core-delaunay" and self.core_degrees is None:
            raise ValueError("`core_degrees` is required for complex='core-delaunay'.")
        if self._complex == "multicover":
            if self.multicover_k_max is None:
                raise ValueError("`multicover_k_max` is required for complex='multicover'.")
            if self.multicover_degree is None:
                raise ValueError("`multicover_degree` is required for complex='multicover'.")

    def _validate_knns(self, X):
        if len(self.knns) == 0:
            return
        knns = np.asarray(self.knns, dtype=int)
        if np.any(knns < 1):
            raise ValueError("All kNN parameters should be positive integers.")
        max_size = min(len(x) for x in X)
        if np.max(knns) > max_size:
            raise ValueError("All kNN parameters should be at most the sample size.")

    def fit(self, X: np.ndarray | list, y=None):
        # self.bandwidth = "silverman" ## not good, as is can make bandwidth not constant
        self._api = api_from_tensor(X[0])
        self._complex = self.complex or ("delaunay" if X[0].shape[1] <= 4 else "rips")
        self._validate_complex_parameters()
        self._validate_knns(X)
        self._define_sts()
        self._define_bandwidths(X)
        self._precompile_codensities(X)
        return self

    def transform(self, X):
        self._validate_knns(X)
        self._precompile_codensities(X)
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
        bandwidths=None,
        masses=None,
        knns=None,
        threshold: float = np.inf,
        complex: Optional[
            Literal[
                "alpha",
                "rips",
                "delaunay",
                "degree-rips",
                "core-delaunay",
                "multicover",
            ]
        ] = None,
        node_degrees: Optional[Iterable[int]] = None,
        core_degrees: Optional[Iterable[int]] = None,
        multicover_k_max: Optional[int] = None,
        multicover_degree: Optional[int] = None,
        sparse: float | None = None,
        kernel: available_kernels = "gaussian",
        log_density: bool = True,
        progress: bool = False,
        n_jobs: Optional[int] = None,
        fit_fraction: float = 1,
        verbose: bool = False,
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
        num_collapses: int | Literal["full"] | None = None,
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
        if not X:
            raise ValueError("FilteredComplexPreprocess requires non-empty input")

        first = X[0]
        while isinstance(first, (list, tuple)):
            if len(first) == 0:
                raise ValueError("Cannot infer complex type from empty container")
            first = first[0]

        self._is_st = mp.simplex_tree_multi.is_simplextree_multi(first)
        if not self._is_st and not mp.slicer.is_slicer(first, allow_minpres=False):
            raise ValueError(
                "FilteredComplexPreprocess expects SimplexTreeMulti or Slicer inputs."
            )

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
        if self._is_st and self.num_collapses is not None:
            if self.num_collapses == "full":
                cplx.collapse_edges(full=True)
            else:
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

    def _process(self, cplx):
        if mp.simplex_tree_multi.is_simplextree_multi(cplx) or mp.slicer.is_slicer(
            cplx, allow_minpres=False
        ):
            return self._process_complex(cplx)
        if isinstance(cplx, tuple):
            return tuple(self._process(sub_item) for sub_item in cplx)
        if isinstance(cplx, list):
            return [self._process(sub_item) for sub_item in cplx]
        return cplx

    def transform(self, X):
        return Parallel(backend="threading", n_jobs=self.n_jobs)(
            delayed(self._process)(entry) for entry in X
        )
