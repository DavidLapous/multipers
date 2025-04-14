import numpy as np
import ot

from multipers.mma_structures import PyMultiDiagrams_type
from multipers.multiparameter_module_approximation import PyModule_type
from multipers.simplex_tree_multi import SimplexTreeMulti_type


def sm2diff(sm1, sm2, threshold=None):
    pts = sm1[0]
    dtype = pts.dtype
    if isinstance(pts, np.ndarray):

        def backend_concatenate(a, b):
            return np.concatenate([a, b], axis=0, dtype=dtype)

        def backend_tensor(x):
            return np.asarray(x, dtype=int)

    else:
        import torch

        assert isinstance(pts, torch.Tensor), "Invalid backend. Numpy or torch."

        def backend_concatenate(a, b):
            return torch.concatenate([a, b], dim=0)

        def backend_tensor(x):
            return torch.tensor(x).type(torch.int)

    pts1, w1 = sm1
    pts2, w2 = sm2
    ## TODO: optimize this
    pos_indices1 = backend_tensor(
        [i for i, w in enumerate(w1) for _ in range(w) if w > 0]
    )
    pos_indices2 = backend_tensor(
        [i for i, w in enumerate(w2) for _ in range(w) if w > 0]
    )
    neg_indices1 = backend_tensor(
        [i for i, w in enumerate(w1) for _ in range(-w) if w < 0]
    )
    neg_indices2 = backend_tensor(
        [i for i, w in enumerate(w2) for _ in range(-w) if w < 0]
    )
    x = backend_concatenate(pts1[pos_indices1], pts2[neg_indices2])
    y = backend_concatenate(pts1[neg_indices1], pts2[pos_indices2])
    if threshold is not None:
        x[x>threshold]=threshold
        y[y>threshold]=threshold
    return x, y


def sm_distance(
    sm1: tuple,
    sm2: tuple,
    reg: float = 0,
    reg_m: float = 0,
    numItermax: int = 10000,
    p: float = 1,
    threshold=None,
):
    """
    Computes the wasserstein distances between two signed measures,
    of the form
     - (pts,weights)
    with
     - pts : (num_pts, dim) float array
     - weights : (num_pts,) int array

    Regularisation:
     - sinkhorn if reg != 0
     - sinkhorn unbalanced if reg_m != 0
    """
    x, y = sm2diff(sm1, sm2, threshold=threshold)
    loss = ot.dist(
        x, y, metric="sqeuclidean", p=p
    )  # only euc + sqeuclidian are implemented in pot for the moment with torch backend # TODO : check later
    if isinstance(x, np.ndarray):
        empty_tensor = np.array([])  # uniform weights
    else:
        import torch

        assert isinstance(x, torch.Tensor), "Unimplemented backend."
        empty_tensor = torch.tensor([])  # uniform weights

    if reg == 0:
        return ot.lp.emd2(empty_tensor, empty_tensor, M=loss) * len(x)
    if reg_m == 0:
        return ot.sinkhorn2(
            a=empty_tensor, b=empty_tensor, M=loss, reg=reg, numItermax=numItermax
        )
    return ot.sinkhorn_unbalanced2(
        a=empty_tensor,
        b=empty_tensor,
        M=loss,
        reg=reg,
        reg_m=reg_m,
        numItermax=numItermax,
    )
    # return ot.sinkhorn2(a=onesx,b=onesy,M=loss,reg=reg, numItermax=numItermax)
    # return ot.bregman.empirical_sinkhorn2(x,y,reg=reg)


def estimate_matching(b1: PyMultiDiagrams_type, b2: PyMultiDiagrams_type):
    assert len(b1) == len(b2)
    from gudhi.bottleneck import bottleneck_distance

    def get_bc(b: PyMultiDiagrams_type, i: int) -> np.ndarray:
        temp = b[i].get_points()
        out = (
            np.array(temp)[:, :, 0] if len(temp) > 0 else np.empty((0, 2))
        )  # GUDHI FIX
        return out

    return max(
        (bottleneck_distance(get_bc(b1, i), get_bc(b2, i)) for i in range(len(b1)))
    )


# Functions to estimate precision
def estimate_error(
    st: SimplexTreeMulti_type,
    module: PyModule_type,
    degree: int,
    nlines: int = 100,
    verbose: bool = False,
):
    """
    Given an MMA SimplexTree and PyModule, estimates the bottleneck distance using barcodes given by gudhi.

    Parameters
    ----------
     - st:SimplexTree
        The simplextree representing the n-filtered complex. Used to define the gudhi simplextrees on different lines.
     - module:PyModule
        The module on which to estimate approximation error, w.r.t. the original simplextree st.
     - degree:int
        The homology degree to consider

    Returns
    -------
     - float:The estimation of the matching distance, i.e., the maximum of the sampled bottleneck distances.

    """
    from time import perf_counter

    parameter = 0

    def _get_bc_ST(st, basepoint, degree: int):
        """
        Slices an mma simplextree to a gudhi simplextree, and compute its persistence on the diagonal line crossing the given basepoint.
        """
        gst = st.project_on_line(
            basepoint=basepoint, parameter=parameter
        )  # we consider only the 1rst coordinate (as )
        gst.compute_persistence()
        return gst.persistence_intervals_in_dimension(degree)

    from gudhi.bottleneck import bottleneck_distance

    low, high = module.get_box()
    nfiltration = len(low)
    basepoints = np.random.uniform(low=low, high=high, size=(nlines, nfiltration))
    # barcodes from module
    print("Computing mma barcodes...", flush=1, end="") if verbose else None
    time = perf_counter()
    bcs_from_mod = module.barcodes(degree=degree, basepoints=basepoints).get_points()
    print(f"Done. {perf_counter() - time}s.") if verbose else None

    def clean(dgm):
        return np.array(
            [
                [birth[parameter], death[parameter]]
                for birth, death in dgm
                if len(birth) > 0 and birth[parameter] != np.inf
            ]
        )

    bcs_from_mod = [
        clean(dgm) for dgm in bcs_from_mod
    ]  # we only consider the 1st coordinate of the barcode
    # Computes gudhi barcodes
    from tqdm import tqdm

    bcs_from_gudhi = [
        _get_bc_ST(st, basepoint=basepoint, degree=degree)
        for basepoint in tqdm(
            basepoints, disable=not verbose, desc="Computing gudhi barcodes"
        )
    ]
    return max(
        (
            bottleneck_distance(a, b)
            for a, b in tqdm(
                zip(bcs_from_mod, bcs_from_gudhi),
                disable=not verbose,
                total=nlines,
                desc="Computing bottleneck distances",
            )
        )
    )
