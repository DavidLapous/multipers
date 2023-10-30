## This code was written by Luis Scoccola
import numpy as np
from scipy.sparse import coo_array
from scipy.ndimage import convolve1d


def signed_betti(hilbert_function, threshold=False, sparse=False):
    n = len(hilbert_function.shape)
    res = np.copy(hilbert_function)
    # zero out the "end" of the Hilbert function
    if threshold:
        for dimension in range(n):
            slicer = tuple([slice(None) if i != dimension else -1 for i in range(n)])
            res[slicer] = 0
    weights = np.array([0, 1, -1], dtype=int)
    for i in range(n):
        res = convolve1d(res, weights, axis=i, mode="constant", cval=0)
    if sparse:
        return coo_array(res)
    else:
        return res

def rank_decomposition_by_rectangles(rank_invariant, threshold=False):
    # takes as input the rank invariant of an n-parameter persistence module
    #   M  :  [0, ..., s_1 - 1] x ... x [0, ..., s_n - 1]  --->  Vec
    # on a grid with dimensions of sizes s_1, ..., s_n. The input is assumed to be
    # given as a tensor of dimensions (s_1, ..., s_n, s_1, ..., s_n), so that,
    # at index [i_1, ..., i_n, j_1, ..., j_n] we have the rank of the structure
    # map M(i) -> M(j), where i = (i_1, ..., i_n) and j = (j_1, ..., j_n), and
    # i <= j, meaning that i_1 <= j_1, ..., i_n <= j_n.
    # NOTE :
    #   - About the input, we assume that, if not( i <= j ), then at index
    #     [i_1, ..., i_n, j_1, ..., j_n] we have a zero.
    #   - Similarly, the output at index [i_1, ..., i_n, j_1, ..., j_n] only
    #     makes sense when i <= j. For indices where not( i <= j ) the output
    #     may take arbitrary values and they should be ignored.
    n = len(rank_invariant.shape) // 2
    if threshold:
        rank_invariant = rank_invariant.copy()
        # print(rank_invariant)
        # zero out the "end"
        for dimension in range(n):
            slicer = tuple(
                [slice(None) for _ in range(n)]
                + [slice(None) if i != dimension else -1 for i in range(n)]
            )
            rank_invariant[slicer] = 0
        # print(rank_invariant)
    to_flip = tuple(range(n, 2 * n))
    return np.flip(signed_betti(np.flip(rank_invariant, to_flip)), to_flip)