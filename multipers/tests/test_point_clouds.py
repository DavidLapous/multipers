import numpy as np

import multipers as mp
import multipers.ml.point_clouds as mmp
from multipers.tests import assert_sm

mp.simplex_tree_multi.SAFE_CONVERSION = True
np.random.seed(0)


def test_throw_test():
    pts = np.array([[1, 1], [2, 2]], dtype=np.float32)
    st = mmp.PointCloud2SimplexTree(masses=[0.1]).fit_transform([pts])[0][0]
    assert isinstance(st, mp.simplex_tree_multi.SimplexTreeMulti_type)
    st = mmp.PointCloud2SimplexTree(bandwidths=[-0.1], complex="alpha").fit_transform(
        [pts]
    )[0][0]
    assert isinstance(st, mp.simplex_tree_multi.SimplexTreeMulti_type)
    st1, st2 = mmp.PointCloud2SimplexTree(bandwidths=[0.1], masses=[0.1]).fit_transform(
        [pts]
    )[0]
    assert isinstance(st1, mp.simplex_tree_multi.SimplexTreeMulti_type)
    assert isinstance(st2, mp.simplex_tree_multi.SimplexTreeMulti_type)
    ## ensures it doesn't throw
    assert isinstance(
        st.persistence_approximation(),
        mp.multiparameter_module_approximation.PyModule_type,
    )
    assert mp.signed_measure(st, degree=None, invariant="euler")[0][0].ndim == 2
    assert mp.signed_measure(st, degree=0, invariant="hilbert")[0][0].ndim == 2
    assert mp.signed_measure(st, degree=0, invariant="rank")[0][0].ndim == 2
    assert mp.signed_measure(st, degree=0, invariant="rank")[0][0].shape[1] == 4


# def test_1():
#     import multipers.data
#
#     pts = mp.data.noisy_annulus(20, 20)
#     st = mmp.PointCloud2SimplexTree(bandwidths=[-0.5], expand_dim=2).fit_transform(
#         [pts]
#     )[0][0]
#     F = st.get_filtration_grid()
#     for invariant in ["hilbert", "rank"]:
#         sm1 = mp.signed_measure(
#             st.grid_squeeze(F), degree=1, invariant=invariant, mass_default=None
#         )
#         st.collapse_edges(-2, ignore_warning=True)
#         s = mp.Slicer(st, dtype=np.float64)
#         mp.io._init_external_softwares()
#         if mp.io.pathes["mpfree"] is None:
#             from warnings import warn
#
#             warn("Could not find mpfree, skipping this test")
#         else:
#             s = s.minpres(degree=1).grid_squeeze(F)
#         sm2 = mp.signed_measure(s, invariant=invariant, degree=1)
#         assert_sm(
#             sm1, sm2, exact=False, max_error=0.1
#         )  ## TODO : Fix this. Error is too large (grid not the same ? )
