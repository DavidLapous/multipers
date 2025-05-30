import platform

import numpy as np
import pytest

import multipers as mp

np.random.seed(0)


@pytest.mark.skipif(
    platform.system() == "Windows",
    reason="Detected windows. Pykeops is not compatible with windows yet. Skipping this ftm.",
)
def test_throw_test():
    import multipers.ml.point_clouds as mmp

    pts = np.array([[1, 1], [2, 2]], dtype=np.float32)
    st = mmp.PointCloud2FilteredComplex(masses=[0.1]).fit_transform([pts])[0][0]
    assert isinstance(st, mp.simplex_tree_multi.SimplexTreeMulti_type)
    st = mmp.PointCloud2FilteredComplex(
        bandwidths=[-0.1], complex="alpha"
    ).fit_transform([pts])[0][0]
    assert isinstance(st, mp.simplex_tree_multi.SimplexTreeMulti_type)

    st = mmp.PointCloud2FilteredComplex(
        masses=[0.1], complex="alpha", output_type="slicer_novine"
    ).fit_transform([pts])[0][0]
    assert isinstance(st, mp.slicer.Slicer_type)
    assert st.is_vine is False

    st = mmp.PointCloud2FilteredComplex(
        masses=[0.1], complex="alpha", output_type="slicer"
    ).fit_transform([pts])[0][0]
    assert isinstance(st, mp.slicer.Slicer_type)
    assert st.is_vine

    st1, st2 = mmp.PointCloud2FilteredComplex(
        bandwidths=[0.1], masses=[0.1]
    ).fit_transform([pts])[0]
    assert isinstance(st1, mp.simplex_tree_multi.SimplexTreeMulti_type)
    assert isinstance(st2, mp.simplex_tree_multi.SimplexTreeMulti_type)
    ## ensures it doesn't throw
    assert isinstance(
        mp.module_approximation(st),
        mp.multiparameter_module_approximation.PyModule_type,
    )
    assert mp.signed_measure(st, degree=None, invariant="euler")[0][0].ndim == 2
    assert mp.signed_measure(st, degree=0, invariant="hilbert")[0][0].ndim == 2
    assert mp.signed_measure(st, degree=0, invariant="rank")[0][0].ndim == 2
    assert mp.signed_measure(st, degree=0, invariant="rank")[0][0].shape[1] == 4
