import warnings
import sys

import multipers as mp
import numpy as np
import pytest
from multipers.data import three_annulus
from multipers.filtrations import CoreDelaunay, DegreeRips
from multipers.tests import assert_sm

pytestmark = pytest.mark.skipif(
    sys.platform.startswith("win"),
    reason="multi_critical/ops is unavailable on Windows",
)


def _skip_if_fallback_was_used(caught):
    if any(
        "Falling back to external multi_critical binary" in str(w.message)
        for w in caught
    ):
        pytest.skip("multi_critical interface unavailable in this build")


def _run_or_skip(callable_):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            out = callable_()
        except ValueError as e:
            if "Did not find multi_critical" in str(e):
                pytest.skip("multi_critical backend unavailable in this environment")
            raise
    _skip_if_fallback_was_used(caught)
    return out


def _small_core_delaunay_slicer():
    np.random.seed(0)
    X = three_annulus(50, 50)
    k_max = len(X) // 10
    ks = np.unique(np.linspace(1, k_max, num=10, dtype=int)).tolist()
    st = CoreDelaunay(points=X, beta=1.5, ks=ks)
    return mp.Slicer(st)


def test_one_criticalify_core_delaunay_random_points_regression():
    np.random.seed(0)
    points = np.random.uniform(size=(200, 2))
    st = CoreDelaunay(points=points, ks=np.arange(1, 20).tolist())
    slicer = mp.Slicer(st)

    out = _run_or_skip(lambda: mp.ops.one_criticalify(slicer))
    assert mp.slicer.is_slicer(out, allow_minpres=False)
    assert not out.is_kcritical


def _small_degree_rips_slicer():
    np.random.seed(0)
    points = three_annulus(50, 0)
    st = DegreeRips(points=points, ks=list(range(1, 10)), squeeze=False)
    return mp.Slicer(st)


@pytest.fixture(params=["core_delaunay", "degree_rips"])
def small_slicer(request):
    if request.param == "core_delaunay":
        return request.param, _small_core_delaunay_slicer()
    return request.param, _small_degree_rips_slicer()


def test_one_criticalify_keeps_input_dimensions_small(small_slicer):
    kind, slicer = small_slicer
    assert slicer.num_parameters == 2

    input_dims = tuple(np.unique(slicer.get_dimensions()))
    out = _run_or_skip(lambda: mp.ops.one_criticalify(slicer))
    out_dims = np.unique(out.get_dimensions())
    assert np.array_equal(np.unique(slicer.get_dimensions()), input_dims)
    assert np.all(np.isin(input_dims, out_dims))
    assert kind in {"core_delaunay", "degree_rips"}


def test_one_criticalify_degree_force_resolution_profiles_small(small_slicer):
    kind, slicer = small_slicer

    for degree in (0, 1):
        out_degree = _run_or_skip(
            lambda: mp.ops.one_criticalify(
                slicer,
                degree=degree,
                force_resolution=False,
            )
        )
        assert tuple(np.unique(out_degree.get_dimensions())) == (degree, degree + 1)

        out_degree_res = _run_or_skip(
            lambda: mp.ops.one_criticalify(
                slicer,
                degree=degree,
                force_resolution=True,
            )
        )
        if degree == 0:
            assert tuple(np.unique(out_degree_res.get_dimensions())) == (0, 1, 2)
        elif kind == "core_delaunay":
            assert tuple(np.unique(out_degree_res.get_dimensions())) == (1, 2, 3)
        else:
            assert tuple(np.unique(out_degree_res.get_dimensions())) == (1, 2)


def test_one_criticalify_reduce_outputs_profiles_small(small_slicer):
    _, slicer = small_slicer

    out_all = _run_or_skip(
        lambda: mp.ops.one_criticalify(
            slicer,
            reduce=True,
            force_resolution=False,
        )
    )
    assert isinstance(out_all, tuple)
    for i, block in enumerate(out_all):
        block_dims = np.unique(block.get_dimensions())
        if block_dims.size == 0:
            continue
        assert np.array_equal(block_dims, [i, i + 1])

    out_all_res = _run_or_skip(
        lambda: mp.ops.one_criticalify(
            slicer,
            reduce=True,
            force_resolution=True,
        )
    )
    assert isinstance(out_all_res, tuple)
    assert len(out_all_res) > 1
    assert np.array_equal(np.unique(out_all_res[1].get_dimensions()), [1, 2, 3])


def test_one_criticalify_hilbert_consistency_force_resolution_small(small_slicer):
    _, slicer = small_slicer
    out = _run_or_skip(lambda: mp.ops.one_criticalify(slicer))

    for degree in (0, 1):
        out_degree_res = _run_or_skip(
            lambda: mp.ops.one_criticalify(
                slicer,
                degree=degree,
                force_resolution=True,
            )
        )
        (sm_out,) = mp.signed_measure(out, degree=degree, invariant="hilbert")
        (sm_degree_res,) = mp.signed_measure(
            out_degree_res,
            degree=degree,
            invariant="hilbert",
        )
        assert_sm(
            sm_out,
            sm_degree_res,
            exact=False,
            max_error=1e-8,
            threshold=1,
            reg=0,
        )
