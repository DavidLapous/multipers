import warnings
import sys
import os

import multipers as mp
import numpy as np
import pytest
from multipers.data import three_annulus
from multipers.filtrations import CoreDelaunay
from multipers.tests import assert_sm

from multipers.io import _multi_critical_from_slicer


pytestmark = pytest.mark.skipif(
    sys.platform.startswith("win"),
    reason="multi_critical/ops is unavailable on Windows",
)

if (
    os.environ.get("GITHUB_ACTIONS") == "true"
    and sys.platform == "darwin"
):
    pytestmark = [
        pytestmark,
        pytest.mark.skip(
            reason=(
                "Github action failing?"
            )
        ),
    ]


def _tiny_slicer():
    st = mp.SimplexTreeMulti(num_parameters=2)
    st.insert([0], [0.0, 0.0])
    st.insert([1], [1.0, 0.0])
    st.insert([0, 1], [1.0, 1.0])
    return mp.Slicer(st)


def _skip_if_fallback_was_used(caught):
    if any(
        "Falling back to external multi_critical binary" in str(w.message)
        for w in caught
    ):
        pytest.skip("multi_critical in-memory interface unavailable in this build")


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


def test_multi_critical_in_memory_resolution_and_reduce_degree():
    slicer = _tiny_slicer()

    out = _run_or_skip(
        lambda: _multi_critical_from_slicer(slicer, reduce=False, clear=True)
    )
    assert mp.slicer.is_slicer(out, allow_minpres=False)
    assert out.num_parameters == 2

    reduced = _run_or_skip(
        lambda: _multi_critical_from_slicer(slicer, reduce=True, degree=0, clear=True)
    )
    assert reduced.is_minpres
    assert reduced.minpres_degree == 0


def test_multi_critical_in_memory_reduce_all_marks_degrees():
    slicer = _tiny_slicer()

    reduced_all = _run_or_skip(
        lambda: _multi_critical_from_slicer(
            slicer, reduce=True, degree=None, clear=True
        )
    )
    assert isinstance(reduced_all, tuple)
    assert len(reduced_all) > 0
    for degree, out in enumerate(reduced_all):
        assert out.is_minpres
        assert out.minpres_degree == degree


def test_multi_critical_direct_bridge_without_temp_scc(monkeypatch):
    slicer = _tiny_slicer()
    _run_or_skip(lambda: _multi_critical_from_slicer(slicer, reduce=False, clear=True))

    import tempfile

    def _no_tmpdir(*args, **kwargs):
        raise AssertionError(
            "Direct multi_critical bridge should not use temporary SCC files"
        )

    monkeypatch.setattr(tempfile, "TemporaryDirectory", _no_tmpdir)
    out = _multi_critical_from_slicer(slicer, reduce=False, clear=True)
    assert mp.slicer.is_slicer(out, allow_minpres=False)


def test_one_criticalify_core_delaunay_regression_small():
    np.random.seed(0)
    X = three_annulus(50, 50)
    k_max = len(X) // 10
    ks = np.unique(np.linspace(1, k_max, num=10, dtype=int)).tolist()
    st = CoreDelaunay(points=X, beta=1.5, ks=ks)
    slicer = mp.Slicer(st)

    out = _run_or_skip(lambda: mp.ops.one_criticalify(slicer))
    assert mp.slicer.is_slicer(out, allow_minpres=False)
    assert not out.is_kcritical


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
    st = CoreDelaunay(points=points, beta=1.5, ks=list(range(1, 10)))
    return mp.Slicer(
        st,
        vineyard=False,
        backend="matrix",
        filtration_container="contiguous",
    )


def test_one_criticalify_degree_rips_keeps_input_dimensions_small():
    slicer = _small_degree_rips_slicer()
    assert slicer.num_parameters == 2

    input_dims = tuple(np.unique(slicer.get_dimensions()))
    out = _run_or_skip(lambda: mp.ops.one_criticalify(slicer))
    out_dims = np.unique(out.get_dimensions())
    assert np.array_equal(np.unique(slicer.get_dimensions()), input_dims)
    assert np.all(np.isin(input_dims, out_dims))


def test_one_criticalify_degree_rips_degree_force_resolution_profiles_small():
    slicer = _small_degree_rips_slicer()

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
        assert tuple(np.unique(out_degree_res.get_dimensions())) == (
            degree,
            degree + 1,
            degree + 2,
        )


def test_one_criticalify_degree_rips_reduce_outputs_profiles_small():
    slicer = _small_degree_rips_slicer()

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


def test_one_criticalify_degree_rips_hilbert_consistency_force_resolution_small():
    slicer = _small_degree_rips_slicer()
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
