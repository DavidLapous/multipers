import numpy as np
import pytest

import multipers as mp

from multipers.invariants.skyscraper import SkyscraperInvariant


def test_packed_queries():
    invariant = SkyscraperInvariant(
        x_grid=[0, 1], y_grid=[0], box=[[0, 0], [2, 2]],
        source_offsets=[0, 0, 2], slopes=[0.75, 0.25], factor_ranks=[1, 2],
        factor_group_ids=[0, 1], staircase_offsets=[0, 1, 3],
        corner_offsets=[0, 1, 2, 3], corners=[[1, 1], [2, 2], [3, 3]], metadata={},
    )
    assert invariant.filtered_rank(0, (-1, 0), (2, 2)) == 0
    assert invariant.filtered_rank(0.25, (1, 0), (1.5, 1.5)) == 2
    assert invariant.filtered_rank(0.75, (1, 0), (1.5, 1.5)) == 0
    assert invariant.filtered_rank(0.76, (1, 0), (1.5, 1.5)) == 0
    assert invariant.filtered_rank(0, (1, 0), (0, 1)) == 0
    np.testing.assert_array_equal(invariant.filtered_rank_on_grid(0.25, (1.5, 1.5)), [[0, 2]])
    assert len(invariant.at(1, 0)["staircases"]) == 2
    with pytest.raises(ValueError, match="source_offsets"):
        SkyscraperInvariant(
            x_grid=[0], y_grid=[0], box=[[0, 0], [1, 1]],
            source_offsets=[0], slopes=[], factor_ranks=[], factor_group_ids=[],
            staircase_offsets=[0], corner_offsets=[0], corners=np.empty((0, 2)), metadata={},
        )


def test_backend_stub_contract():
    from multipers import _skyscraper_interface

    assert isinstance(_skyscraper_interface.available(), bool)
    if not _skyscraper_interface.available():
        with pytest.raises(RuntimeError, match="unavailable"):
            _skyscraper_interface.require()


def test_native_bridge_accepts_nonsquare_presentation():
    from multipers import _skyscraper_interface
    from multipers.invariants import skyscraper_invariant

    if not _skyscraper_interface.available():
        pytest.skip("Skyscraper backend unavailable")
    SlicerType = mp.Slicer(return_type_only=True, dtype=np.float64)
    presentation = SlicerType(
        [[], [0], [0]],
        np.array([0, 1, 1], dtype=np.int32),
        np.array([[0, 0], [1, 0], [0, 1]], dtype=np.float64),
    )
    presentation.minpres_degree = 0
    result = _skyscraper_interface.fixed_grid(
        [presentation],
        [0, 1],
        [0, 1],
        np.array([[0, 0], [2, 2]], dtype=float),
        max_rank=2,
    )
    assert isinstance(result, SkyscraperInvariant)
    assert len(result.source_offsets) == 5
    assert result.source_offsets[-1] == len(result.slopes)
    invariant = skyscraper_invariant(presentation, grid=([0, 1], [0, 1]))
    np.testing.assert_allclose(invariant.box, [[-0.1, -0.1], [1.1, 1.1]])
    assert invariant.filtered_rank(0, (0, 0), (0, 0)) == 1
    assert invariant.metadata["degree"] == 0
    assert invariant.metadata["coordinates"] == "physical"
    assert invariant.metadata["backend_revision"] == "cea2ef8fd7dcdba24bd3c53820b18287de1308fe"

    constant_presentation = SlicerType(
        [[]],
        np.array([0], dtype=np.int32),
        np.array([[0, 0]], dtype=np.float64),
    )
    constant_presentation.minpres_degree = 0
    constant = skyscraper_invariant(constant_presentation, grid=([0], [0]))
    np.testing.assert_allclose(constant.box, [[-0.1, -0.1], [0.1, 0.1]])

    coarsened_presentation = SlicerType(
        [[], [], [0]],
        np.array([0, 0, 1], dtype=np.int32),
        np.array([[0, 0], [3, 3], [1, 0]], dtype=np.float64),
    )
    coarsened_presentation.minpres_degree = 0
    coarse_grid = ([2, 3, 4], [2, 3, 4])
    squeezed = coarsened_presentation.grid_squeeze(coarse_grid)
    assert squeezed.is_pres and not squeezed.is_minpres
    recomputed = skyscraper_invariant(
        squeezed, grid=coarse_grid, box=[[0, 0], [5, 5]]
    )
    assert recomputed.metadata["degree"] == 0


def test_fersztand_jendrysiak_paper_stable_example():
    """Reproduce Example 9/Figure 6, Computing the Skyscraper Invariant."""
    from multipers import _skyscraper_interface
    from multipers.invariants import skyscraper_invariant

    if not _skyscraper_interface.available():
        pytest.skip("Skyscraper backend unavailable")
    SlicerType = mp.Slicer(return_type_only=True, dtype=np.float64)
    module = SlicerType(
        [[], [], [0], [1], [0, 1], [1], [0]],
        np.array([0, 0, 1, 1, 1, 1, 1], dtype=np.int32),
        np.array([
            [0, 0], [0, 0],
            [0, 2], [0, 3], [1, 1], [2, 0], [3, 0],
        ], dtype=np.float64),
    )
    module.minpres_degree = 0

    invariant = skyscraper_invariant(
        module, grid=([0], [0]), box=[[0, 0], [3, 3]], max_rank=2
    )
    origin = invariant.at(0, 0)

    # Paper uses unnormalized area: 2/9. API normalizes by rectangle area 9.
    np.testing.assert_allclose(origin["slopes"], [2.0])
    np.testing.assert_array_equal(origin["factor_ranks"], [2])
    np.testing.assert_array_equal(origin["factor_group_ids"], [0])
    assert invariant.filtered_rank(2, (0, 0), (0.5, 0.5)) == 2
    assert invariant.filtered_rank(2, (0, 0), (1.5, 1.5)) == 1
    assert invariant.filtered_rank(2, (0, 0), (2.5, 1.5)) == 0
    assert invariant.filtered_rank(np.nextafter(2.0, np.inf), (0, 0), (0.5, 0.5)) == 0


def _single_piece(*, x=(0, 1, 2, 3, 4), y=(0, 1, 2, 3, 4), slope=0.5, corner=(4, 4)):
    return SkyscraperInvariant(
        x_grid=x, y_grid=y, box=[[0, 0], [x[-1], y[-1]]],
        source_offsets=[0, 1] + [1] * (len(x) * len(y) - 1), slopes=[slope],
        factor_ranks=[1], factor_group_ids=[7], staircase_offsets=[0, 1],
        corner_offsets=[0, 1], corners=[corner], metadata={},
    )


def test_reference_landscape_exact_and_threshold_equality():
    invariant = _single_piece()
    expected = np.zeros((1, 5, 5))
    np.fill_diagonal(expected[0], [0, 1, 2, 1, 0])
    np.testing.assert_array_equal(invariant.reference_landscape(0.5), expected)
    np.testing.assert_array_equal(invariant.filtered_landscape(0.5), expected)
    np.testing.assert_array_equal(invariant.reference_landscape(0.50001), 0)
    np.testing.assert_array_equal(
        invariant.filtered_landscape_difference(0.50001, 0.5), expected
    )
    with pytest.raises(ValueError, match="must not exceed"):
        invariant.filtered_landscape_difference(0.5, 0.50001)
    levels = invariant.reference_landscape(0, k=2)
    assert levels.shape == (2, 5, 5)
    assert np.all(levels[:-1] >= levels[1:])
    with pytest.raises(ValueError, match="positive"):
        invariant.reference_landscape(0, k=0)


def test_filtered_landscape_plot(monkeypatch):
    import multipers.plots as plots

    calls = []
    monkeypatch.setattr(plots, "plot_surfaces", lambda data, **kwargs: calls.append((data, kwargs)))
    invariant = _single_piece(x=(0, 2, 4), y=(0, 1, 2), corner=(4, 2))
    landscape = invariant.filtered_landscape(0, k=2, plot=True)

    (grid, plotted), kwargs = calls.pop()
    np.testing.assert_array_equal(grid[0], invariant.x_grid)
    np.testing.assert_array_equal(grid[1], invariant.y_grid)
    np.testing.assert_array_equal(plotted, np.swapaxes(landscape, -1, -2))
    assert kwargs == {"cmap": "hot", "contour": False}


def test_reference_landscape_non_square_step_and_grid_validation():
    invariant = _single_piece(x=(0, 2, 4), y=(0, 1, 2), corner=(4, 2))
    np.testing.assert_array_equal(np.diag(invariant.reference_landscape(0)[0]), [0, 2, 0])
    irregular = _single_piece(x=(0, 1, 3), y=(0, 1, 2), corner=(3, 2))
    with pytest.raises(ValueError, match="regular grids"):
        irregular.reference_landscape(0)

    from multipers import _skyscraper_interface

    if _skyscraper_interface.available():
        with pytest.raises(ValueError, match="regular grids"):
            _skyscraper_interface.filtered_landscape(irregular, 0, 1)


def test_python_fallback_matches_native(tmp_path, monkeypatch):
    from multipers import _skyscraper_interface
    from multipers.invariants.skyscraper import _PythonSkyscraperInvariant

    invariant = _single_piece()
    expected = invariant.filtered_landscape(0.5, k=2)
    monkeypatch.setattr(_skyscraper_interface, "available", lambda: False)
    fallback = _PythonSkyscraperInvariant(
        *(getattr(invariant, name) for name in (
            "x_grid", "y_grid", "box", "source_offsets", "slopes", "factor_ranks",
            "factor_group_ids", "staircase_offsets", "corner_offsets", "corners"
        )),
        metadata=dict(invariant.metadata),
    )
    np.testing.assert_array_equal(fallback.filtered_landscape(0.5, k=2), expected)

    path = tmp_path / "fallback.HNF1"
    fallback.to_sky(path, orientation="yx")
    restored = _PythonSkyscraperInvariant.from_sky(path, orientation="yx")
    for name in ("x_grid", "y_grid", "source_offsets", "slopes", "factor_ranks",
                 "factor_group_ids", "staircase_offsets", "corner_offsets", "corners"):
        np.testing.assert_array_equal(getattr(restored, name), getattr(fallback, name))
    path.write_text(path.read_text().replace("bounds,", "invalid,"))
    with pytest.raises(ValueError, match="Malformed HNF1"):
        _PythonSkyscraperInvariant.from_sky(path, orientation="yx")

    path = tmp_path / "fallback.HNF"
    fallback.to_sky(path, version="HNF", orientation="xy")
    assert "G,0,1, (1.0, 0.0)" in path.read_text()
    restored = _PythonSkyscraperInvariant.from_sky(path, orientation="xy")
    np.testing.assert_array_equal(restored.source_offsets, fallback.source_offsets)
    np.testing.assert_array_equal(restored.corners, fallback.corners)


@pytest.mark.parametrize("column_row", [False, True])
@pytest.mark.parametrize("fallback", [False, True])
@pytest.mark.parametrize("fine_grid", [False, True])
def test_legacy_grid_index_compatibility(tmp_path, monkeypatch, column_row, fallback, fine_grid):
    from multipers import _skyscraper_interface
    from multipers.invariants.skyscraper import _PythonSkyscraperInvariant

    if fallback:
        monkeypatch.setattr(_skyscraper_interface, "available", lambda: False)
    elif not _skyscraper_interface.available():
        pytest.skip("Skyscraper backend unavailable")
    if fine_grid:
        record = "G,1,2" if column_row else "G,2,1"
        lattice = "(0,0),(2e-16,2e-16),(1e-16,1e-16)"
        shown = "(1e-16,2e-16)"
        expected_offsets = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
    else:
        record = "G,2,1" if column_row else "G,1,2"
        lattice = "(10,100),(14.000000000000002,103.00000000000001),(2,3)"
        shown = "(14,103)"
        expected_offsets = [0, 0, 0, 0, 0, 0, 1]
    path = tmp_path / "legacy.HNF"
    path.write_text(
        f"HNF\n3,{3 if fine_grid else 2}\n{lattice}\n{record},{shown}\n1,(20;200)\n"
    )
    invariant_type = _PythonSkyscraperInvariant if fallback else SkyscraperInvariant
    restored = invariant_type.from_sky(path, orientation="xy")
    np.testing.assert_array_equal(restored.source_offsets, expected_offsets)
    np.testing.assert_array_equal(restored.corners, [[20, 200]])


@pytest.mark.parametrize("orientation", ["xy", "yx"])
def test_hnf1_roundtrip_and_legacy_orientation(tmp_path, orientation):
    invariant = _single_piece()
    path = tmp_path / "data.sky"
    invariant.to_sky(path, orientation=orientation)
    restored = SkyscraperInvariant.from_sky(path, orientation=orientation)
    for name in ("x_grid", "y_grid", "source_offsets", "slopes", "factor_ranks",
                 "factor_group_ids", "staircase_offsets", "corner_offsets", "corners"):
        np.testing.assert_array_equal(getattr(restored, name), getattr(invariant, name))
    assert restored.metadata == {"sky_version": "HNF1", "grouping_lost": False}

    invariant.to_sky(path, version="HNF", orientation="yx")
    legacy = SkyscraperInvariant.from_sky(path, orientation="yx")
    np.testing.assert_array_equal(legacy.corners, invariant.corners)
    assert legacy.metadata == {"sky_version": "HNF", "grouping_lost": True}
    with pytest.raises(TypeError):
        getattr(SkyscraperInvariant, "from_sky")(path)


def test_sky_rejects_malformed_data(tmp_path):
    path = tmp_path / "bad.sky"
    path.write_text("HNF1\norientation,xy\nbroken\n")
    with pytest.raises(ValueError, match="Malformed"):
        SkyscraperInvariant.from_sky(path, orientation="xy")
    path.write_text("HNF\n2,2\n(0,0),(1,1),(1,1)\nnot-a-factor\n")
    with pytest.raises(ValueError, match="Malformed"):
        SkyscraperInvariant.from_sky(path, orientation="xy")


def test_seeded_packed_payload_properties_and_validation(tmp_path):
    from multipers.invariants.skyscraper import _PythonSkyscraperInvariant

    rng = np.random.default_rng(0x5A17C0DE)
    for _ in range(20):
        nx, ny = map(int, rng.integers(1, 4, size=2))
        x, y = np.arange(nx, dtype=float), 2 * np.arange(ny, dtype=float)
        slopes, ranks, groups, source_offsets = [], [], [], [0]
        staircase_offsets, corner_offsets, corners = [0], [0], []
        for _ in range(nx * ny):
            for _ in range(int(rng.integers(3))):
                slopes.append(float(rng.integers(-2, 3)))
                rank = int(rng.integers(1, 3))
                ranks.append(rank)
                groups.append(len(groups))
                for _ in range(rank):
                    corners.extend((float(rng.integers(-1, nx + 2)), float(2 * rng.integers(-1, ny + 2)))
                                   for _ in range(int(rng.integers(3))))
                    corner_offsets.append(len(corners))
                staircase_offsets.append(staircase_offsets[-1] + rank)
            source_offsets.append(len(slopes))
        invariant = SkyscraperInvariant(
            x, y, [[-1, -2], [nx, 2 * ny]], source_offsets, slopes, ranks, groups,
            staircase_offsets, corner_offsets, np.asarray(corners).reshape(-1, 2), {},
        )
        reference = _PythonSkyscraperInvariant(
            x, y, [[-1, -2], [nx, 2 * ny]], source_offsets, slopes, ranks, groups,
            staircase_offsets, corner_offsets, np.asarray(corners).reshape(-1, 2), {},
        )
        target = (float(rng.integers(-1, nx + 2)), float(2 * rng.integers(-1, ny + 2)))
        previous = None
        for theta in sorted({*slopes, -3.0, 3.0}):
            grid = invariant.filtered_rank_on_grid(theta, target)
            scalar = [[invariant.filtered_rank(theta, (u, v), target) for u in x] for v in y]
            np.testing.assert_array_equal(grid, scalar)
            np.testing.assert_array_equal(grid, reference.filtered_rank_on_grid(theta, target))
            if previous is not None:
                assert np.all(previous >= grid)
            previous = grid

    zero_corner = SkyscraperInvariant(
        [0, 1], [0, 1], [[0, 0], [1, 1]], [0, 1, 1, 1, 1], [1], [1], [0],
        [0, 1], [0, 0], np.empty((0, 2)), {},
    )
    path = tmp_path / "zero.HNF1"
    zero_corner.to_sky(path, version="HNF1", orientation="xy")
    restored = SkyscraperInvariant.from_sky(path, orientation="xy")
    np.testing.assert_array_equal(restored.filtered_rank_on_grid(1, (1, 1)), [[1, 0], [0, 0]])
    with pytest.raises(ValueError, match="cannot encode"):
        zero_corner.to_sky(path, version="HNF", orientation="xy")

    with pytest.raises(ValueError):
        SkyscraperInvariant([0], [0], [[0, 0], [1, 1]], [0, 1], [1], [1], [0], [1, 0], [0], np.empty((0, 2)), {})
    with pytest.raises(ValueError, match="finite"):
        SkyscraperInvariant([np.nan], [0], [[0, 0], [1, 1]], [0, 0], [], [], [], [0], [0], np.empty((0, 2)), {})
    with pytest.raises(ValueError, match="nonnegative integers"):
        SkyscraperInvariant([0], [0], [[0, 0], [1, 1]], [0, 1], [1], [1.5], [0], [0, 1], [0, 0], np.empty((0, 2)), {})
    for invalid_ids in ([True], np.array([True]), np.array([2**53 + 1], dtype=object)):
        with pytest.raises(ValueError, match="nonnegative integers"):
            SkyscraperInvariant(
                [0], [0], [[0, 0], [1, 1]], [0, 1], [1], [1], invalid_ids,
                [0, 1], [0, 0], np.empty((0, 2)), {},
            )
    with pytest.raises(ValueError, match=r"corners must have shape \(n, 2\)"):
        SkyscraperInvariant(
            [0], [0], [[0, 0], [1, 1]], [0, 1], [1], [1], [0],
            [0, 1], [0, 0], np.array([0.0, 0.0]), {},
        )
    with pytest.raises(ValueError, match=r"corners must have shape \(n, 2\)"):
        _PythonSkyscraperInvariant(
            [0], [0], [[0, 0], [1, 1]], [0, 1], [1], [1], [0],
            [0, 1], [0, 0], np.array([0.0, 0.0]), {},
        )

    large_ids = np.array([2**53 + 1, np.iinfo(np.uint64).max], dtype=np.uint64)
    exact = SkyscraperInvariant(
        [0], [0], [[0, 0], [1, 1]], [0, 2], [1, 1], [1, 1], large_ids,
        [0, 1, 2], [0, 0, 0], np.empty((0, 2)), {},
    )
    np.testing.assert_array_equal(exact.factor_group_ids, large_ids)
    with pytest.raises(ValueError, match="nonnegative integers"):
        SkyscraperInvariant(
            [0], [0], [[0, 0], [1, 1]], [0, 1], [1], [1], [float(2**64)],
            [0, 1], [0, 0], np.empty((0, 2)), {},
        )
    with pytest.raises(ValueError, match="finite"):
        exact.slopes_at(np.nan, 0)
    partial_metadata = SkyscraperInvariant(
        [0], [0], [[0, 0], [1, 1]], [0, 0], [], [], [], [0], [0], np.empty((0, 2)),
        {"algorithm": "custom"},
    )
    assert partial_metadata.metadata == {"algorithm": "custom"}
    with pytest.raises(ValueError, match="Unknown Skyscraper metadata"):
        SkyscraperInvariant(
            [0], [0], [[0, 0], [1, 1]], [0, 0], [], [], [], [0], [0], np.empty((0, 2)),
            {"unknown": True},
        )

    offsets = np.array([0, 1], dtype=np.uint64)
    owned = SkyscraperInvariant([0], [0], [[0, 0], [1, 1]], offsets, [1], [1], [0], [0, 1], [0, 0], np.empty((0, 2)), {})
    offsets[1] = 0
    assert owned.source_offsets[1] == 1
    returned_offsets = owned.source_offsets
    from multipers import _skyscraper_interface

    if _skyscraper_interface.available():
        assert type(owned).__module__ == "multipers._skyscraper_interface"
        returned_offsets[1] = 0
        assert owned.source_offsets[1] == 1
    else:
        assert not returned_offsets.flags.writeable
