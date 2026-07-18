from __future__ import annotations

from dataclasses import dataclass, field
from operator import index
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np

import multipers.logs as _mp_logs
from multipers.array_api import api_from_tensors
from multipers.grids import compute_bounding_box, compute_grid

from ._utils import _as_slicer

_METADATA_FIELDS = {
    "degree", "coordinates", "max_induced_rank", "rank_cap", "complete_subspace_mode",
    "algorithm", "field", "coordinate_order", "backend_revision",
    "slope_tie_relative_tolerance", "sky_version", "grouping_lost",
}


def _plot_filtered_landscape(grid, landscape) -> None:
    from multipers.plots import plot_surfaces

    plot_surfaces(
        (grid, np.swapaxes(landscape, -1, -2)),
        cmap="hot",
        contour=False,
    )


@dataclass(frozen=True)
class _PythonSkyscraperInvariant:
    x_grid: Any
    y_grid: Any
    box: Any
    source_offsets: Any
    slopes: Any
    factor_ranks: Any
    factor_group_ids: Any
    staircase_offsets: Any
    corner_offsets: Any
    corners: Any
    metadata: dict[str, Any] = field(default_factory=dict)
    _piece_factors: np.ndarray = field(init=False, repr=False, compare=False)
    _piece_cells: np.ndarray = field(init=False, repr=False, compare=False)
    _corner_pieces: np.ndarray = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        arrays = {
            "x_grid": (np.float64, 1), "y_grid": (np.float64, 1),
            "box": (np.float64, 2), "source_offsets": (np.uint64, 1),
            "slopes": (np.float64, 1), "factor_ranks": (np.uint64, 1),
            "factor_group_ids": (np.uint64, 1), "staircase_offsets": (np.uint64, 1),
            "corner_offsets": (np.uint64, 1), "corners": (np.float64, 2),
        }
        for name, (dtype, ndim) in arrays.items():
            raw = np.asarray(getattr(self, name))
            if dtype == np.uint64:
                try:
                    valid = raw.dtype.kind in "iuf" and not (raw.dtype.kind == "f" and raw.dtype.itemsize > 8)
                    valid = valid and np.all(np.isfinite(raw)) and np.all(raw >= 0) and np.all(raw == np.floor(raw))
                    valid = valid and np.all(raw <= np.iinfo(np.uint64).max)
                except TypeError:
                    valid = False
                if not valid:
                    raise ValueError(f"{name} must contain nonnegative integers.")
            value = np.array(raw, dtype=dtype, copy=True, order="C")
            if value.ndim != ndim:
                if name == "box":
                    raise ValueError("box must have shape (2, 2)")
                if name == "corners":
                    raise ValueError("corners must have shape (n, 2)")
                raise ValueError(f"{name} must be {ndim}D.")
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        if self.box.shape != (2, 2) or self.corners.shape[1:] != (2,):
            raise ValueError("Invalid box or corner shape.")
        if not all(np.all(np.isfinite(a)) for a in (self.x_grid, self.y_grid, self.box, self.slopes, self.corners)):
            raise ValueError("Grids, box, slopes, and corners must be finite.")
        if any(len(a) == 0 or np.any(a[1:] <= a[:-1]) for a in (self.x_grid, self.y_grid)):
            raise ValueError("Grid axes must be nonempty and strictly increasing.")
        if np.any(self.box[0] >= self.box[1]):
            raise ValueError("box must define a nonempty rectangle.")
        if len(self.source_offsets) != len(self.x_grid) * len(self.y_grid) + 1:
            raise ValueError("source_offsets does not match grid shape.")
        if self.source_offsets[0] or self.source_offsets[-1] != len(self.slopes):
            raise ValueError("Invalid source_offsets.")
        if len(self.factor_ranks) != len(self.slopes) or len(self.factor_group_ids) != len(self.slopes):
            raise ValueError("Factor arrays must have equal lengths.")
        if len(self.staircase_offsets) != len(self.slopes) + 1:
            raise ValueError("Invalid staircase_offsets.")
        if np.any(np.diff(self.staircase_offsets) != self.factor_ranks):
            raise ValueError("Each factor must have one staircase per rank.")
        if len(self.corner_offsets) != int(self.staircase_offsets[-1]) + 1:
            raise ValueError("Invalid corner_offsets.")
        if self.corner_offsets[-1] != len(self.corners):
            raise ValueError("Invalid packed corners.")
        if any(np.any(a[1:] < a[:-1]) for a in (self.source_offsets, self.staircase_offsets, self.corner_offsets)):
            raise ValueError("Packed offsets must be nondecreasing.")
        piece_factors = np.repeat(np.arange(len(self.slopes)), np.diff(self.staircase_offsets).astype(np.intp))
        factor_cells = np.repeat(np.arange(len(self.source_offsets) - 1), np.diff(self.source_offsets).astype(np.intp))
        piece_cells = factor_cells[piece_factors]
        corner_pieces = np.repeat(np.arange(len(self.corner_offsets) - 1), np.diff(self.corner_offsets).astype(np.intp))
        for name, value in {
            "_piece_factors": piece_factors,
            "_piece_cells": piece_cells,
            "_corner_pieces": corner_pieces,
        }.items():
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        metadata = dict(self.metadata)
        if set(metadata) - _METADATA_FIELDS:
            raise ValueError("Unknown Skyscraper metadata field.")
        object.__setattr__(self, "metadata", MappingProxyType(metadata))

    def _cell(self, x, y) -> int:
        if not np.isfinite(x) or not np.isfinite(y):
            raise ValueError("Coordinates must be finite.")
        ix = np.searchsorted(self.x_grid, x, side="right") - 1
        iy = np.searchsorted(self.y_grid, y, side="right") - 1
        if ix < 0 or iy < 0:
            return -1
        return int(iy * len(self.x_grid) + ix)

    def slopes_at(self, x, y) -> np.ndarray:
        cell = self._cell(x, y)
        if cell < 0:
            return self.slopes[:0]
        return self.slopes[self.source_offsets[cell] : self.source_offsets[cell + 1]]

    def at(self, x, y) -> dict[str, Any]:
        cell = self._cell(x, y)
        start = end = 0 if cell < 0 else int(self.source_offsets[cell])
        if cell >= 0:
            end = int(self.source_offsets[cell + 1])
        staircases = []
        for factor in range(start, end):
            pieces = []
            for piece in range(int(self.staircase_offsets[factor]), int(self.staircase_offsets[factor + 1])):
                pieces.append(self.corners[self.corner_offsets[piece] : self.corner_offsets[piece + 1]])
            staircases.append(tuple(pieces))
        return {
            "slopes": self.slopes[start:end],
            "factor_ranks": self.factor_ranks[start:end],
            "factor_group_ids": self.factor_group_ids[start:end],
            "staircases": tuple(staircases),
        }

    def filtered_rank(self, theta, source, target) -> int:
        """Evaluate filtered rank from ``source`` to ``target``.

        Factors with slope equal to ``theta`` are retained. Each packed
        staircase contributes one when ``target`` lies in its support.
        """
        source = np.asarray(source, dtype=np.float64)
        target = np.asarray(target, dtype=np.float64)
        if source.shape != (2,) or target.shape != (2,):
            raise ValueError("source and target must be 2D coordinates.")
        if not np.isfinite(theta) or not np.all(np.isfinite(source)) or not np.all(np.isfinite(target)):
            raise ValueError("theta, source, and target must be finite.")
        if np.any(target < source):
            return 0
        cell = self._cell(*source)
        if cell < 0:
            return 0
        total = 0
        start, end = map(int, self.source_offsets[cell : cell + 2])
        for factor in range(start, end):
            if self.slopes[factor] < theta:
                continue
            for piece in range(int(self.staircase_offsets[factor]), int(self.staircase_offsets[factor + 1])):
                corners = self.corners[self.corner_offsets[piece] : self.corner_offsets[piece + 1]]
                if not np.any(np.all(corners <= target, axis=1)):
                    total += 1
        return total

    def filtered_rank_on_grid(self, theta, target) -> np.ndarray:
        target = np.asarray(target, dtype=np.float64)
        if target.shape != (2,):
            raise ValueError("target must be a 2D coordinate.")
        if not np.isfinite(theta) or not np.all(np.isfinite(target)):
            raise ValueError("theta and target must be finite.")

        blocked = np.zeros(len(self.corner_offsets) - 1, dtype=bool)
        np.logical_or.at(blocked, self._corner_pieces, np.all(self.corners <= target, axis=1))
        retained = (self.slopes[self._piece_factors] >= theta) & ~blocked
        ranks = np.bincount(
            self._piece_cells, weights=retained, minlength=len(self.source_offsets) - 1
        ).astype(np.uint64, copy=False)
        ranks = ranks.reshape(len(self.y_grid), len(self.x_grid))
        ranks[self.y_grid > target[1], :] = 0
        ranks[:, self.x_grid > target[0]] = 0
        return ranks

    def filtered_landscape(self, theta, k=1, plot=False) -> np.ndarray:
        """Compute filtered Skyscraper landscape; ``k`` is number of levels.

        Upstream levels are 1-indexed; returned axis 0 stores levels 1 through
        ``k``. Requires regular x/y grids with positive steps. Set ``plot=True``
        to plot every level with :func:`multipers.plots.plot_surfaces`.
        """
        k = index(k)
        if k < 1:
            raise ValueError("k must be positive.")
        if not np.isfinite(theta):
            raise ValueError("theta must be finite.")
        if len(self.x_grid) < 2 or len(self.y_grid) < 2:
            raise ValueError("filtered_landscape requires at least two points per grid axis.")
        dx, dy = np.diff(self.x_grid), np.diff(self.y_grid)
        if (dx[0] <= 0 or dy[0] <= 0 or not np.all(np.isfinite(dx)) or not np.all(np.isfinite(dy))
                or not np.allclose(dx, dx[0]) or not np.allclose(dy, dy[0])):
            raise ValueError("filtered_landscape requires regular grids with positive steps.")
        if not np.isfinite(self.x_grid[-1] - self.x_grid[0]) or not np.isfinite(self.y_grid[-1] - self.y_grid[0]):
            raise ValueError("filtered_landscape grid span must be finite.")
        from multipers import _skyscraper_interface

        if _skyscraper_interface.available():
            out = _skyscraper_interface.filtered_landscape(self, theta, k)
            if plot:
                _plot_filtered_landscape((self.x_grid, self.y_grid), out)
            return out
        step_x, slope = float(dx[0]), float(dy[0] / dx[0])
        if slope <= 0 or not np.isfinite(slope):
            raise ValueError("filtered_landscape direction must be finite.")
        out = np.zeros((k, len(self.y_grid), len(self.x_grid)), dtype=np.float64)
        for iy, y in enumerate(self.y_grid):
            for ix, x in enumerate(self.x_grid):
                lengths = []
                cell = iy * len(self.x_grid) + ix
                start, end = map(int, self.source_offsets[cell : cell + 2])
                for factor in range(start, end):
                    if self.slopes[factor] < theta:
                        continue
                    for piece in range(int(self.staircase_offsets[factor]), int(self.staircase_offsets[factor + 1])):
                        corners = self.corners[self.corner_offsets[piece] : self.corner_offsets[piece + 1]]
                        if len(corners):
                            relative = corners - (x, y)
                            projected = relative[:, 1] / slope
                            if not np.all(np.isfinite(relative)) or not np.all(np.isfinite(projected)):
                                raise ValueError("filtered_landscape arithmetic is not finite.")
                            lengths.append(float(np.min(np.maximum(relative[:, 0], projected))))
                lengths.sort(reverse=True)
                for level, length in enumerate(lengths[:k]):
                    d = length / 2
                    for t in range(min(len(self.x_grid) - ix, len(self.y_grid) - iy)):
                        distance = self.x_grid[ix + t] - self.x_grid[ix]
                        value = d - abs(d - distance)
                        if not np.isfinite(distance) or not np.isfinite(value):
                            raise ValueError("filtered_landscape arithmetic is not finite.")
                        value = max(0.0, value)
                        out[level, iy + t, ix + t] = max(out[level, iy + t, ix + t], value)
                        if t and value == 0:
                            break
        out = np.minimum.accumulate(out, axis=0)
        if plot:
            _plot_filtered_landscape((self.x_grid, self.y_grid), out)
        return out

    def reference_landscape(self, theta, k=1) -> np.ndarray:
        """Alias for :meth:`filtered_landscape` retained for compatibility."""
        return self.filtered_landscape(theta, k=k)

    def filtered_landscape_difference(self, theta, theta_prime, k=1) -> np.ndarray:
        """Return ``L(theta_prime) - L(theta)`` for ``theta_prime <= theta``."""
        if theta_prime > theta:
            raise ValueError("theta_prime must not exceed theta.")
        return self.filtered_landscape(theta_prime, k=k) - self.filtered_landscape(theta, k=k)

    def to_sky(self, path, version="HNF1", orientation="xy") -> None:
        """Write versioned lossless data, or upstream-compatible legacy ``HNF``."""
        if version not in {"HNF", "HNF1"} or orientation not in {"xy", "yx"}:
            raise ValueError("version must be 'HNF' or 'HNF1' and orientation must be 'xy' or 'yx'.")
        from multipers import _skyscraper_interface

        if _skyscraper_interface.available():
            _skyscraper_interface.write_sky(self, str(Path(path)), version, orientation)
            return
        if version == "HNF1":
            if self.source_offsets.nbytes > 64 << 20:
                raise ValueError("grid exceeds HNF1 cell limit.")
            lines = ["HNF1", f"orientation,{orientation}", "bounds," + ",".join(repr(float(v)) for v in self.box.ravel())]
            axes = (self.x_grid, self.y_grid) if orientation == "xy" else (self.y_grid, self.x_grid)
            lines += ["x," + ",".join(repr(float(v)) for v in axes[0]), "y," + ",".join(repr(float(v)) for v in axes[1])]
            for cell in range(len(self.source_offsets) - 1):
                for factor in range(int(self.source_offsets[cell]), int(self.source_offsets[cell + 1])):
                    lines.append(f"F,{cell},{float(self.slopes[factor])!r},{int(self.factor_group_ids[factor])}")
                    for piece in range(int(self.staircase_offsets[factor]), int(self.staircase_offsets[factor + 1])):
                        corners = self.corners[self.corner_offsets[piece] : self.corner_offsets[piece + 1]]
                        if orientation == "yx":
                            corners = corners[:, ::-1]
                        lines.append("P" + "".join(f",({float(a)!r};{float(b)!r})" for a, b in corners))
            lines.append(f"END,{len(self.slopes)},{int(self.staircase_offsets[-1])},{len(self.corners)}")
            Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
            return
        _write_legacy_sky(self, Path(path), orientation)

    @classmethod
    def from_sky(cls, path, orientation):
        """Read ``HNF1`` or legacy ``HNF`` using explicit coordinate orientation."""
        if orientation not in {"xy", "yx"}:
            raise ValueError("orientation must be explicitly 'xy' or 'yx'.")
        from multipers import _skyscraper_interface

        if _skyscraper_interface.available():
            raw = _skyscraper_interface.read_sky(str(Path(path)), orientation)
            metadata = {name: raw.pop(name) for name in ("sky_version", "grouping_lost")}
            return cls(metadata=metadata, **raw)
        lines = Path(path).read_text(encoding="utf-8").splitlines()
        if not lines:
            raise ValueError("Empty sky file.")
        return _read_hnf1(cls, lines, orientation) if lines[0] == "HNF1" else _read_legacy_sky(cls, lines, orientation)


def _read_hnf1(cls, lines, orientation):
    try:
        if lines[1] != f"orientation,{orientation}":
            raise ValueError("HNF1 orientation does not match requested orientation.")
        bounds_fields = lines[2].split(",")
        first_fields = lines[3].split(",")
        second_fields = lines[4].split(",")
        if len(bounds_fields) != 5 or bounds_fields[0] != "bounds" or first_fields[0] != "x" or second_fields[0] != "y":
            raise ValueError
        bounds = [float(x) for x in bounds_fields[1:]]
        first = np.asarray([float(x) for x in first_fields[1:]])
        second = np.asarray([float(x) for x in second_fields[1:]])
        x_grid, y_grid = (first, second) if orientation == "xy" else (second, first)
        n_cells = len(x_grid) * len(y_grid)
        if (n_cells + 1) * np.dtype(np.uintp).itemsize > 64 << 20:
            raise ValueError
        footer = lines[-1].split(",")
        if len(footer) != 4 or footer[0] != "END":
            raise ValueError
        expected_counts = tuple(map(int, footer[1:]))
        factors: list[list[Any]] = []
        current: list[Any] | None = None
        for line in lines[5:-1]:
            fields = line.split(",")
            if fields[0] == "F" and len(fields) == 4:
                current = [int(fields[1]), float(fields[2]), int(fields[3]), []]
                factors.append(current)
            elif fields[0] == "P" and current is not None:
                corners = []
                for token in fields[1:]:
                    if not (token.startswith("(") and token.endswith(")") and ";" in token):
                        raise ValueError
                    a, b = map(float, token[1:-1].split(";"))
                    corners.append((a, b) if orientation == "xy" else (b, a))
                current[3].append(corners)
            else:
                raise ValueError
        pieces = sum(len(factor[3]) for factor in factors)
        corners = sum(len(piece) for factor in factors for piece in factor[3])
        if expected_counts != (len(factors), pieces, corners):
            raise ValueError
        if any(f[0] < 0 or f[0] >= n_cells for f in factors) or any(factors[i][0] > factors[i + 1][0] for i in range(len(factors) - 1)):
            raise ValueError
        return _pack_sky(cls, x_grid, y_grid, bounds, factors, {"sky_version": "HNF1", "grouping_lost": False})
    except (IndexError, TypeError, ValueError) as error:
        if isinstance(error, ValueError) and str(error).startswith("HNF1 orientation"):
            raise
        raise ValueError("Malformed HNF1 sky file.") from error


def _write_legacy_sky(invariant, path, orientation):
    axes = (invariant.x_grid, invariant.y_grid) if orientation == "xy" else (invariant.y_grid, invariant.x_grid)
    if any(len(axis) < 2 or np.diff(axis)[0] <= 0 or not np.array_equal(axis, axis[0] + np.arange(len(axis)) * np.diff(axis)[0]) for axis in axes):
        raise ValueError("Legacy HNF requires regular grids with positive steps.")
    if np.any(invariant.corner_offsets[1:] == invariant.corner_offsets[:-1]):
        raise ValueError("Legacy HNF cannot encode staircases without relation corners; use HNF1.")
    lines = ["HNF", f"{len(axes[0])},{len(axes[1])}",
             f"({float(axes[0][0])!r}, {float(axes[1][0])!r}),({float(axes[0][-1])!r}, {float(axes[1][-1])!r}),({float(np.diff(axes[0])[0])!r}, {float(np.diff(axes[1])[0])!r})"]
    nx = len(invariant.x_grid)
    for cell in range(len(invariant.source_offsets) - 1):
        iy, ix = divmod(cell, nx)
        i, j = (ix, iy) if orientation == "xy" else (iy, ix)
        source = (invariant.x_grid[ix], invariant.y_grid[iy])
        shown = source if orientation == "xy" else source[::-1]
        lines.append(f"G,{j},{i}, ({float(shown[0])!r}, {float(shown[1])!r})")
        for factor in range(int(invariant.source_offsets[cell]), int(invariant.source_offsets[cell + 1])):
            for piece in range(int(invariant.staircase_offsets[factor]), int(invariant.staircase_offsets[factor + 1])):
                corners = invariant.corners[invariant.corner_offsets[piece] : invariant.corner_offsets[piece + 1]]
                if orientation == "yx":
                    corners = corners[:, ::-1]
                lines.append(repr(float(invariant.slopes[factor])) + "".join(f",({float(a)!r};{float(b)!r})" for a, b in corners))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _read_legacy_sky(cls, lines, orientation):
    try:
        if lines[0] != "HNF":
            raise ValueError
        nx, ny = map(int, lines[1].split(","))
        import re
        pairs = re.findall(r"\(([^,]+),\s*([^\)]+)\)", lines[2])
        if len(pairs) != 3 or nx < 1 or ny < 1:
            raise ValueError
        (start_a, start_b), (end_a, end_b), (step_a, step_b) = [tuple(map(float, pair)) for pair in pairs]
        if step_a <= 0 or step_b <= 0 or not np.all(np.isfinite([start_a, start_b, end_a, end_b, step_a, step_b])):
            raise ValueError
        first, second = start_a + np.arange(nx) * step_a, start_b + np.arange(ny) * step_b
        tolerance = 32 * np.finfo(float).eps
        def coordinate_matches(left, right, step):
            return abs(left - right) <= min(tolerance * max(1, abs(left), abs(right)), step / 4)

        if not np.isclose(end_a, first[-1], rtol=tolerance, atol=tolerance) or not np.isclose(
                end_b, second[-1], rtol=tolerance, atol=tolerance):
            raise ValueError
        x_grid, y_grid = (first, second) if orientation == "xy" else (second, first)
        if len(x_grid) * len(y_grid) > sum(len(line) + 1 for line in lines):
            raise ValueError
        factors, cell, grid_order = [], None, None
        for line in lines[3:]:
            if not line.strip():
                continue
            if line.startswith("G,"):
                fields = line.split(",", 3)
                if len(fields) != 4:
                    raise ValueError
                i, j = int(fields[1]), int(fields[2])
                match = re.fullmatch(r"\s*\(\s*([^,()]+)\s*,\s*([^()]+)\s*\)\s*", fields[3])
                if match is None:
                    raise ValueError
                shown = tuple(map(float, match.groups()))
                row_column = (0 <= i < ny and 0 <= j < nx and
                              coordinate_matches(shown[0], first[j], step_a) and
                              coordinate_matches(shown[1], second[i], step_b))
                column_row = (0 <= i < nx and 0 <= j < ny and
                              coordinate_matches(shown[0], first[i], step_a) and
                              coordinate_matches(shown[1], second[j], step_b))
                if not row_column and not column_row or grid_order == "row_column" and not row_column or grid_order == "column_row" and not column_row:
                    raise ValueError
                if grid_order is None and row_column != column_row:
                    grid_order = "row_column" if row_column else "column_row"
                use_row_column = grid_order != "column_row" and row_column
                first_index, second_index = (j, i) if use_row_column else (i, j)
                cell = (second_index * len(x_grid) + first_index if orientation == "xy"
                        else first_index * len(x_grid) + second_index)
            else:
                if cell is None:
                    raise ValueError
                fields = line.split(",")
                if len(fields) < 2:
                    raise ValueError
                corners = []
                for token in fields[1:]:
                    if not (token.startswith("(") and token.endswith(")") and ";" in token):
                        raise ValueError
                    a, b = map(float, token[1:-1].split(";"))
                    corners.append((a, b) if orientation == "xy" else (b, a))
                factors.append([cell, float(fields[0]), len(factors), [corners]])
        factors.sort(key=lambda f: f[0])
        return _pack_sky(cls, x_grid, y_grid, [x_grid[0], y_grid[0], x_grid[-1], y_grid[-1]], factors,
                         {"sky_version": "HNF", "grouping_lost": True})
    except (IndexError, TypeError, ValueError) as error:
        raise ValueError("Malformed legacy HNF sky file.") from error


def _pack_sky(cls, x_grid, y_grid, bounds, factors, metadata):
    source_offsets, slopes, ranks, groups = [0], [], [], []
    staircase_offsets, corner_offsets, corners = [0], [0], []
    position = 0
    for cell in range(len(x_grid) * len(y_grid)):
        while position < len(factors) and factors[position][0] == cell:
            _, slope, group, pieces = factors[position]
            slopes.append(slope); ranks.append(len(pieces)); groups.append(group)
            for piece in pieces:
                corners.extend(piece); corner_offsets.append(len(corners))
            staircase_offsets.append(staircase_offsets[-1] + len(pieces))
            position += 1
        source_offsets.append(len(slopes))
    return cls(x_grid=x_grid, y_grid=y_grid, box=np.asarray(bounds).reshape(2, 2), source_offsets=source_offsets,
               slopes=slopes, factor_ranks=ranks, factor_group_ids=groups, staircase_offsets=staircase_offsets,
               corner_offsets=corner_offsets, corners=np.asarray(corners, dtype=float).reshape(-1, 2), metadata=metadata)


from multipers import _skyscraper_interface

SkyscraperInvariant = (
    _skyscraper_interface.SkyscraperInvariant
    if _skyscraper_interface.available()
    else _PythonSkyscraperInvariant
)


def skyscraper_invariant(
    filtered_complex,
    degree=None,
    *,
    grid=None,
    box=None,
    max_rank=7,
    grid_strategy="exact",
    resolution=None,
    inflate=0.1,
    minpres_kwargs=None,
):
    """Compute fixed-grid Skyscraper invariant of a 2-parameter module."""
    from multipers import _skyscraper_interface
    from multipers.ops import aida

    _skyscraper_interface.require()
    slicer = _as_slicer(filtered_complex)
    if slicer.num_parameters != 2:
        raise ValueError("skyscraper_invariant expects exactly two parameters.")
    if slicer.is_pres:
        inferred = int(slicer.pres_degree)
        if degree is None:
            degree = inferred
        elif index(degree) != inferred:
            raise ValueError("Cannot change degree of an existing presentation.")
    elif degree is None:
        raise ValueError("degree is required unless input is a presentation.")
    degree = index(degree)
    if degree < 0:
        raise ValueError("degree must be non-negative.")
    if grid is None:
        grid = compute_grid(slicer, strategy=grid_strategy, resolution=resolution)
    if len(grid) != 2:
        raise ValueError("grid must contain two nonempty axes.")
    grid_api = api_from_tensors(*grid)
    if any(grid_api.has_grad(axis) for axis in grid):
        _mp_logs.warn_autodiff(
            "skyscraper_invariant converts grid axes to NumPy; grid gradients will be lost."
        )
    grid = tuple(grid_api.asnumpy(axis, dtype=np.float64, contiguous=True) for axis in grid)
    if any(len(axis) == 0 for axis in grid):
        raise ValueError("grid must contain two nonempty axes.")
    if any(not np.all(np.isfinite(axis)) or np.any(np.diff(axis) <= 0) for axis in grid):
        raise ValueError("grid axes must contain finite, strictly increasing coordinates.")
    if box is None:
        box = compute_bounding_box(slicer, inflate=inflate, relative=True)
    box = np.ascontiguousarray(box, dtype=np.float64)
    if box.shape != (2, 2) or not np.all(np.isfinite(box)) or np.any(box[0] >= box[1]):
        raise ValueError("box must be a finite nonempty rectangle with shape (2, 2).")
    kwargs = {} if minpres_kwargs is None else dict(minpres_kwargs)
    if kwargs.pop("full_resolution", False):
        _mp_logs.warn_superfluous_computation(
            "skyscraper_invariant ignores `full_resolution=True`; only a minimal presentation is needed."
        )
    presentation = slicer if slicer.is_minpres else slicer.minpres(degree=degree, full_resolution=False, **kwargs)
    summands = [summand.unsqueeze() if summand.is_squeezed else summand for summand in aida(presentation)]
    max_rank = index(max_rank)
    if max_rank < 1:
        raise ValueError("max_rank must be positive.")
    return _skyscraper_interface.fixed_grid(
        summands, grid[0], grid[1], box, max_rank=max_rank, degree=degree
    )
