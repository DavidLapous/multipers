from typing import Optional, Union, Any

import matplotlib.colors as mcolors
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.collections import LineCollection
from numpy.typing import ArrayLike

from multipers.array_api import to_numpy

_custom_colors = [
    "#03045e",
    "#0077b6",
    "#00b4d8",
    "#90e0ef",
]
_cmap_ = ListedColormap(_custom_colors)
_cmap = mcolors.LinearSegmentedColormap.from_list(
    "continuous_cmap", _cmap_.colors, N=256
)


def _plot_rectangle(rectangle: np.ndarray, weight, **plt_kwargs):
    rectangle = np.asarray(rectangle)
    x_axis = rectangle[[0, 2]]
    y_axis = rectangle[[1, 3]]
    color = "blue" if weight > 0 else "red"
    plt.plot(x_axis, y_axis, c=color, **plt_kwargs)


def _plot_signed_measure_2(
    pts, weights, temp_alpha=0.7, threshold=(np.inf, np.inf), **plt_kwargs
):
    import matplotlib.colors

    pts = np.clip(pts, a_min=-np.inf, a_max=np.asarray(threshold)[None, :])
    weights = np.asarray(weights)
    color_weights = np.array(weights, dtype=float)
    neg_idx = weights < 0
    pos_idx = weights > 0
    if np.any(neg_idx):
        current_weights = -weights[neg_idx]
        min_weight = np.max(current_weights)
        color_weights[neg_idx] /= min_weight
        color_weights[neg_idx] -= 1
    else:
        min_weight = 0

    if np.any(pos_idx):
        current_weights = weights[pos_idx]
        max_weight = np.max(current_weights)
        color_weights[pos_idx] /= max_weight
        color_weights[pos_idx] += 1
    else:
        max_weight = 1

    bordeaux = np.array([0.70567316, 0.01555616, 0.15023281, 1])
    light_bordeaux = np.array([0.70567316, 0.01555616, 0.15023281, temp_alpha])
    bleu = np.array([0.2298057, 0.29871797, 0.75368315, 1])
    light_bleu = np.array([0.2298057, 0.29871797, 0.75368315, temp_alpha])
    norm = plt.Normalize(-2, 2)
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "", [bordeaux, light_bordeaux, "white", light_bleu, bleu]
    )
    plt.scatter(
        pts[:, 0], pts[:, 1], c=color_weights, cmap=cmap, norm=norm, **plt_kwargs
    )
    plt.scatter([], [], color=bleu, label="positive mass", **plt_kwargs)
    plt.scatter([], [], color=bordeaux, label="negative mass", **plt_kwargs)
    plt.legend()


def _plot_signed_measure_4(
    pts,
    weights,
    x_smoothing: float = 1,
    area_alpha: bool = True,
    threshold=(np.inf, np.inf),
    alpha=None,
    **plt_kwargs,  # ignored ftm
):
    # compute the maximal rectangle area
    pts = np.clip(pts, a_min=-np.inf, a_max=np.array((*threshold, *threshold))[None, :])
    alpha_rescaling = 0
    for rectangle in pts:
        if rectangle[2] >= x_smoothing * rectangle[0]:
            alpha_rescaling = max(
                alpha_rescaling,
                (rectangle[2] / x_smoothing - rectangle[0])
                * (rectangle[3] - rectangle[1]),
            )

    segments = []
    rgba_list = []
    for rectangle, weight in zip(pts, weights):
        if rectangle[2] < x_smoothing * rectangle[0]:
            continue
        start = (rectangle[0], rectangle[1])
        end = (rectangle[2] / x_smoothing, rectangle[3])
        segments.append([start, end])
        color_base = "blue" if weight > 0 else "red"
        if area_alpha:
            density_alpha = (
                (end[0] - start[0]) * (end[1] - start[1]) / (alpha_rescaling or 1)
                if alpha is None
                else alpha
            )
        else:
            density_alpha = 1 if alpha is None else alpha
        rgba_list.append(mcolors.to_rgba(color_base, alpha=density_alpha))

    if not segments:
        return

    ax = plt.gca()
    lc_kwargs = plt_kwargs.copy()
    label = lc_kwargs.pop("label", None)
    lc_kwargs.pop("alpha", None)
    lc_kwargs.pop("color", None)
    collection = LineCollection(segments, colors=rgba_list, **lc_kwargs)
    if label:
        collection.set_label(label)
    ax.add_collection(collection)
    ax.autoscale_view()


def plot_signed_measure(signed_measure, threshold=None, ax=None, s=None, **plt_kwargs):
    if ax is None:
        ax = plt.gca()
    else:
        plt.sca(ax)
    pts, weights = signed_measure
    pts = to_numpy(pts)
    weights = to_numpy(weights)
    num_pts = pts.shape[0]
    num_parameters = pts.shape[1]
    if threshold is None:
        if num_pts == 0:
            threshold = (np.inf, np.inf)
        else:
            if num_parameters == 4:
                pts_ = np.concatenate([pts[:, :2], pts[:, 2:]], axis=0)
            else:
                pts_ = pts
            threshold = np.max(np.ma.masked_invalid(pts_), axis=0)
        threshold = np.max(
            [threshold, [plt.gca().get_xlim()[1], plt.gca().get_ylim()[1]]], axis=0
        )

    assert num_parameters in (2, 4)

    if num_parameters == 2:
        _plot_signed_measure_2(
            pts=pts, weights=weights, threshold=threshold, s=s, **plt_kwargs
        )
    else:
        _plot_signed_measure_4(
            pts=pts, weights=weights, threshold=threshold, **plt_kwargs
        )


def plot_signed_measures(
    signed_measures, threshold=None, size=4, alpha=None, s=None, **plot_kwargs
):
    num_degrees = len(signed_measures)
    if num_degrees <= 1:
        axes = [plt.gca()]
    else:
        fig, axes = plt.subplots(
            nrows=1, ncols=num_degrees, figsize=(num_degrees * size, size)
        )
    for ax, signed_measure in zip(axes, signed_measures):
        plot_signed_measure(
            signed_measure=signed_measure,
            ax=ax,
            threshold=threshold,
            alpha=alpha,
            s=s,
            **plot_kwargs,
        )
    plt.tight_layout()


def plot_surface(
    grid,
    hf,
    fig=None,
    ax=None,
    cmap: Optional[Union[str, Any]] = None,
    discrete_surface: Optional[bool] = None,
    has_negative_values: Optional[bool] = None,
    contour: bool = True,
    threshold_max=10,
    threshold_min=-10,
    **plt_args,
):
    import matplotlib

    grid = [to_numpy(g) for g in grid]
    hf = to_numpy(hf)
    if ax is None:
        ax = plt.gca()
    else:
        plt.sca(ax)
    if hf.ndim == 3 and hf.shape[0] == 1:
        hf = hf[0]
    assert hf.ndim == 2, "Can only plot a 2d surface"
    if discrete_surface is None:
        discrete_surface = np.issubdtype(hf.dtype, np.integer)
    fig = plt.gcf() if fig is None else fig
    cmap_arg = plt_args.pop("cmap", None)
    if cmap is None:
        if cmap_arg is not None:
            cmap = cmap_arg
        elif discrete_surface:
            if has_negative_values is None:
                has_negative_values = np.any(hf.ravel() < 0)
            cmap = matplotlib.colormaps["gray_r"]
        else:
            cmap = _cmap

    if isinstance(cmap, str):
        cmap = matplotlib.colormaps[cmap]
    assert cmap is not None

    plt_args.pop("norm", None)
    plt_args.pop("shading", None)

    if discrete_surface or not contour:
        new_grid = []
        for g in grid:
            if len(g) > 1:
                step = g[-1] - g[-2]
                new_grid.append(np.concatenate([g, [g[-1] + step]]))
            else:
                new_grid.append(np.concatenate([g, [g[-1] + 1]]))
        grid = new_grid

    if discrete_surface:
        # Fix 1 & 3: Bounds and small threshold handling
        t_min = threshold_min if has_negative_values else 0
        t_max = max(threshold_max, t_min + 1)
        bounds = np.arange(t_min, t_max + 1, 1, dtype=int)

        # Fix: Ensure colormap has enough colors for all discrete bins
        n_bins = len(bounds)
        if cmap.N < n_bins:
            if hasattr(cmap, "resampled"):
                cmap = cmap.resampled(n_bins)
            else:
                cmap = mcolors.LinearSegmentedColormap.from_list(
                    "resampled", cmap(np.linspace(0, 1, n_bins))
                )

        norm = mcolors.BoundaryNorm(bounds, cmap.N, extend="max")

        im = ax.pcolormesh(
            grid[0], grid[1], hf.T, cmap=cmap, norm=norm, shading="flat", **plt_args
        )
        cbar = fig.colorbar(
            cm.ScalarMappable(cmap=cmap, norm=norm),
            spacing="proportional",
            ax=ax,
        )
        # Limit the number of ticks to 10
        if len(bounds) > 10:
            indices = np.linspace(0, len(bounds) - 1, 10, dtype=int)
            ticks = bounds[indices]
        else:
            ticks = bounds
        cbar.set_ticks(ticks=ticks, labels=ticks)
        return im

    if contour:
        levels = plt_args.pop("levels", 50)
        im = ax.contourf(grid[0], grid[1], hf.T, cmap=cmap, levels=levels, **plt_args)
    else:
        im = ax.pcolormesh(
            grid[0], grid[1], hf.T, cmap=cmap, shading="flat", **plt_args
        )
    return im


def plot_surfaces(HF, size=4, **plt_args):
    grid, hf = HF
    hf = to_numpy(hf)
    assert hf.ndim == 3, (
        f"Found hf.shape = {hf.shape}, expected ndim = 3 : degree, 2-parameter surface."
    )

    if "discrete_surface" not in plt_args:
        plt_args["discrete_surface"] = np.issubdtype(hf.dtype, np.integer)

    if plt_args["discrete_surface"]:
        if "has_negative_values" not in plt_args:
            plt_args["has_negative_values"] = bool(np.any(hf < 0))
        if "threshold_min" not in plt_args:
            plt_args["threshold_min"] = int(np.floor(np.min(hf)))
        if "threshold_max" not in plt_args:
            plt_args["threshold_max"] = int(np.ceil(np.max(hf)))

    num_degrees = hf.shape[0]
    fig, axes = plt.subplots(
        nrows=1, ncols=num_degrees, figsize=(num_degrees * size, size)
    )
    if num_degrees == 1:
        axes = [axes]
    for ax, hf_of_degree in zip(axes, hf):
        plot_surface(grid=grid, hf=hf_of_degree, fig=fig, ax=ax, **plt_args)
    plt.tight_layout()


def _rectangle(x, y, color, alpha):
    """
    Defines a rectangle patch in the format {z | x  ≤ z ≤ y} with color and alpha
    """
    from matplotlib.patches import Rectangle as RectanglePatch

    return RectanglePatch(
        x, max(y[0] - x[0], 0), max(y[1] - x[1], 0), color=color, alpha=alpha
    )


def _d_inf(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    return np.min(np.abs(b - a))


HAS_SHAPELY = None


def plot2d_PyModule(
    corners,
    box,
    *,
    dimension=-1,
    separated=False,
    min_persistence=0,
    alpha=None,
    verbose=False,
    save=False,
    dpi=200,
    xlabel=None,
    ylabel=None,
    cmap=None,
    outline_width=0.2,
    outline_threshold=np.inf,
    interleavings=None,
    backend=None,
):
    global HAS_SHAPELY
    if HAS_SHAPELY is None:
        try:
            import shapely
            from shapely import union_all

            HAS_SHAPELY = True
        except ImportError:
            HAS_SHAPELY = False
            from warnings import warn

            warn(
                "Shapely is not installed. MMA plots may be imprecise.",
                ImportWarning,
            )
    if not HAS_SHAPELY:
        backend = "matplotlib" if backend is None else backend
        alpha = 1 if alpha is None else alpha
    else:
        backend = "shapely" if backend is None else backend
        alpha = 0.8 if alpha is None else alpha

    cmap_instance = (
        matplotlib.colormaps["Spectral"] if cmap is None else matplotlib.colormaps[cmap]
    )

    box = np.asarray(box)
    if not separated:
        ax = plt.gca()
        ax.set(xlim=[box[0][0], box[1][0]], ylim=[box[0][1], box[1][1]])

    n_summands = len(corners)

    for i in range(n_summands):
        summand_interleaving = 0 if interleavings is None else interleavings[i]

        births = np.asarray(corners[i][0])
        deaths = np.asarray(corners[i][1])

        if births.size == 0 or deaths.size == 0:
            continue

        if births.ndim == 1:
            births = births[None, :]
        if deaths.ndim == 1:
            deaths = deaths[None, :]
        if births.ndim != 2 or deaths.ndim != 2:
            raise ValueError(
                f"Invalid corners format. Got {births.shape=}, {deaths.shape=}"
            )

        b_expanded = births[:, None, :]
        d_expanded = deaths[None, :, :]

        births_grid, deaths_grid = np.broadcast_arrays(b_expanded, d_expanded)
        births_flat = births_grid.reshape(-1, 2)
        deaths_flat = deaths_grid.reshape(-1, 2)

        births_flat = np.maximum(births_flat, box[0])
        deaths_flat = np.minimum(deaths_flat, box[1])

        is_valid = np.all(deaths_flat > births_flat, axis=1)

        if not np.any(is_valid):
            continue

        valid_births = births_flat[is_valid]
        valid_deaths = deaths_flat[is_valid]

        if interleavings is None:
            diffs = valid_deaths - valid_births
            d_infs = np.min(diffs, axis=1)
            current_max_interleaving = np.max(d_infs) if d_infs.size > 0 else 0
            summand_interleaving = max(summand_interleaving, current_max_interleaving)

        if summand_interleaving < min_persistence:
            continue

        # --- Plotting ---
        color = cmap_instance(i / n_summands)
        outline_summand = (
            "black" if (summand_interleaving > outline_threshold) else None
        )

        if separated:
            fig, ax = plt.subplots()
            ax.set(xlim=[box[0][0], box[1][0]], ylim=[box[0][1], box[1][1]])

        if HAS_SHAPELY:
            # OPTIMIZATION: Shapely Union
            import shapely
            from shapely import union_all

            rects = shapely.box(
                valid_births[:, 0],
                valid_births[:, 1],
                valid_deaths[:, 0],
                valid_deaths[:, 1],
            )
            summand_shape = union_all(rects)

            geoms = getattr(summand_shape, "geoms", [summand_shape])
            for geom in geoms:
                if geom.is_empty:
                    continue
                xs, ys = geom.exterior.xy
                ax.fill(
                    xs,
                    ys,
                    alpha=alpha,
                    fc=color,
                    ec=outline_summand,
                    lw=outline_width,
                    ls="-",
                )
        else:
            from matplotlib.collections import PolyCollection

            # Construct vertices: (N, 4, 2)
            # (x0, y0), (x0, y1), (x1, y1), (x1, y0)
            verts = np.stack(
                [
                    np.stack([valid_births[:, 0], valid_births[:, 1]], axis=1),
                    np.stack([valid_births[:, 0], valid_deaths[:, 1]], axis=1),
                    np.stack([valid_deaths[:, 0], valid_deaths[:, 1]], axis=1),
                    np.stack([valid_deaths[:, 0], valid_births[:, 1]], axis=1),
                ],
                axis=1,
            )

            pc = PolyCollection(
                verts,
                facecolors=color,
                edgecolors=outline_summand,
                alpha=alpha,
                linewidths=outline_width,
            )
            ax.add_collection(pc)

        if separated:
            if xlabel:
                plt.xlabel(xlabel)
            if ylabel:
                plt.ylabel(ylabel)
            if dimension >= 0:
                plt.title(f"$\\mathrm{{H}}_{dimension}$ 2-persistence")

    if not separated:
        if xlabel:
            plt.xlabel(xlabel)
        if ylabel:
            plt.ylabel(ylabel)
        if dimension >= 0:
            plt.title(f"$\\mathrm{{H}}_{dimension}$ 2-persistence")

    return


def plot_simplicial_complex(
    st,
    pts: ArrayLike,
    x: float,
    y: float,
    mma=None,
    degree=None,
    show_pos: bool = True,
):
    """
    Scatters the points, with the simplices in the filtration at coordinates (x,y).
    if an mma module is given, plots it in a second axis
    """
    if mma is not None:
        fig, (a, b) = plt.subplots(ncols=2, figsize=(15, 5))
        plt.sca(a)
        plot_simplicial_complex(st, pts, x, y)
        plt.sca(b)
        mma.plot(degree=degree)
        box = mma.get_box()
        a, b, c, d = box.ravel()
        mma.plot(degree=1, min_persistence=0.01)
        plt.vlines(x, b, d, color="k", linestyle="--")
        plt.hlines(y, a, c, color="k", linestyle="--")
        plt.scatter([x], [y], c="r", zorder=10)
        if show_pos:
            plt.text(x + 0.01 * (b - a), y + 0.01 * (d - c), f"({x},{y})")
        return

    pts = np.asarray(pts)
    values = np.array([-f[1] for s, f in st.get_skeleton(0)])
    qs = np.quantile(values, np.linspace(0, 1, 100))

    def color_idx(d):
        return np.searchsorted(qs, d) / 100

    from matplotlib.pyplot import get_cmap

    def color(d):
        return get_cmap("viridis")([0, color_idx(d), 1])[1]

    cols_pc = np.asarray([color(v) for v in values])
    ax = plt.gca()
    for s, f in st:  # simplexe, filtration
        density = -f[1]
        if len(s) <= 1 or f[0] > x or density < -y:  # simplexe  = point
            continue
        if len(s) == 2:  # simplexe = segment
            xx = np.array([pts[a, 0] for a in s])
            yy = np.array([pts[a, 1] for a in s])
            plt.plot(xx, yy, c=color(density), alpha=1, zorder=10 * density, lw=1.5)
        if len(s) == 3:  # simplexe = triangle
            xx = np.array([pts[a, 0] for a in s])
            yy = np.array([pts[a, 1] for a in s])
            _c = color(density)
            ax.fill(xx, yy, c=_c, alpha=0.3, zorder=0)
    out = plt.scatter(pts[:, 0], pts[:, 1], c=cols_pc, zorder=10, s=10)
    ax.set_aspect(1)
    return out


def plot_point_cloud(
    pts,
    function,
    x,
    y,
    mma=None,
    degree=None,
    ball_alpha=0.3,
    point_cmap="viridis",
    color_bias=1,
    ball_color=None,
    point_size=20,
):
    if mma is not None:
        fig, (a, b) = plt.subplots(ncols=2, figsize=(15, 5))
        plt.sca(a)
        plot_point_cloud(pts, function, x, y)
        plt.sca(b)
        mma.plot(degree=degree)
        box = mma.get_box()
        a, b, c, d = box.ravel()
        mma.plot(degree=1, min_persistence=0.01)
        plt.vlines(x, b, d, color="k", linestyle="--")
        plt.hlines(y, a, c, color="k", linestyle="--")
        plt.scatter([x], [y], c="r", zorder=10)
        plt.text(x + 0.01 * (b - a), y + 0.01 * (d - c), f"({x},{y})")
        return
    values = -function
    qs = np.quantile(values, np.linspace(0, 1, 100))

    def color_idx(d):
        return np.searchsorted(qs, d * color_bias) / 100

    from matplotlib.collections import PatchCollection
    from matplotlib.pyplot import get_cmap

    def color(d):
        return get_cmap(point_cmap)([0, color_idx(d), 1])[1]

    _colors = np.array([color(v) for v in values])
    ax = plt.gca()
    idx = function <= y
    circles = [plt.Circle(pt, x) for pt, c in zip(pts[idx], function)]
    pc = PatchCollection(circles, alpha=ball_alpha, color=ball_color)
    ax.add_collection(pc)
    plt.scatter(*pts.T, c=_colors, s=point_size)
    ax.set_aspect(1)
