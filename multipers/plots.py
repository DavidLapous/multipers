from typing import Optional
from warnings import warn

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
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
    pts = np.clip(pts, a_min=-np.inf,
                  a_max=np.array((*threshold, *threshold))[None, :])
    alpha_rescaling = 0
    for rectangle, weight in zip(pts, weights):
        if rectangle[2] >= x_smoothing * rectangle[0]:
            alpha_rescaling = max(
                alpha_rescaling,
                (rectangle[2] / x_smoothing - rectangle[0])
                * (rectangle[3] - rectangle[1]),
            )
    # draw the rectangles
    for rectangle, weight in zip(pts, weights):
        # draw only the rectangles that have not been reduced to the empty set
        if rectangle[2] >= x_smoothing * rectangle[0]:
            # make the alpha channel proportional to the rectangle's area
            if area_alpha:
                _plot_rectangle(
                    rectangle=[
                        rectangle[0],
                        rectangle[1],
                        rectangle[2] / x_smoothing,
                        rectangle[3],
                    ],
                    weight=weight,
                    alpha=(
                        (rectangle[2] / x_smoothing - rectangle[0])
                        * (rectangle[3] - rectangle[1])
                        / alpha_rescaling
                        if alpha is None
                        else alpha
                    ),
                    **plt_kwargs,
                )
            else:
                _plot_rectangle(
                    rectangle=[
                        rectangle[0],
                        rectangle[1],
                        rectangle[2] / x_smoothing,
                        rectangle[3],
                    ],
                    weight=weight,
                    alpha=1 if alpha is None else alpha,
                    **plt_kwargs,
                )


def plot_signed_measure(signed_measure, threshold=None, ax=None, **plt_kwargs):
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
            pts=pts, weights=weights, threshold=threshold, **plt_kwargs
        )
    else:
        _plot_signed_measure_4(
            pts=pts, weights=weights, threshold=threshold, **plt_kwargs
        )


def plot_signed_measures(signed_measures, threshold=None, size=4, alpha=None):
    num_degrees = len(signed_measures)
    if num_degrees <= 1:
        axes = [plt.gca()]
    else:
        fig, axes = plt.subplots(
            nrows=1, ncols=num_degrees, figsize=(num_degrees * size, size)
        )
    for ax, signed_measure in zip(axes, signed_measures):
        plot_signed_measure(signed_measure=signed_measure,
                            ax=ax, threshold=threshold,
                            alpha=alpha)
    plt.tight_layout()


def plot_surface(
    grid,
    hf,
    fig=None,
    ax=None,
    cmap: Optional[str] = None,
    discrete_surface: bool = False,
    has_negative_values: bool = False,
    contour: bool = True,
    **plt_args,
):
    import matplotlib

    if ax is None:
        ax = plt.gca()
    else:
        plt.sca(ax)
    if hf.ndim == 3 and hf.shape[0] == 1:
        hf = hf[0]
    assert hf.ndim == 2, "Can only plot a 2d surface"
    fig = plt.gcf() if fig is None else fig
    if cmap is None:
        if discrete_surface:
            cmap = matplotlib.colormaps["gray_r"]
        else:
            cmap = _cmap
    if discrete_surface or not contour:
        # for shading="flat"
        grid = [np.concatenate([g, [g[-1]*1.1 - .1*g[0]]]) for g in grid]
    if discrete_surface:
        if has_negative_values:
            bounds = np.arange(-5, 6, 1, dtype=int)
        else:
            bounds = np.arange(0, 11, 1, dtype=int)
        norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N, extend="max")
        im = ax.pcolormesh(grid[0], grid[1], hf.T, cmap=cmap,
                           norm=norm, shading="flat", **plt_args)
        cbar = fig.colorbar(
            matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm),
            spacing="proportional",
            ax=ax,
        )
        cbar.set_ticks(ticks=bounds, labels=bounds)
        return im

    if contour:
        levels = plt_args.pop("levels", 50)
        im = ax.contourf(grid[0], grid[1], hf.T,
                         cmap=cmap, levels=levels, **plt_args)
    else:
        im = ax.pcolormesh(grid[0], grid[1], hf.T,
                           cmap=cmap,  shading="flat", **plt_args)
    return im


def plot_surfaces(HF, size=4, **plt_args):
    grid, hf = HF
    assert (
        hf.ndim == 3
    ), f"Found hf.shape = {hf.shape}, expected ndim = 3 : degree, 2-parameter surface."
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
    shapely=True,
    xlabel=None,
    ylabel=None,
    cmap=None,
):
    import matplotlib

    try:
        from shapely import union_all
        from shapely.geometry import Polygon as _Polygon
        from shapely.geometry import box as _rectangle_box

        shapely = True and shapely
    except ImportError:
        shapely = False
        warn(
            "Shapely not installed. Fallbacking to matplotlib. The plots may be inacurate."
        )
    if alpha is None:
        alpha = 0.8 if shapely else 1
    if not shapely and alpha != 1:
        warn("Opacity without shapely will lead to incorect plots.")
    cmap = (
        matplotlib.colormaps["Spectral"] if cmap is None else matplotlib.colormaps[cmap]
    )
    box = list(box)
    if not (separated):
        # fig, ax = plt.subplots()
        ax = plt.gca()
        ax.set(xlim=[box[0][0], box[1][0]], ylim=[box[0][1], box[1][1]])
    n_summands = len(corners)
    for i in range(n_summands):
        trivial_summand = True
        list_of_rect = []
        for birth in corners[i][0]:
            if len(birth) == 1:
                birth = np.asarray([birth[0]] * 2)
            birth = np.asarray(birth).clip(min=box[0])
            for death in corners[i][1]:
                if len(death) == 1:
                    death = np.asarray([death[0]] * 2)
                death = np.asarray(death).clip(max=box[1])
                if death[1] > birth[1] and death[0] > birth[0]:
                    if trivial_summand and _d_inf(birth, death) > min_persistence:
                        trivial_summand = False
                    if shapely:
                        list_of_rect.append(
                            _rectangle_box(
                                birth[0], birth[1], death[0], death[1])
                        )
                    else:
                        list_of_rect.append(
                            _rectangle(birth, death, cmap(
                                i / n_summands), alpha)
                        )
        if not (trivial_summand):
            if separated:
                fig, ax = plt.subplots()
                ax.set(xlim=[box[0][0], box[1][0]],
                       ylim=[box[0][1], box[1][1]])
            if shapely:
                summand_shape = union_all(list_of_rect)
                if type(summand_shape) is _Polygon:
                    xs, ys = summand_shape.exterior.xy
                    ax.fill(xs, ys, alpha=alpha, fc=cmap(
                        i / n_summands), ec="None")
                else:
                    for polygon in summand_shape.geoms:
                        xs, ys = polygon.exterior.xy
                        ax.fill(xs, ys, alpha=alpha, fc=cmap(
                            i / n_summands), ec="None")
            else:
                for rectangle in list_of_rect:
                    ax.add_patch(rectangle)
            if separated:
                if xlabel:
                    plt.xlabel(xlabel)
                if ylabel:
                    plt.ylabel(ylabel)
                if dimension >= 0:
                    plt.title(rf"$H_{dimension}$ $2$-persistence")
    if not (separated):
        if xlabel is not None:
            plt.xlabel(xlabel)
        if ylabel is not None:
            plt.ylabel(ylabel)
        if dimension >= 0:
            plt.title(rf"$H_{dimension}$ $2$-persistence")
    return


def plot_simplicial_complex(
    st, pts: ArrayLike, x: float, y: float, mma=None, degree=None
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
            plt.plot(xx, yy, c=color(density), alpha=1,
                     zorder=10 * density, lw=1.5)
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
