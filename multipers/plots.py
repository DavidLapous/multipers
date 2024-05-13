from typing import Optional

import matplotlib.pyplot as plt
import numpy as np


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
    **plt_kwargs,  # ignored ftm
):
    # compute the maximal rectangle area
    pts = np.clip(pts, a_min=-np.inf, a_max=np.array((*threshold, *threshold))[None, :])
    alpha_rescaling = 0
    for rectangle, weight in zip(pts, weights):
        if rectangle[2] > x_smoothing * rectangle[0]:
            alpha_rescaling = max(
                alpha_rescaling,
                (rectangle[2] / x_smoothing - rectangle[0])
                * (rectangle[3] - rectangle[1]),
            )
    # draw the rectangles
    for rectangle, weight in zip(pts, weights):
        # draw only the rectangles that have not been reduced to the empty set
        if rectangle[2] > x_smoothing * rectangle[0]:
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
                    alpha=(rectangle[2] / x_smoothing - rectangle[0])
                    * (rectangle[3] - rectangle[1])
                    / alpha_rescaling,
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
                    alpha=1,
                )


def plot_signed_measure(signed_measure, threshold=None, ax=None, **plt_kwargs):
    if ax is None:
        ax = plt.gca()
    else:
        plt.sca(ax)
    pts, weights = signed_measure
    pts = np.asarray(pts)
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
    if isinstance(pts, np.ndarray):
        pass
    else:
        import torch

        if isinstance(pts, torch.Tensor):
            pts = pts.detach().numpy()
        else:
            raise Exception("Invalid measure type.")

    assert num_parameters in (2, 4)
    if num_parameters == 2:
        _plot_signed_measure_2(
            pts=pts, weights=weights, threshold=threshold, **plt_kwargs
        )
    else:
        _plot_signed_measure_4(
            pts=pts, weights=weights, threshold=threshold, **plt_kwargs
        )


def plot_signed_measures(signed_measures, threshold=None, size=4):
    num_degrees = len(signed_measures)
    fig, axes = plt.subplots(
        nrows=1, ncols=num_degrees, figsize=(num_degrees * size, size)
    )
    if num_degrees == 1:
        axes = [axes]
    for ax, signed_measure in zip(axes, signed_measures):
        plot_signed_measure(signed_measure=signed_measure, ax=ax, threshold=threshold)
    plt.tight_layout()


def plot_surface(
    grid,
    hf,
    fig=None,
    ax=None,
    cmap: Optional[str] = None,
    discrete_surface=False,
    has_negative_values=False,
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
            cmap = matplotlib.colormaps["plasma"]
    if discrete_surface:
        if has_negative_values:
            bounds = np.arange(-5, 6, 1, dtype=int)
        else:
            bounds = np.arange(0, 11, 1, dtype=int)
        norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N, extend="max")
        im = ax.pcolormesh(grid[0], grid[1], hf.T, cmap=cmap, norm=norm, **plt_args)
        cbar = fig.colorbar(
            matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm),
            spacing="proportional",
            ax=ax,
        )
        cbar.set_ticks(ticks=bounds, labels=bounds)
        return
    im = ax.pcolormesh(grid[0], grid[1], hf.T, cmap=cmap, **plt_args)
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
    if type(a) != np.ndarray or type(b) != np.ndarray:
        a = np.array(a)
        b = np.array(b)
    return np.min(np.abs(b - a))


def plot2d_PyModule(
    corners,
    box,
    *,
    dimension=-1,
    separated=False,
    min_persistence=0,
    alpha=1,
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
        from warnings import warn

        shapely = False
        warn(
            "Shapely not installed. Fallbacking to matplotlib. The plots may be inacurate."
        )
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
                            _rectangle_box(birth[0], birth[1], death[0], death[1])
                        )
                    else:
                        list_of_rect.append(
                            _rectangle(birth, death, cmap(i / n_summands), alpha)
                        )
        if not (trivial_summand):
            if separated:
                fig, ax = plt.subplots()
                ax.set(xlim=[box[0][0], box[1][0]], ylim=[box[0][1], box[1][1]])
            if shapely:
                summand_shape = union_all(list_of_rect)
                if type(summand_shape) is _Polygon:
                    xs, ys = summand_shape.exterior.xy
                    ax.fill(xs, ys, alpha=alpha, fc=cmap(i / n_summands), ec="None")
                else:
                    for polygon in summand_shape.geoms:
                        xs, ys = polygon.exterior.xy
                        ax.fill(xs, ys, alpha=alpha, fc=cmap(i / n_summands), ec="None")
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
