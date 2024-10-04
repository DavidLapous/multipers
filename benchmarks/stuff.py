import shutil
from collections.abc import Callable, Sequence
from itertools import product
from timeit import timeit

import numpy as np
import pandas as pd

import multipers as mp
import multipers.grids as mpg
import multipers.ml.point_clouds as mmp
from multipers.data import noisy_annulus, orbit, three_annulus
from multipers.slicer import Slicer_type, available_columns

np.random.seed(0)

available_dataset: dict[str, Callable] = {
    "orbit35": lambda n: orbit(n=n, r=3.5),
    "orbit41": lambda n: orbit(n=n, r=4.1),
    "orbit43": lambda n: orbit(n=n, r=4.3),
    "na": lambda n: noisy_annulus(n1=(m := int((2 / 3) * n)), n2=n - m),
    "3a": lambda n: three_annulus(num_pts=(m := int((2 / 3) * n)), num_outliers=n - m),
}


datasets: Sequence[str] = list(available_dataset.keys())
degrees: Sequence[int] = [0, 1]
num_pts: Sequence[int] = [200]
complexes = ["delaunay", "rips"]
invariants = ["mma", "slice", "hilbert", "rank"]
vineyard = ["vine", "novine"]
num_lines = 50
num_repetition = 5
timings = {}
available_dtype = [np.float64]


def fill_timing(arg, f):
    timings[arg] = timeit(f, number=num_repetition)
    terminal_width = shutil.get_terminal_size().columns
    left = str(args)
    right = f"{timings[arg]:.4f}"
    dots = terminal_width - (len(left) + len(right) + 2)
    print(f"{left} {'.' * dots} {right}", end="\n")


for args in product(
    num_pts,
    datasets,
    complexes,
    invariants,
    degrees,
    vineyard,
    available_dtype,
    available_columns,
):
    n, dataset, cplx, inv, degree, vine, dtype, col = args
    pts = np.asarray(available_dataset[dataset](n))
    s: Slicer_type = mmp.PointCloud2FilteredComplex(
        complex=cplx,
        bandwidths=[0.2],
        num_collapses=2,
        output_type="slicer",
        reduce_degrees=[degree],
        expand_dim=degree + 1,
    ).fit_transform([pts])[0][0]
    s = mp.Slicer(s, vineyard=(vine == "vine"), dtype=dtype, column_type=col)
    box = mpg.compute_bounding_box(s)
    s.minpres_degree = -1  ## makes it non-minpres again
    if inv == "mma":
        if vine == "vine":
            f = lambda: mp.module_approximation(s)
        else:
            f = lambda: 1
    elif inv == "slice":
        basepoints = np.random.uniform(
            low=box[None, :, 0],
            high=box[None, :, 1],
            size=(num_lines, s.num_parameters),
        )
        directions = [np.ones(s.num_parameters)] * num_lines
        f = lambda: s.persistence_on_lines(basepoints, directions)
    elif inv == "hilbert":
        grid = mpg.compute_grid(s, resolution=50, strategy="regular")
        f = lambda: mp.signed_measure(s, grid=grid, degree=degree, invariant="hilbert")

    elif inv == "rank":
        grid = mpg.compute_grid(s, resolution=20, strategy="regular")
        f = lambda: mp.signed_measure(s, grid=grid, degree=degree, invariant="rank")
    else:
        raise ValueError(f"Invariant {inv} is not benchmarkable.")

    try:
        fill_timing(args, f)
    except ValueError:
        print("invalid args", args, "with function", f)


pd.DataFrame(
    [(*args, t) for (args), t in timings.items()],
    columns=["npts", "dataset", "complex", "inv", "degree", "vine", "dtype", "timing"],
).to_csv(f"benchmark_v{mp.__version__}.csv", index=False)
