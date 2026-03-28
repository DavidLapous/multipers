import re
import os
import itertools
import subprocess
import tempfile
import threading
import time
from collections import defaultdict
from shutil import which
from typing import Literal, Optional

import gudhi as gd
import numpy as np
from gudhi import SimplexTree

import multipers.logs as _mp_logs

try:
    from . import _function_delaunay_interface
except ImportError:
    _function_delaunay_interface = None

try:
    from . import _multi_critical_interface
except ImportError:
    _multi_critical_interface = None

try:
    from . import _rhomboid_tiling_interface
except ImportError:
    _rhomboid_tiling_interface = None

try:
    from . import _2pac_interface
except ImportError:
    _2pac_interface = None

current_doc_url = "https://davidlapous.github.io/multipers/"
doc_soft_urls = {
    "mpfree": "https://bitbucket.org/mkerber/mpfree/",
    "multi_chunk": "https://bitbucket.org/mkerber/multi_chunk/",
    "function_delaunay": "https://bitbucket.org/mkerber/function_delaunay/",
    "2pac": "https://gitlab.com/flenzen/2pac",
    "rhomboid_tiling": "https://github.com/DavidLapous/rhomboidtiling_newer_cgal_version",
    "multi_critical": "https://bitbucket.org/mkerber/multi_critical",
}
doc_soft_easy_install = {
    "mpfree": f"""
```sh
git clone {doc_soft_urls["mpfree"]}
cd mpfree
cmake . --fresh
make
cp mpfree $CONDA_PREFIX/bin/
cd .. 
rm -rf mpfree
```
        """,
    "multi_chunk": f"""
```sh
git clone {doc_soft_urls["multi_chunk"]}
cd multi_chunk
cmake . --fresh
make
cp multi_chunk $CONDA_PREFIX/bin/
cd .. 
rm -rf multi_chunk
```
        """,
    "function_delaunay": f"""
```sh
git clone {doc_soft_urls["function_delaunay"]}
cd function_delaunay
cmake . --fresh
make
cp main $CONDA_PREFIX/bin/function_delaunay
cd ..
rm -rf function_delaunay
```
        """,
    "2pac": f"""
```sh
git clone {doc_soft_urls["2pac"]} 2pac
cd 2pac && mkdir build && cd build
cmake ..
make
cp 2pac $CONDA_PREFIX/bin
```
""",
    "rhomboid_tiling": f"""
git clone {doc_soft_urls["rhomboid_tiling"]} rhomboid_tiling
cd rhomboid_tiling
sh build.sh
cp orderk $CONDA_PREFIX/bin/rhomboid_tiling
""",
    "multi_critical": f"""
```sh
git clone {doc_soft_urls["multi_critical"]}
cd multi_critical
cmake . --fresh
make
cp multi_critical $CONDA_PREFIX/bin/
cd ..
""",
}
doc_soft_urls = defaultdict(lambda: "<Unknown url>", doc_soft_urls)
doc_soft_easy_install = defaultdict(lambda: "<Unknown>", doc_soft_easy_install)

available_reduce_softs = Literal["mpfree", "multi_chunk", "2pac", "multi_critical"]


def _interface_is_available(module):
    return module is not None and module._is_available()


def _path_init(soft: str | os.PathLike):
    a = which(f"./{soft}")
    b = which(f"{soft}")
    if a:
        pathes[soft] = a
    elif b:
        pathes[soft] = b

    if pathes[soft] is not None:
        verbose_arg = "> /dev/null 2>&1"
        test = os.system(pathes[soft] + " --help " + verbose_arg)
        if test % 256 != 0:
            _mp_logs.warn_fallback(f"""
            Found external software {soft} at {pathes[soft]}
            but may not behave well.
            """)


pathes = {
    "mpfree": None,
    "2pac": None,
    "function_delaunay": None,
    "multi_chunk": None,
    "multi_critical": None,
    "rhomboid_tiling": None,
}


## TODO : optimize with Python.h ?
def scc_parser(path: str | os.PathLike):
    """
    Parse an scc file into the scc python format, aka blocks.
    """
    pass_line_regex = re.compile(r"^\s*$|^#|^scc2020$")

    def valid_line(line):
        return pass_line_regex.match(line) is None

    parse_line_regex = re.compile(r"^(?P<filtration>[^;]+);(?P<boundary>[^;]*)$")
    with open(path, "r") as f:
        lines = (x.strip() for x in f if valid_line(x))
        num_parameters = int(next(lines))
        sizes = np.cumsum(np.asarray([0] + next(lines).split(), dtype=np.int32))
        lines = (parse_line_regex.match(a) for a in lines)
        clines = tuple((a.group("filtration"), a.group("boundary")) for a in lines)
    # F = np.fromiter((a[0].split() for a in clines), dtype=np.dtype((np.float64,2)), count = sizes[-1])
    F = np.fromiter(
        (np.fromstring(a[0], sep=r" ", dtype=np.float64) for a in clines),
        dtype=np.dtype((np.float64, num_parameters)),
        count=sizes[-1],
    )

    # B = tuple(np.asarray(a[1].split(), dtype=np.int32) if len(a[1])>0 else np.empty(0, dtype=np.int32) for a in clines) ## TODO : this is very slow : optimize
    B = tuple(np.fromstring(a[1], sep=" ", dtype=np.int32) for a in clines)
    # block_lines = (tuple(get_bf(x, num_parameters) for x in lines[sizes[i]:sizes[i+1]]) for i in range(len(sizes)-1))

    # blocks = [(np.asarray([x[0] for x in b if len(x)>0], dtype=float),tuple(x[1] for x in b))  for b in block_lines]
    blocks = [
        (F[sizes[i] : sizes[i + 1]], B[sizes[i] : sizes[i + 1]])
        for i in range(len(sizes) - 1)
    ]

    return blocks


def _init_external_softwares(requires=[]):
    global pathes
    any_required = False
    for soft, soft_path in pathes.items():
        if soft_path is None:
            _path_init(soft)
            any_required = any_required or (soft in requires)

    if any_required:
        for soft in requires:
            if pathes[soft] is None:
                global doc_soft_urls
                raise ValueError(f"""
Did not find {soft}.
Install it from {doc_soft_urls[soft]}, and put it in your current directory,
or in you $PATH.
Documentation is available here: {current_doc_url}compilation.html#external-libraries
For instance:
{doc_soft_easy_install[soft]}
                                 """)


_init_external_softwares()


def _check_available(soft: str):
    _init_external_softwares()
    return pathes.get(soft, None) is not None


def scc_reduce_from_str_to_slicer(
    path: str | os.PathLike,
    slicer,
    full_resolution=True,
    dimension: int | np.int64 = 1,
    clear: bool = True,
    verbose: bool = False,
    backend: Literal["mpfree", "multi_chunk", "2pac"] = "mpfree",
    shift_dimension=0,
):
    """
    Computes a minimal presentation of the file in path,
    using mpfree.

    path:PathLike
    slicer: empty slicer to fill
    full_resolution: bool
    dimension: int, presentation dimension to consider
    clear: bool, removes temporary files if True
    verbose: bool
    backend: "mpfree", "multi_chunk" or "2pac"
    """
    global pathes
    _init_external_softwares(requires=[backend])

    with tempfile.TemporaryDirectory(prefix="multipers", delete=clear) as tmpdir:
        output_path = os.path.join(tmpdir, "multipers_output.scc")

        resolution_str = "--resolution" if full_resolution else ""

        if not os.path.exists(path):
            raise ValueError(f"No file found at {path}.")

        verbose_arg = "> /dev/null 2>&1" if not verbose else ""
        if backend == "mpfree":
            more_verbose = "-v" if verbose else ""
            command = f"{pathes[backend]} {more_verbose} {resolution_str} --dim={dimension} {path} {output_path} {verbose_arg}"
        elif backend == "multi_chunk":
            command = f"{pathes[backend]}  {path} {output_path} {verbose_arg}"
        elif backend in ["twopac", "2pac"]:
            command = f"{pathes[backend]} -f {path} --scc-input -n{dimension} --save-resolution-scc {output_path} {verbose_arg}"
        else:
            raise ValueError(f"Unsupported backend {backend}.")
        if verbose:
            print(f"Calling :\n\n {command}")
        os.system(command)

        slicer._build_from_scc_file(path=output_path, shift_dimension=shift_dimension)


def _minimal_presentation_from_slicer(
    slicer,
    degree,
    backend="mpfree",
    auto_clean=True,
    verbose=False,
    full_resolution=True,
    use_clearing=True,
    use_chunk=True,
):
    """
    Computes a minimal presentation from a slicer, using the in-memory bridge when
    available and falling back to SCC file I/O otherwise.
    """
    if backend == "mpfree":
        from multipers import _mpfree_interface

        if _mpfree_interface._is_available():
            if verbose:
                print(
                    f"[multipers.io] backend=mpfree mode=cpp_interface degree={degree}",
                    flush=True,
                )
            new_slicer = _mpfree_interface.minimal_presentation(
                slicer,
                degree=degree,
                verbose=verbose,
                _backend_stdout=_mp_logs.ext_log_enabled(),
                use_chunk=use_chunk,
                use_clearing=use_clearing,
                full_resolution=full_resolution,
            )
            new_slicer.minpres_degree = degree
            new_slicer.filtration_grid = (
                slicer.filtration_grid if slicer.is_squeezed else None
            )
            if new_slicer.is_squeezed and auto_clean:
                new_slicer = new_slicer._clean_filtration_grid()
            return new_slicer

    if backend == "2pac" and _interface_is_available(_2pac_interface):
        if verbose:
            print(
                f"[multipers.io] backend=2pac mode=cpp_interface degree={degree}",
                flush=True,
            )
        new_slicer = _2pac_interface.minimal_presentation(
            slicer,
            degree=degree,
            verbose=verbose,
            _backend_stdout=_mp_logs.ext_log_enabled(),
            use_chunk=use_chunk,
            use_clearing=use_clearing,
            full_resolution=full_resolution,
        )
        new_slicer.minpres_degree = degree
        new_slicer.filtration_grid = (
            slicer.filtration_grid if slicer.is_squeezed else None
        )
        if new_slicer.is_squeezed and auto_clean:
            new_slicer = new_slicer._clean_filtration_grid()
        return new_slicer

    _init_external_softwares(requires=[backend])
    if verbose:
        print(
            f"[multipers.io] backend={backend} mode=disk_interface degree={degree}",
            flush=True,
        )
    dimension = slicer.dimension - degree
    with tempfile.TemporaryDirectory(prefix="multipers") as tmpdir:
        tmp_path = os.path.join(tmpdir, "multipers.scc")
        slicer.to_scc(
            path=tmp_path, strip_comments=True, degree=degree - 1, unsqueeze=False
        )
        new_slicer = type(slicer)()
        if backend == "mpfree":
            shift_dimension = degree - 1
        else:
            shift_dimension = degree
        scc_reduce_from_str_to_slicer(
            path=tmp_path,
            slicer=new_slicer,
            dimension=dimension,
            backend=backend,
            shift_dimension=shift_dimension,
            verbose=verbose,
        )

        new_slicer.minpres_degree = degree
        new_slicer.filtration_grid = (
            slicer.filtration_grid if slicer.is_squeezed else None
        )
        if new_slicer.is_squeezed and auto_clean:
            new_slicer = new_slicer._clean_filtration_grid()
        return new_slicer


def function_delaunay_presentation_to_slicer(
    slicer,
    point_cloud: np.ndarray,
    function_values: np.ndarray,
    clear: bool = True,
    verbose: bool = False,
    degree=-1,
    multi_chunk=False,
):
    """
    Computes a function delaunay presentation, and returns it as a slicer.

    slicer: empty slicer to fill
    points : (num_pts, n) float array
    grades : (num_pts,) float array
    degree (opt) : if given, computes a minimal presentation of this homological degree first
    clear:bool, removes temporary files if true
    degree: computes minimal presentation of this degree if given
    verbose : bool
    """
    global pathes

    point_cloud = np.asarray(point_cloud, dtype=np.float64)
    function_values = np.asarray(function_values, dtype=np.float64).reshape(-1)
    if point_cloud.ndim != 2:
        raise ValueError(f"point_cloud should be a 2d array. Got {point_cloud.shape=}")
    if function_values.ndim != 1:
        raise ValueError(
            f"function_values should be a 1d array. Got {function_values.shape=}"
        )
    if point_cloud.shape[0] != function_values.shape[0]:
        raise ValueError(
            f"point_cloud and function_values should have same number of points. "
            f"Got {point_cloud.shape[0]} and {function_values.shape[0]}."
        )

    if _interface_is_available(_function_delaunay_interface):
        if verbose:
            print(
                f"[multipers.io] backend=function_delaunay mode=cpp_interface degree={degree} multi_chunk={multi_chunk}",
                flush=True,
            )
        return _function_delaunay_interface.function_delaunay_to_slicer(
            slicer,
            point_cloud,
            function_values,
            degree,
            multi_chunk,
            verbose,
        )
    # fallbacks to doing scc file io
    if verbose:
        print(
            f"[multipers.io] backend=function_delaunay mode=disk_interface degree={degree} multi_chunk={multi_chunk}",
            flush=True,
        )
    with tempfile.TemporaryDirectory(prefix="multipers", delete=clear) as tmpdir:
        input_path = os.path.join(tmpdir, "multipers_input.scc")
        output_path = os.path.join(tmpdir, "multipers_output.scc")

        backend = "function_delaunay"
        _init_external_softwares(requires=[backend])

        to_write = np.concatenate([point_cloud, function_values.reshape(-1, 1)], axis=1)
        np.savetxt(input_path, to_write, delimiter=" ")
        command = [pathes[backend]]
        if degree >= 0:
            command.extend(["--minpres", str(degree)])
        if multi_chunk:
            command.append("--multi-chunk")
        command.extend([input_path, output_path, "--no-delaunay-compare"])
        if verbose:
            print(" ".join(command))
        subprocess.run(
            command,
            check=True,
            stdout=None if verbose else subprocess.DEVNULL,
            stderr=None if verbose else subprocess.DEVNULL,
        )

        slicer._build_from_scc_file(
            path=output_path, shift_dimension=-1 if degree <= 0 else degree - 1
        )
        return slicer


def function_delaunay_presentation_to_simplextree(
    point_cloud: np.ndarray,
    function_values: np.ndarray,
    clear: bool = True,
    verbose: bool = False,
    dtype=np.float64,
):
    """
    Computes a function delaunay complex and returns it as a SimplexTreeMulti.

    This path is intended for non-reduced outputs (degree < 0).
    """
    from multipers.simplex_tree_multi import SimplexTreeMulti
    from multipers.slicer import to_simplextree
    import multipers

    point_cloud = np.asarray(point_cloud, dtype=np.float64)
    function_values = np.asarray(function_values, dtype=np.float64).reshape(-1)
    if point_cloud.ndim != 2:
        raise ValueError(f"point_cloud should be a 2d array. Got {point_cloud.shape=}")
    if function_values.ndim != 1:
        raise ValueError(
            f"function_values should be a 1d array. Got {function_values.shape=}"
        )
    if point_cloud.shape[0] != function_values.shape[0]:
        raise ValueError(
            f"point_cloud and function_values should have same number of points. "
            f"Got {point_cloud.shape[0]} and {function_values.shape[0]}."
        )

    if _interface_is_available(_function_delaunay_interface):
        st = SimplexTreeMulti(num_parameters=2, dtype=dtype)
        return _function_delaunay_interface.function_delaunay_to_simplextree(
            st,
            point_cloud,
            function_values,
            verbose,
        )

    # external fallback: build slicer from SCC, then convert
    s = multipers.Slicer(None, dtype=dtype)
    function_delaunay_presentation_to_slicer(
        s,
        point_cloud,
        function_values,
        clear=clear,
        verbose=verbose,
        degree=-1,
        multi_chunk=False,
    )
    return to_simplextree(s)


def _rhomboid_tiling_to_slicer(
    slicer,
    point_cloud: np.ndarray,
    k_max,
    degree=-1,
    clear: bool = True,
    verbose=False,
):
    """TODO"""
    global pathes
    backend = "rhomboid_tiling"
    t_total_start = time.perf_counter()
    point_cloud = np.asarray(point_cloud, dtype=np.float64)
    if point_cloud.ndim != 2 or not point_cloud.shape[1] in [2, 3]:
        raise ValueError(
            f"point_cloud should be a 2d array of shape (-,2) or (-,3). Got {point_cloud.shape=}"
        )

    if _interface_is_available(_rhomboid_tiling_interface):
        t_cpp_start = time.perf_counter()
        if verbose:
            print(
                f"[multipers.io] backend=rhomboid_tiling mode=cpp_interface degree={degree} k_max={k_max}",
                flush=True,
            )
        out = _rhomboid_tiling_interface.rhomboid_tiling_to_slicer(
            slicer,
            point_cloud,
            k_max,
            degree,
            verbose,
        )
        if verbose:
            t_cpp = time.perf_counter() - t_cpp_start
            t_total = time.perf_counter() - t_total_start
            print(
                "[multipers.io][timing] "
                f"backend=rhomboid_tiling mode=cpp_interface total={t_total:.3f}s "
                f"cpp_call={t_cpp:.3f}s overhead={max(t_total - t_cpp, 0.0):.3f}s",
                flush=True,
            )
        return out

    _init_external_softwares(requires=[backend])
    if verbose:
        print(
            f"[multipers.io] backend=rhomboid_tiling mode=disk_interface degree={degree} k_max={k_max}",
            flush=True,
        )
    with tempfile.TemporaryDirectory(prefix="multipers", delete=clear) as tmpdir:
        input_path = os.path.join(tmpdir, "point_cloud.txt")
        output_path = os.path.join(tmpdir, "multipers_output.scc")
        t_write_start = time.perf_counter()
        np.savetxt(input_path, point_cloud, delimiter=" ")
        t_write = time.perf_counter() - t_write_start

        verbose_arg = "> /dev/null 2>&1" if not verbose else ""
        command = f"{pathes[backend]} {input_path} {output_path} {point_cloud.shape[1]} {k_max} scc {degree} {verbose_arg}"
        if verbose:
            print(command)
        t_backend_start = time.perf_counter()
        os.system(command)
        t_backend = time.perf_counter() - t_backend_start
        t_read_start = time.perf_counter()
        slicer._build_from_scc_file(
            path=output_path, shift_dimension=-1 if degree <= 0 else degree - 1
        )
        t_read = time.perf_counter() - t_read_start
        if verbose:
            t_total = time.perf_counter() - t_total_start
            t_io = t_write + t_read
            io_fraction = (t_io / t_total) if t_total > 0 else 0.0
            print(
                "[multipers.io][timing] "
                f"backend=rhomboid_tiling mode=disk_interface total={t_total:.3f}s "
                f"write_input={t_write:.3f}s external_binary={t_backend:.3f}s "
                f"read_scc={t_read:.3f}s io_fraction={io_fraction:.1%}",
                flush=True,
            )
        return slicer


def _multi_critical_from_slicer(
    slicer,
    reduce=False,
    algo: Literal["path", "tree"] = "path",
    degree: Optional[int] = None,
    clear=True,
    swedish=None,
    verbose=False,
    kcritical=False,
    filtration_container="contiguous",
    **slicer_kwargs,
):
    need_split = False
    reduce = False if reduce is None else reduce
    swedish = degree is not None if swedish is None else swedish

    if _interface_is_available(_multi_critical_interface):
        if reduce:
            out_kcritical = kcritical
            out_filtration_container = filtration_container
        else:
            out_kcritical = False
            out_filtration_container = "contiguous"
        if verbose:
            print(
                f"[multipers.io] backend=multi_critical mode=cpp_interface algo={algo} reduce={reduce} degree={degree} swedish={swedish}",
                flush=True,
            )
        return _multi_critical_interface.one_criticalify(
            slicer,
            reduce=reduce,
            algo=algo,
            degree=degree,
            swedish=swedish,
            verbose=verbose,
            _backend_stdout=_mp_logs.ext_log_enabled(),
            kcritical=out_kcritical,
            filtration_container=out_filtration_container,
            **slicer_kwargs,
        )

    from multipers import Slicer

    newSlicer = Slicer(
        slicer,
        return_type_only=True,
        kcritical=kcritical,
        filtration_container=filtration_container,
        **slicer_kwargs,
    )
    if verbose:
        print(
            f"[multipers.io] backend=multi_critical mode=disk_interface algo={algo} reduce={reduce} degree={degree} swedish={swedish}",
            flush=True,
        )

    with tempfile.TemporaryDirectory(prefix="multipers", delete=clear) as tmpdir:
        input_path = os.path.join(tmpdir, "multipers_input.scc")
        output_path = os.path.join(tmpdir, "multipers_output.scc")
        slicer.to_scc(input_path, degree=0, strip_comments=True)

        reduce_arg = ""
        if reduce:
            # External multi_critical binaries are not guaranteed to support --swedish.
            # Keep this flag in the in-memory path only.
            if degree is None:
                need_split = True
                reduce_arg += r" --minpres-all"
            else:
                reduce_arg += rf" --minpres {degree + 1}"
        verbose_arg = "> /dev/null 2>&1" if not verbose else "--verbose"

        _init_external_softwares(requires=["multi_critical"])
        command = f"{pathes['multi_critical']} --{algo} {reduce_arg} {input_path} {output_path} {verbose_arg}"
        if verbose:
            print(command)
        os.system(command)
        if need_split:
            os.system(
                f'awk \'/scc2020/ {{n++}} {{print > ("{tmpdir}/multipers_block_" n ".scc")}}\' {output_path}'
            )
            from glob import glob
            import re

            files = glob(tmpdir + "/multipers_block_*.scc")
            files.sort(key=lambda f: int(re.search(r"\d+", f).group()))
            num_degrees = len(files)
            ss = tuple(
                newSlicer()
                ._build_from_scc_file(files[i], shift_dimension=i - 1)
                .minpres(i)
                for i in range(num_degrees)
            )
            return ss
        out = newSlicer()._build_from_scc_file(
            str(output_path), shift_dimension=degree - 1 if reduce else -2
        )
        if reduce:
            out.minpres_degree = degree
        return out


def scc2disk(
    stuff,
    path: str | os.PathLike,
    num_parameters=-1,
    reverse_block=False,
    rivet_compatible=False,
    ignore_last_generators=False,
    strip_comments=False,
):
    """
    Writes a scc python format / blocks into a file.
    """
    if num_parameters == -1:
        for block in stuff:
            if len(block[0]) == 0:
                continue
            num_gens, num_parameters_ = np.asarray(block[0]).shape
            num_parameters = num_parameters_
            break
    assert num_parameters > 0, f"Invalid number of parameters {num_parameters}"

    if reverse_block:
        stuff.reverse()
    with open(path, "w") as f:
        f.write("scc2020\n") if not rivet_compatible else f.write("firep\n")
        if not strip_comments and not rivet_compatible:
            f.write("# Number of parameters\n")
        if rivet_compatible:
            assert num_parameters == 2
            f.write("Filtration 1\n")
            f.write("Filtration 2\n")
        else:
            f.write(f"{num_parameters}\n")

        if not strip_comments:
            f.write("# Sizes of generating sets\n")
        for block in stuff:
            f.write(f"{len(block[0])} ")
        f.write("\n")
        for i, block in enumerate(stuff):
            if (rivet_compatible or ignore_last_generators) and i == len(stuff) - 1:
                continue
            if not strip_comments:
                f.write(f"# Block of dimension {len(stuff) - 1 - i}\n")
            filtration, boundary = block
            filtration = np.asarray(filtration).astype(str)
            # boundary = tuple(x.astype(str) for x in boundary)
            f.write(
                " ".join(
                    itertools.chain.from_iterable(
                        (
                            (
                                *(f.tolist()),
                                ";",
                                *(np.asarray(b).astype(str).tolist()),
                                "\n",
                            )
                            for f, b in zip(filtration, boundary)
                        )
                    )
                )
            )
            # for j in range(<int>len(filtration)):
            #     line = " ".join((
            #         *filtration[j],
            #         ";",
            #         *boundary[j],
            #         "\n",
            #     ))
            #     f.write(line)


def scc2disk_old(
    stuff,
    path: str | os.PathLike,
    num_parameters=-1,
    reverse_block=False,
    rivet_compatible=False,
    ignore_last_generators=False,
    strip_comments=False,
):
    """
    Writes a scc python format / blocks into a file.
    """
    if num_parameters == -1:
        for block in stuff:
            if len(block[0]) == 0:
                continue
            num_gens, num_parameters_ = np.asarray(block[0]).shape
            num_parameters = num_parameters_
            break
    assert num_parameters > 0, f"Invalid number of parameters {num_parameters}"

    if reverse_block:
        stuff.reverse()
    out = []
    if rivet_compatible:
        out.append(r"firep")
    else:
        out.append(r"scc2020")
    if not strip_comments and not rivet_compatible:
        out.append(r"# Number of parameters")
    if rivet_compatible:
        out.append("Filtration 1")
        out.append("Filtration 2\n")
    else:
        out.append(f"{num_parameters}")

    if not strip_comments:
        out.append("# Sizes of generating sets")

    # for block in stuff:
    #     f.write(f"{len(block[0])} ")
    out.append(" ".join(str(len(block[0])) for block in stuff))
    str_blocks = [out]
    for i, block in enumerate(stuff):
        if (rivet_compatible or ignore_last_generators) and i == len(stuff) - 1:
            continue
        if not strip_comments:
            str_blocks.append([f"# Block of dimension {len(stuff) - 1 - i}"])
        filtration, boundary = block
        if len(filtration) == 0:
            continue
        filtration = filtration.astype(str)
        C = filtration[:, 0]
        for i in range(1, filtration.shape[1]):
            C = np.char.add(C, " ")
            C = np.char.add(C, filtration[:, i])
        C = np.char.add(C, ";")
        D = np.fromiter(
            (" ".join(b.astype(str).tolist()) for b in boundary), dtype="<U11"
        )  # int32-> str is "<U11" #check np.array(1, dtype=np.int32).astype(str)
        str_blocks.append(np.char.add(C, D))

    np.savetxt("test.scc", np.concatenate(str_blocks), delimiter="", fmt="%s")
