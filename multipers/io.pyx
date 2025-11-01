import re
import tempfile
from gudhi import SimplexTree
import gudhi as gd
import numpy as np
import os
from shutil import which
from libcpp cimport bool
from typing import Optional, Literal
from collections import defaultdict
import itertools
import threading
import cython
cimport cython

current_doc_url = "https://davidlapous.github.io/multipers/"
doc_soft_urls = {
        "mpfree":"https://bitbucket.org/mkerber/mpfree/",
        "multi_chunk":"https://bitbucket.org/mkerber/multi_chunk/",
        "function_delaunay":"https://bitbucket.org/mkerber/function_delaunay/",
        "2pac":"https://gitlab.com/flenzen/2pac",
        "rhomboid_tiling":"https://github.com/odinhg/rhomboidtiling_newer_cgal_version",
        }
doc_soft_easy_install = {
        "mpfree":f"""
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
        "multi_chunk":f"""
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
        "function_delaunay":f"""
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
        "2pac":f"""
```sh
git clone {doc_soft_urls["2pac"]} 2pac
cd 2pac && mkdir build && cd build
cmake ..
make
cp 2pac $CONDA_PREFIX/bin
```
""",
        "rhomboid_tiling":f"""
git clone {doc_soft_urls["rhomboid_tiling"]} rhomboid_tiling
sh build.sh
cp orderk $CONDA_PREFIX/bin/rhomboid_tiling
"""
        }
doc_soft_urls = defaultdict(lambda:"<Unknown url>", doc_soft_urls)
doc_soft_easy_install = defaultdict(lambda:"<Unknown>", doc_soft_easy_install)

available_reduce_softs = Literal["mpfree","multi_chunk","2pac"]

def _path_init(soft:str|os.PathLike):
    a = which(f"./{soft}")
    b = which(f"{soft}")
    if a:
        pathes[soft] = a
    elif b:
        pathes[soft] = b

    if pathes[soft] is not None:
        verbose_arg = "> /dev/null 2>&1"
        test = os.system(pathes[soft] + " --help " + verbose_arg)
        if test:
            from warnings import warn
            warn(f"""
            Found external software {soft} at {pathes[soft]}
            but may not behave well.
            """)



cdef dict[str,str|None] pathes = {
        "mpfree":None,
        "2pac":None,
        "function_delaunay":None,
        "multi_chunk":None,
        }



## TODO : optimize with Python.h ?
def scc_parser(path: str| os.PathLike):
    """
    Parse an scc file into the scc python format, aka blocks.
    """
    pass_line_regex = re.compile(r"^\s*$|^#|^scc2020$")
    def valid_line(line):
        return pass_line_regex.match(line) is None
    parse_line_regex = re.compile(r"^(?P<filtration>[^;]+);(?P<boundary>[^;]*)$")
    cdef tuple[tuple[str,str]] clines 
    with open(path, "r") as f:
        lines =(x.strip() for x in f if valid_line(x))
        num_parameters = int(next(lines))
        sizes = np.cumsum(np.asarray([0] + next(lines).split(), dtype=np.int32))
        lines = (parse_line_regex.match(a) for a in lines)
        clines = tuple((a.group("filtration"),a.group("boundary")) for a in lines)
    # F = np.fromiter((a[0].split() for a in clines), dtype=np.dtype((np.float64,2)), count = sizes[-1])
    F = np.fromiter((np.fromstring(a[0], sep=r' ', dtype=np.float64) for a in clines), dtype=np.dtype((np.float64,num_parameters)), count = sizes[-1])
    
    # B = tuple(np.asarray(a[1].split(), dtype=np.int32) if len(a[1])>0 else np.empty(0, dtype=np.int32) for a in clines) ## TODO : this is very slow : optimize 
    B = tuple(np.fromstring(a[1], sep=' ', dtype=np.int32) for a in clines)
    # block_lines = (tuple(get_bf(x, num_parameters) for x in lines[sizes[i]:sizes[i+1]]) for i in range(len(sizes)-1))

    # blocks = [(np.asarray([x[0] for x in b if len(x)>0], dtype=float),tuple(x[1] for x in b))  for b in block_lines]
    blocks = [(F[sizes[i]:sizes[i+1]], B[sizes[i]:sizes[i+1]]) for i in range(len(sizes)-1)]

    return blocks




def _init_external_softwares(requires=[]):
    global pathes
    cdef bool any = False
    for soft,soft_path in pathes.items():
        if soft_path is None:
            _path_init(soft)
            any = any or (soft in requires) 

    if any:
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
def _check_available(soft:str):
    _init_external_softwares()
    return pathes.get(soft,None) is not None



def scc_reduce_from_str_to_slicer(
        path:str|os.PathLike,
        slicer,
        bool full_resolution=True,
        int dimension: int | np.int64 = 1,
        bool clear: bool = True,
        bool verbose:bool=False,
        backend:Literal["mpfree","multi_chunk","twopac"]="mpfree",
        shift_dimension=0
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
            command = (
                    f"{pathes[backend]} {more_verbose} {resolution_str} --dim={dimension} {path} {output_path} {verbose_arg}"
                    )
        elif backend == "multi_chunk":
            command = (
                    f"{pathes[backend]}  {path} {output_path} {verbose_arg}"
                    )
        elif backend in ["twopac", "2pac"]:
            command = (
                    f"{pathes[backend]} -f {path} --scc-input -n{dimension} --save-resolution-scc {output_path} {verbose_arg}"
                    )
        else:
            raise ValueError(f"Unsupported backend {backend}.")
        if verbose:
            print(f"Calling :\n\n {command}")
        os.system(command)

        slicer._build_from_scc_file(path=output_path, shift_dimension=shift_dimension)





def function_delaunay_presentation_to_slicer(
        slicer,
        point_cloud:np.ndarray,
        function_values:np.ndarray,
        bool clear:bool = True,
        bool verbose:bool=False,
        int degree = -1,
        bool multi_chunk = False,
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

    with tempfile.TemporaryDirectory(prefix="multipers", delete=clear) as tmpdir:
        input_path = os.path.join(tmpdir, "multipers_input.scc")
        output_path = os.path.join(tmpdir, "multipers_output.scc")

        backend = "function_delaunay"
        _init_external_softwares(requires=[backend])

        to_write = np.concatenate([point_cloud, function_values.reshape(-1,1)], axis=1)
        np.savetxt(input_path,to_write,delimiter=' ')
        verbose_arg = "> /dev/null 2>&1" if not verbose else ""
        degree_arg = f"--minpres {degree}" if degree >= 0 else ""
        multi_chunk_arg = "--multi-chunk" if multi_chunk else ""
        command = f"{pathes[backend]} {degree_arg} {multi_chunk_arg} {input_path} {output_path} {verbose_arg} --no-delaunay-compare"
        if verbose:
            print(command)
        os.system(command)

        slicer._build_from_scc_file(path=output_path, shift_dimension=-1 if degree <= 0 else degree-1 )

def rhomboid_tiling_to_slicer(
        slicer,
        point_cloud:np.ndarray,
        int k_max,
        int degree = -1,
        bool reduce=True,
        bool clear:bool = True,
        bool verbose:bool=False,
        bool multi_chunk = False,
        ):
    """TODO"""
    global pathes
    backend = "rhomboid_tiling"
    _init_external_softwares(requires=[backend])
    if point_cloud.ndim != 2 or not point_cloud.shape[1] in [2,3]:
        raise ValueError("point_cloud should be a 2d array of shape (-,2) or (-,3). Got {point_cloud.shape=}")
    with tempfile.TemporaryDirectory(prefix="multipers", delete=clear) as tmpdir:
        input_path = os.path.join(tmpdir, "point_cloud.txt")
        output_path = os.path.join(tmpdir, "multipers_output.scc")
        np.savetxt(input_path,point_cloud,delimiter=' ')

        verbose_arg = "> /dev/null 2>&1" if not verbose else ""
        degree_arg = f"--minpres {degree}" if degree >= 0 else ""
        multi_chunk_arg = "--multi-chunk" if multi_chunk else ""
        command = f"{pathes[backend]} {input_path} {output_path} {point_cloud.shape[1]} {k_max} scc {degree}"
        if verbose:
            print(command)
        os.system(command)
        slicer._build_from_scc_file(path=output_path, shift_dimension=-1 if degree <= 0 else degree-1 )





@cython.boundscheck(False)
@cython.wraparound(False)
def scc2disk(
        stuff,
        path:str|os.PathLike,
        int num_parameters = -1,
        bool reverse_block = False,
        bool rivet_compatible = False,
        bool ignore_last_generators = False,
        bool strip_comments = False,
        ):
    """
    Writes a scc python format / blocks into a file.
    """
    if num_parameters == -1:
        for block in stuff:
            if len(block[0]) == 0:
                continue
            num_gens, num_parameters_= np.asarray(block[0]).shape 
            num_parameters = num_parameters_
            break
    assert num_parameters > 0, f"Invalid number of parameters {num_parameters}"

    if reverse_block:	stuff.reverse()
    with open(path, "w") as f:
        f.write("scc2020\n") if not rivet_compatible else f.write("firep\n")
        if not strip_comments and not rivet_compatible: f.write("# Number of parameters\n")
        if rivet_compatible:
            assert num_parameters == 2
            f.write("Filtration 1\n")
            f.write("Filtration 2\n")
        else:
            f.write(f"{num_parameters}\n")

        if not strip_comments: f.write("# Sizes of generating sets\n")
        for block in stuff: f.write(f"{len(block[0])} ")
        f.write("\n")
        for i,block in enumerate(stuff):
            if (rivet_compatible or ignore_last_generators) and i == len(stuff)-1: continue
            if not strip_comments: f.write(f"# Block of dimension {len(stuff)-1-i}\n")
            filtration, boundary = block
            filtration = np.asarray(filtration).astype(str)
            # boundary = tuple(x.astype(str) for x in boundary)
            f.write(" ".join(itertools.chain.from_iterable(
                ((*(f.tolist()),  ";",  *(np.asarray(b).astype(str).tolist()), "\n")
                    for f,b in zip(filtration, boundary))
                )
            ))
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
        path:str|os.PathLike,
        num_parameters = -1,
        reverse_block = False,
        rivet_compatible = False,
        ignore_last_generators = False,
        strip_comments = False,
        ):
    """
    Writes a scc python format / blocks into a file.
    """
    if num_parameters == -1:
        for block in stuff:
            if len(block[0]) == 0:
                continue
            num_gens, num_parameters_= np.asarray(block[0]).shape 
            num_parameters = num_parameters_
            break
    assert num_parameters > 0, f"Invalid number of parameters {num_parameters}"

    if reverse_block:	stuff.reverse()
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
    for i,block in enumerate(stuff):
        if (rivet_compatible or ignore_last_generators) and i == len(stuff)-1: continue
        if not strip_comments: 
            str_blocks.append([f"# Block of dimension {len(stuff)-1-i}"])
        filtration, boundary = block
        if len(filtration) == 0:
            continue
        filtration = filtration.astype(str)
        C = filtration[:,0]
        for i in range(1,filtration.shape[1]):
            C = np.char.add(C," ")
            C = np.char.add(C,filtration[:,i])
        C = np.char.add(C, ";")
        D = np.fromiter((" ".join(b.astype(str).tolist()) for b in boundary), dtype="<U11") #int32-> str is "<U11" #check np.array(1, dtype=np.int32).astype(str)
        str_blocks.append(np.char.add(C,D))

    np.savetxt("test.scc", np.concatenate(str_blocks), delimiter="", fmt="%s")
