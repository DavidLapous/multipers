import re
from gudhi import SimplexTree
import multipers.slicer as mps
import gudhi as gd
import numpy as np
import os
from shutil import which
from libcpp cimport bool
from typing import Optional, Literal
from collections import defaultdict
import itertools
import threading

# from multipers.filtration_conversions cimport *
# from multipers.mma_structures cimport boundary_matrix,float,pair,vector,intptr_t
# cimport numpy as cnp

doc_soft_urls = {
        "mpfree":"https://bitbucket.org/mkerber/mpfree/",
        "multi_chunk":"",
        "function_delaunay":"https://bitbucket.org/mkerber/function_delaunay/",
        "2pac":"https://gitlab.com/flenzen/2pac",
        }
doc_soft_easy_install = {
        "mpfree":f"""
```sh
git clone {doc_soft_urls["mpfree"]}
cd mpfree
sudo cp mpfree /usr/bin/
cd .. 
rm -rf mpfree
```
        """,
        "multi_chunk":f"""
```sh
git clone {doc_soft_urls["multi_chunk"]}
cd multi_chunk
sudo cp multi_chunk /usr/bin/
cd .. 
rm -rf multi_chunk
```
        """,
        "function_delaunay":f"""
```sh
git clone {doc_soft_urls["function_delaunay"]}
cd function_delaunay
sudo cp main /usr/bin/function_delaunay
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
sudo cp 2pac /usr/bin
```
""",
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



pathes = {
        "mpfree":None,
        "2pac":None,
        "function_delaunay":None,
        "multi_chunk":None,
        }

# mpfree_in_path:str|os.PathLike = "multipers_mpfree_input.scc"
# mpfree_out_path:str|os.PathLike = "multipers_mpfree_output.scc"
# twopac_in_path:str|os.PathLike = "multipers_twopac_input.scc"
# twopac_out_path:str|os.PathLike = "multipers_twopac_output.scc"
# multi_chunk_in_path:str|os.PathLike = "multipers_multi_chunk_input.scc"
# multi_chunk_out_path:str|os.PathLike = "multipers_multi_chunk_output.scc"
# function_delaunay_out_path:str|os.PathLike = "function_delaunay_output.scc"
# function_delaunay_in_path:str|os.PathLike = "function_delaunay_input.txt" # point cloud
input_path:str|os.PathLike = "multipers_input.scc"
output_path:str|os.PathLike = "multipers_output.scc"



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
    F = np.fromiter((a[0].split() for a in clines), dtype=np.dtype((np.float32,2)), count = sizes[-1])
    
    B = tuple(np.asarray(a[1].split(), dtype=np.int32) if len(a[1])>0 else np.empty(0, dtype=np.int32) for a in clines) ## TODO : this is very slow : optimize 
    # block_lines = (tuple(get_bf(x, num_parameters) for x in lines[sizes[i]:sizes[i+1]]) for i in range(len(sizes)-1))

    # blocks = [(np.asarray([x[0] for x in b if len(x)>0], dtype=float),tuple(x[1] for x in b))  for b in block_lines]
    blocks = [(F[sizes[i]:sizes[i+1]], B[sizes[i]:sizes[i+1]]) for i in range(len(sizes)-1)]

    return blocks


def scc_parser__old(path: str):
    """
    Parse an scc file into the scc python format, aka blocks.
    """
    with open(path, "r") as f:
        lines = f.readlines()
    # Find scc2020
    while lines[0].strip() != "scc2020":
        lines = lines[1:]
    lines = lines[1:]
    # stripped scc2020 we can start

    def pass_line(line):
        return re.match(r"^\s*$|^#", line) is not None

    for i, line in enumerate(lines):
        line = line.strip()
        if pass_line(line):
            continue
        num_parameters = int(line)
        lines = lines[i + 1 :]
        break

    block_sizes = []

    for i, line in enumerate(lines):
        line = line.strip()
        if pass_line(line):
            continue
        block_sizes = tuple(int(i) for i in line.split(" "))
        lines = lines[i + 1 :]
        break
    blocks = []
    cdef int counter
    for block_size in block_sizes:
        counter = block_size
        block_filtrations = []
        block_boundaries = []
        for i, line in enumerate(lines):
            if counter == 0:
                lines = lines[i:]
                break
            line = line.strip()
            if pass_line(line):
                continue
            splitted_line = re.match(r"^(?P<floats>[^;]+);(?P<ints>[^;]*)$", line)
            filtrations = np.asarray(splitted_line.group("floats").split(), dtype=float)
            boundary = np.asarray(splitted_line.group("ints").split(), dtype=int)
            block_filtrations.append(filtrations)
            block_boundaries.append(boundary)
            # filtration_boundary = line.split(";")
            # if len(filtration_boundary) == 1:
            #     # happens when last generators do not have a ";" in the end
            #     filtration_boundary.append(" ")
            # filtration, boundary = filtration_boundary
            # block_filtrations.append(
            #         tuple(float(x) for x in filtration.split(" ") if len(x) > 0)
            #         )
            # block_boundaries.append(tuple(int(x) for x in boundary.split(" ") if len(x) > 0))
            counter -= 1
        blocks.append((np.asarray(block_filtrations, dtype=float), tuple(block_boundaries)))

    return blocks



def _put_temp_files_to_ram():
    global input_path,output_path
    shm_memory = "/tmp/"  # on unix, we can write in RAM instead of disk.
    if os.access(shm_memory, os.W_OK) and not input_path.startswith(shm_memory):
        input_path = shm_memory + input_path
        output_path = shm_memory + output_path

def _init_external_softwares(requires=[]):
    global pathes
    cdef bool any = False
    for soft,soft_path in pathes.items():
        if soft_path is None:
            _path_init(soft)
            any = any or (soft in requires) 

    if any:
        _put_temp_files_to_ram()
        for soft in requires:
            if pathes[soft] is None:
                global doc_soft_urls
                raise ValueError(f"""
Did not found {soft}.
Install it from {doc_soft_urls[soft]}, and put it in your current directory,
or in you $PATH.
For instance:
{doc_soft_easy_install[soft]}
                                 """)

def scc_reduce_from_str(
        path:str|os.PathLike,
        bool full_resolution=True,
        int dimension: int | np.int64 = 1,
        bool clear: bool = True,
        id: Optional[str] = None,  # For parallel stuff
        bool verbose:bool=False,
        backend:Literal["mpfree","multi_chunk","twopac"]="mpfree"
        ):
    """
    Computes a minimal presentation of the file in path,
    using mpfree.

    path:PathLike
    full_resolution: bool
    dimension: int, presentation dimension to consider
    clear: bool, removes temporary files if True
    id: str, temporary files are of this id, allowing for multiprocessing
    verbose: bool
    backend: "mpfree", "multi_chunk" or "2pac"
    """
    global pathes, input_path, output_path
    if pathes[backend] is None:
        _init_external_softwares(requires=[backend])


    resolution_str = "--resolution" if full_resolution else ""
    # print(mpfree_in_path + id, mpfree_out_path + id)
    if id is None:
        id = str(threading.get_native_id())
    if not os.path.exists(path):
        raise ValueError(f"No file found at {path}.")
    if os.path.exists(output_path + id):
        os.remove(output_path + id)
    verbose_arg = "> /dev/null 2>&1" if not verbose else ""
    if backend == "mpfree":
        more_verbose = "-v" if verbose else ""
        command = (
                f"{pathes[backend]} {more_verbose} {resolution_str} --dim={dimension} {path} {output_path+id} {verbose_arg}"
                )
    elif backend == "multi_chunk":
        command = (
                f"{pathes[backend]}  {path} {output_path+id} {verbose_arg}"
                )
    elif backend in ["twopac", "2pac"]:
        command = (
                f"{pathes[backend]} -f {path} --scc-input -n{dimension} --save-resolution-scc {output_path+id} {verbose_arg}"
                )
    else:
        raise ValueError(f"Unsupported backend {backend}.")
    if verbose:
        print(f"Calling :\n\n {command}")
    os.system(command)

    blocks = scc_parser(output_path + id)
    if clear:
        clear_io(input_path+id, output_path + id)
    return blocks

def reduce_complex(
        complex, # Simplextree, Slicer, or str
        bool full_resolution: bool = True,
        int dimension: int | np.int64 = 1,
        bool clear: bool = True,
        id: Optional[str]=None,  # For parallel stuff
        bool verbose:bool=False,
        backend:available_reduce_softs="mpfree"
        ):
    """
    Computes a minimal presentation of the file in path,
    using `backend`.

    simplextree
    full_resolution: bool
    dimension: int, presentation dimension to consider
    clear: bool, removes temporary files if True
    id: str, temporary files are of this id, allowing for multiprocessing
    verbose: bool
    """

    from multipers.simplex_tree_multi import is_simplextree_multi
    if id is None:
        id = str(threading.get_native_id())
    path = input_path+id
    if is_simplextree_multi(complex):
        complex.to_scc(
                path=path,
                rivet_compatible=False,
                strip_comments=False,
                ignore_last_generators=False,
                overwrite=True,
                reverse_block=False,
                )
        dimension = complex.dimension - dimension
    elif isinstance(complex,str):
        path = complex
    elif isinstance(complex, list) or isinstance(complex, tuple):
        scc2disk(complex,path=path)
    else:
        # Assumes its a slicer
        blocks = mps.slicer2blocks(complex)
        scc2disk(blocks,path=path)
        dimension = len(blocks) -2 -dimension

    return scc_reduce_from_str(
            path=path,
            full_resolution=full_resolution,
            dimension=dimension,
            clear=clear,
            id=id,
            verbose=verbose,
            backend=backend
            )




def function_delaunay_presentation(
        point_cloud:np.ndarray,
        function_values:np.ndarray,
        id:Optional[str] = None,
        bool clear:bool = True,
        bool verbose:bool=False,
        int degree = -1,
        bool multi_chunk = False,
        ):
    """
    Computes a function delaunay presentation, and returns it as blocks.

    points : (num_pts, n) float array
    grades : (num_pts,) float array
    degree (opt) : if given, computes a minimal presentation of this homological degree first
    clear:bool, removes temporary files if true
    degree: computes minimal presentation of this degree if given
    verbose : bool
    """
    if id is None:
        id = str(threading.get_native_id())
    global input_path, output_path, pathes
    backend = "function_delaunay"
    if  pathes[backend] is None :
        _init_external_softwares(requires=[backend])

    to_write = np.concatenate([point_cloud, function_values.reshape(-1,1)], axis=1)
    np.savetxt(input_path+id,to_write,delimiter=' ')
    verbose_arg = "> /dev/null 2>&1" if not verbose else ""
    degree_arg = f"--minpres {degree}" if degree > 0 else ""
    multi_chunk_arg = "--multi-chunk" if multi_chunk else ""
    if os.path.exists(output_path + id):
        os.remove(output_path+ id)
    command = f"{pathes[backend]} {degree_arg} {multi_chunk_arg} {input_path+id} {output_path+id} {verbose_arg} --no-delaunay-compare"
    if verbose:
        print(command)
    os.system(command)

    blocks = scc_parser(output_path + id)
    if clear:
        clear_io(output_path + id, input_path + id)
    return blocks



def clear_io(*args):
    """Removes temporary files"""
    global input_path,output_path
    for x in [input_path,output_path] + list(args):
        if os.path.exists(x):
            os.remove(x)






# cdef extern from "multiparameter_module_approximation/format_python-cpp.h" namespace "Gudhi::multiparameter::mma":
    # pair[boundary_matrix, vector[Finitely_critical_multi_filtration[double]]] simplextree_to_boundary_filtration(intptr_t)
    # vector[pair[ vector[vector[float]],boundary_matrix]] simplextree_to_scc(intptr_t)
    # vector[pair[ vector[vector[vector[float]]],boundary_matrix]] function_simplextree_to_scc(intptr_t)
    # pair[vector[vector[float]],boundary_matrix ] simplextree_to_ordered_bf(intptr_t)

# def simplex_tree2boundary_filtrations(simplextree:SimplexTreeMulti | SimplexTree):
#     """Computes a (sparse) boundary matrix, with associated filtration. Can be used as an input of approx afterwards.
#
#     Parameters
#     ----------
#     simplextree: Gudhi or mma simplextree
#         The simplextree defining the filtration to convert to boundary-filtration.
#
#     Returns
#     -------
#     B:List of lists of ints
#         The boundary matrix.
#     F: List of 1D filtration
#         The filtrations aligned with B; the i-th simplex of this simplextree has boundary B[i] and filtration(s) F[i].
#
#     """
#     cdef intptr_t cptr
#     if isinstance(simplextree, SimplexTreeMulti):
#         cptr = simplextree.thisptr
#     elif isinstance(simplextree, SimplexTree):
#         temp_st = gd.SimplexTreeMulti(simplextree, parameters=1)
#         cptr = temp_st.thisptr
#     else:
#         raise TypeError("Has to be a simplextree")
#     cdef pair[boundary_matrix, vector[Finitely_critical_multi_filtration[double]]] cboundary_filtration = simplextree_to_boundary_filtration(cptr)
#     boundary = cboundary_filtration.first
#     # multi_filtrations = np.array(<vector[vector[float]]>Finitely_critical_multi_filtration.to_python(cboundary_filtration.second))
#     cdef cnp.ndarray[double, ndim=2] multi_filtrations = _fmf2numpy_f64(cboundary_filtration.second)
#     return boundary, multi_filtrations

# def simplextree2scc(simplextree:SimplexTreeMulti | SimplexTree, filtration_dtype=np.float32, bool flattened=False):
#     """
#     Turns a simplextree into a (simplicial) module presentation.
#     """
#     cdef intptr_t cptr
#     cdef bool is_function_st = False
#     if isinstance(simplextree, SimplexTreeMulti):
#         cptr = simplextree.thisptr
#         is_function_st = simplextree._is_function_simplextree
#     elif isinstance(simplextree, SimplexTree):
#         temp_st = gd.SimplexTreeMulti(simplextree, parameters=1)
#         cptr = temp_st.thisptr
#     else:
#         raise TypeError("Has to be a simplextree")
#     
#     cdef pair[vector[vector[float]], boundary_matrix] out
#     if flattened:
#         out = simplextree_to_ordered_bf(cptr)
#         return np.asarray(out.first,dtype=filtration_dtype), tuple(out.second)
#
#     if is_function_st:
#         blocks = function_simplextree_to_scc(cptr)
#     else:
#         blocks = simplextree_to_scc(cptr)
#     # reduces the space in memory
#     if is_function_st:
#         blocks = [(tuple(f), tuple(b)) for f,b in blocks[::-1]]
#     else:
#         blocks = [(np.asarray(f,dtype=filtration_dtype), tuple(b)) for f,b in blocks[::-1]] ## presentation is on the other order 
#     return blocks+[(np.empty(0,dtype=filtration_dtype),[])]

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
                ((*(f.tolist()),";",*(np.asarray(b).astype(str).tolist()),"\n") for f,b in zip(filtration, boundary))
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
