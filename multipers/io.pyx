from multipers.simplex_tree_multi import SimplexTreeMulti
from gudhi import SimplexTree
import multipers.slicer as mps
import gudhi as gd
import numpy as np
import os
from libcpp cimport bool 
from shutil import which

from typing import Optional
from collections import defaultdict
doc_soft_urls = {
    "mpfree":"https://bitbucket.org/mkerber/mpfree/",
    "multi_chunk":"",
    "function_delaunay":"https://bitbucket.org/mkerber/function_delaunay/",
    "2pac":"https://gitlab.com/flenzen/2-parameter-persistent-cohomology",
}
doc_soft_urls = defaultdict(lambda:"<Unknown url>")
doc_soft_urls["mpfree"]= "https://bitbucket.org/mkerber/mpfree/"
doc_soft_urls["multi_chunk"]= "https://bitbucket.org/mkerber/multi_chunk"
doc_soft_urls["function_delaunay"]= "https://bitbucket.org/mkerber/function_delaunay/"
doc_soft_urls["twopac"]= "https://gitlab.com/flenzen/2-parameter-persistent-cohomology"



def _path_init(soft:str|os.PathLike):
    a = which(f"./{soft}")
    b = which(f"{soft}")
    if a:
        pathes[soft] = a
    elif b:
        pathes[soft] = b


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

def scc_parser(path: str):
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
        return len(line) == 0 or line[0] == "#"

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
            filtration_boundary = line.split(";")
            if len(filtration_boundary) == 1:
                # happens when last generators do not have a ";" in the end
                filtration_boundary.append(" ")
            filtration, boundary = filtration_boundary
            block_filtrations.append(
                    tuple(float(x) for x in filtration.split(" ") if len(x) > 0)
                    )
            block_boundaries.append(tuple(int(x) for x in boundary.split(" ") if len(x) > 0))
            counter -= 1
        blocks.append((np.asarray(block_filtrations, dtype=float), tuple(block_boundaries)))

    return blocks

def _put_temp_files_to_ram():
    global input_path,output_path
    shm_memory = "/tmp/"  # on unix, we can write in RAM instead of disk.
    if os.access(shm_memory, os.W_OK) and not input_path.startswith("/tmp/"):
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
""")

def scc_reduce_from_str(
        path:str|os.PathLike,
        bool full_resolution=True,
        int dimension: int | np.int64 = 1,
        bool clear: bool = True,
        id: str = "",  # For parallel stuff
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
        clear_io(output_path + id)
    return blocks

def reduce_complex(
        complex, # Simplextree, Slicer, or str
        bool full_resolution: bool = True,
        int dimension: int | np.int64 = 1,
        bool clear: bool = True,
        id: str = "",  # For parallel stuff
        bool verbose:bool=False,
        backend:Literal[*pathes.keys()]="mpfree"
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
    
    path = input_path+id
    if isinstance(complex, SimplexTreeMulti):
        complex.to_scc(
                path=path,
                rivet_compatible=False,
                strip_comments=False,
                ignore_last_generators=False,
                overwrite=True,
                reverse_block=True,
                )
    elif not isinstance(complex,str):
        # Assumes its a slicer
        blocks = mps.slicer2blocks(complex, degree=dimension)
        scc2disk(blocks,path=path)
    else:
        path = complex
    
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
        id:str = "",
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
    global input_path,output_path
    for x in [input_path,output_path] + list(args):
        if os.path.exists(x):
            os.remove(x)





from multipers.mma_structures cimport Finitely_critical_multi_filtration,uintptr_t,boundary_matrix,float,pair,vector,intptr_t
cdef extern from "multiparameter_module_approximation/format_python-cpp.h" namespace "Gudhi::multiparameter::mma":
    pair[boundary_matrix, vector[Finitely_critical_multi_filtration]] simplextree_to_boundary_filtration(uintptr_t)
    vector[pair[ vector[vector[float]],boundary_matrix]] simplextree_to_scc(uintptr_t)

def simplex_tree2boundary_filtrations(simplextree:SimplexTreeMulti | SimplexTree):
    """Computes a (sparse) boundary matrix, with associated filtration. Can be used as an input of approx afterwards.

    Parameters
    ----------
    simplextree: Gudhi or mma simplextree
        The simplextree defining the filtration to convert to boundary-filtration.

    Returns
    -------
    B:List of lists of ints
        The boundary matrix.
    F: List of 1D filtration
        The filtrations aligned with B; the i-th simplex of this simplextree has boundary B[i] and filtration(s) F[i].

    """
    cdef intptr_t cptr
    if isinstance(simplextree, SimplexTreeMulti):
        cptr = simplextree.thisptr
    elif isinstance(simplextree, SimplexTree):
        temp_st = gd.SimplexTreeMulti(simplextree, parameters=1)
        cptr = temp_st.thisptr
    else:
        raise TypeError("Has to be a simplextree")
    cdef pair[boundary_matrix, vector[Finitely_critical_multi_filtration]] cboundary_filtration = simplextree_to_boundary_filtration(cptr)
    boundary = cboundary_filtration.first
    multi_filtrations = np.array(Finitely_critical_multi_filtration.to_python(cboundary_filtration.second))
    return boundary, multi_filtrations

def simplextree2scc(simplextree:SimplexTreeMulti | SimplexTree, filtration_dtype=np.float32):
    """
    Turns a simplextree into a block / scc python format
    """
    cdef intptr_t cptr
    if isinstance(simplextree, SimplexTreeMulti):
        cptr = simplextree.thisptr
    elif isinstance(simplextree, SimplexTree):
        temp_st = gd.SimplexTreeMulti(simplextree, parameters=1)
        cptr = temp_st.thisptr
    else:
        raise TypeError("Has to be a simplextree")

    blocks = simplextree_to_scc(cptr)
    # reduces the space in memory
    blocks = [(np.asarray(f,dtype=filtration_dtype), tuple(b)) for f,b in blocks[::-1]] ## presentation is on the other order 
    return blocks

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
            for filtration,boundary in zip(*block):
                line = " ".join(tuple(str(x) for x in filtration)) + " ; " + " ".join(tuple(str(x) for x in boundary)) +"\n"
                f.write(line)
