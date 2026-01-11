import cython
from multipers.vector_interface cimport *
import numpy as np
import multipers as mp

@cython.boundscheck(False)
@cython.wraparound(False) 
cdef vector[pair[double, double]] array_view_to_vect_pair(double[:, :] arr_view) noexcept nogil:
    cdef int n = arr_view.shape[0]
    cdef vector[pair[double, double]] result_vector
    result_vector.resize(n)
    for i in range(n):
        result_vector[i] = pair[double, double](arr_view[i, 0], arr_view[i, 1])
    return result_vector

def _aida(col_degrees, row_degrees, matrix):
  cdef multipers_interface_input stuff = multipers_interface_input(col_degrees, row_degrees, matrix)
  cdef AIDA_functor truc 
  truc.config.show_info = True
  truc.config.sort_output = True
  truc.config.sort = True
  cdef multipers_interface_output stuff2 = truc.multipers_interface(stuff)
  out = []
  for i in range(stuff2.summands.size()):
    out.append((stuff2.summands[i].col_degrees, stuff2.summands[i].row_degrees, stuff2.summands[i].matrix))
  return out

def aida(s, bool sort=True, bool verbose=False, bool progress = False):
    """
    Decomposes (a minimal presentation of a) 2-parameter persistence module as
    a direct sum of indecomposables.

    From [Decomposing Multiparameter Persistence Modules](https://doi.org/10.4230/LIPIcs.SoCG.2025.41).

    Parameters:
     - s : The slicer to reduce. Has to be a minimal presentation.
     - verbose : shows log.
     - progress : shows a progress bar
     - sort : sorts the input first in a colexical order (debug)
    """

    from multipers.slicer import is_slicer
    if not is_slicer(s):
        raise ValueError(f"Input has to be a slicer. Got {type(s)=}.")
    if not s.is_minpres:
        raise ValueError(f"AIDA takes a minimal presentation as an input. Got {s.minpres_degree=}.")
    if s.num_parameters != 2 or not s.is_minpres:
        raise ValueError(f"AIDA is only compatible with 2-parameter minimal presentations. Got {s.num_parameters=} and {s.is_minpres=}.")
    cdef bool is_squeezed = s.is_squeezed

    cdef int degree = s.minpres_degree 
    if sort:
        s = s.to_colexical()
    F = np.asarray(s.get_filtrations())
    D = s.get_dimensions()
    cdef double[:,:] row_degree_ = np.asarray(F[D==degree],   dtype = np.float64)
    cdef double[:,:] col_degree_ = np.asarray(F[D==degree+1], dtype = np.float64)
    cdef vector[pair[double,double]] row_degree = array_view_to_vect_pair(row_degree_)
    cdef vector[pair[double,double]] col_degree = array_view_to_vect_pair(col_degree_)
    i,j = np.searchsorted(D, [degree+1,degree+2])
    cdef vector[vector[int]] matrix = s.get_boundaries()[i:j]

    cdef AIDA_functor truc 
    cdef multipers_interface_input stuff 
    cdef multipers_interface_output stuff2
    with nogil:
        truc.config.show_info = verbose
        truc.config.sort_output = False
        truc.config.sort = sort
        truc.config.progress = progress
        stuff = multipers_interface_input(col_degree, row_degree, matrix)
        stuff2 = truc.multipers_interface(stuff)
    out = []
    _Slicer = mp.Slicer(return_type_only=True, dtype=np.float64)
    out = [_Slicer() for _ in range(stuff2.summands.size())]
    dim_container_ = s.get_dimensions().copy()
    cdef int32_t[:] dim_container = np.asarray(dim_container_, dtype=np.int32)
    cdef list boundary_container
    cdef vector[pair[double,double]] FR
    cdef vector[pair[double,double]] FG
    cdef vector[vector[int]] B
    for i in range(stuff2.summands.size()):
        FR = stuff2.summands[i].col_degrees
        FG = stuff2.summands[i].row_degrees
        B = stuff2.summands[i].matrix

        for j in range(FG.size()):
            dim_container[j] = degree
        for j in range(FG.size(),FG.size()+FR.size()):
            dim_container[j] = degree +1

        boundary_container = [[] for _ in range(FG.size())]
        boundary_container.extend(B)
        
        if FR.size() == 0:
            filtration_values = np.asarray(FG)
        else:
            filtration_values = np.concatenate([FG,FR], dtype=np.float64)

        s_summand = _Slicer(
            boundary_container,
            dim_container[:FG.size()+FR.size()],
            filtration_values
        )
        if s.is_squeezed:
            s_summand.filtration_grid = s.filtration_grid
            s_summand._clean_filtration_grid()
        out[i] = s_summand
            
    return out

def one_criticalify(
        slicer,
        bool reduce=False,
        degree:Optional[int]=None,
        bool clear = True,
        swedish:Optional[bool] = None,
        bool verbose = False,
        bool kcritical=False,
        str algo:Literal["path","tree"]="path",
    ):
    """
    Computes a free implicit representation  of a given multi-critical
    multifiltration of a given homological degree (i.e., for a given
    homological degree, a quasi-isomorphic 1-critical filtration), or free
    resolution of the multifiltration (i.e., quasi-isomorphic 1-critical chain
    complex).

    From [Fast free resolutions of bifiltered chain complexes](https://doi.org/10.48550/arXiv.2512.08652), 
    whose code is available here: https://bitbucket.org/mkerber/multi_critical

    Parameters:
     - slicer : multicritical filtration to represent
     - reduce : returns a (or multiple, see degree) minimal presentation(s) instead of the chain complex.
     - degree : If an int is given, and `reduce` is true, only returns the minimal presentation of this degree.
                If None is given and `reduce` is true, returns a minimal presentation of all possible degrees.
                If reduce is false : has no effect.
     - clear : Clears the temporary files.
     - swedish : if True, `reduce=True` and `degree=None` skips the computation of the 1critical chain complex,
                 and directly (sequentially) computes the individual minimal presentations.
     - verbose : shows log
     - kcritical : do not use
     - algo : see ref.
    """
    from multipers.io import _multi_critical_from_slicer
    from multipers.slicer import is_slicer
    if not is_slicer(slicer):
        raise ValueError(f"Invalid input. Expected `SlicerType` got {type(slicer)=}.")
    if not slicer.is_kcritical:
        return slicer
    if slicer.is_squeezed:
        F = slicer.filtration_grid
    else:
        F = None
    out = _multi_critical_from_slicer(
           slicer, reduce=reduce, algo=algo,
           degree=degree, clear=clear,
           swedish=swedish, verbose=verbose,
           kcritical=kcritical
    )
    if is_slicer(out, allow_minpres=False):
        out.filtration_grid = F
    else:
        for stuff in out:
            stuff.filtration_grid = F
    return out

def minimal_presentation(
        slicer,
        int degree = -1, 
        degrees:Iterable[int]=[],
        str backend:Literal["mpfree", "2pac", ""]="mpfree", 
        int n_jobs = -1,
        bool force=False,
        bool auto_clean = True,
        ):
    """
    Computes a minimal presentation a (1-critical) multifiltered  complex.

    From [Fast minimal presentations of bi-graded persistence modules](https://doi.org/10.1137/1.9781611976472.16),
    whose code is available here: https://bitbucket.org/mkerber/mpfree

    Backends differents than `mpfree` are unstable.

    Parameters:
     - slicer : the filtration/free implicit representation to reduce
     - degree : the homological degree to reduce
     - degrees : a list of homological degrees to reduce. Output will be a list.
     - backend : a callable `scc`-compatible backend
     - n_jobs : process minpres in parallel if degrees is given
     - force : if input is already reduced, force the re-computation of the minimal presentation.
     - auto_clean : if input is squeezed, some filtraton values may disappear.
       This is a postprocessing to remove unnecessary coordinates.
    """
    from multipers.io import _init_external_softwares, scc_reduce_from_str_to_slicer
    from joblib import Parallel, delayed
    from multipers.slicer import is_slicer
    from multipers import Slicer
    import os
    import tempfile

    if is_slicer(slicer) and slicer.is_minpres and not force:
        from warnings import warn
        warn(f"(unnecessary computation) The slicer seems to be already reduced, from homology of degree {slicer.minpres_degree}.")
        return slicer
    _init_external_softwares(requires=[backend])
    if len(degrees)>0:
        def todo(int degree):
            return minimal_presentation(slicer, degree=degree, backend=backend, force=force, auto_clean=auto_clean)
        return tuple(
          Parallel(n_jobs=n_jobs, backend="threading")(delayed(todo)(d) for d in degrees)
        )
    assert degree>=0, f"Degree not provided."
    if not np.any(slicer.get_dimensions() == degree):
        return type(slicer)()
    dimension = slicer.dimension - degree # latest  = L-1, which is empty, -1 for degree 0, -2 for degree 1 etc.
    with tempfile.TemporaryDirectory(prefix="multipers") as tmpdir:
        tmp_path = os.path.join(tmpdir, "multipers.scc")
        slicer.to_scc(path=tmp_path, strip_comments=True, degree=degree-1, unsqueeze = False)
        new_slicer = type(slicer)()
        if backend=="mpfree":
            shift_dimension=degree-1
        else:
            shift_dimension=degree
        scc_reduce_from_str_to_slicer(path=tmp_path, slicer=new_slicer, dimension=dimension, backend=backend, shift_dimension=shift_dimension)

        new_slicer.minpres_degree = degree
        new_slicer.filtration_grid = slicer.filtration_grid if slicer.is_squeezed else None
        if new_slicer.is_squeezed and auto_clean:
            new_slicer = new_slicer._clean_filtration_grid()
        return new_slicer
