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
