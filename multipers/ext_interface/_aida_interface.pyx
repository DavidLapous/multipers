import cython
import numpy as np
cimport numpy as cnp
from libc.stdint cimport int32_t, int64_t
from libcpp.utility cimport pair
from libcpp.vector cimport vector
from libcpp cimport bool

from multipers.ext_interface._helper cimport vect_vect_boundary_to_numpy_slices, view_2_2Darray, vect_pair_double_to_array
import multipers as mp


cdef extern from "config.hpp":
  cdef cppclass aida_config "aida::AIDA_config":
    bool sort
    bool exhaustive
    bool brute_force
    bool sort_output
    bool compare_both
    bool exhaustive_test
    bool progress
    bool save_base_change
    bool turn_off_hom_optimisation
    bool show_info
    bool compare_hom
    bool supress_col_sweep
    bool alpha_hom


cdef extern from "aida_interface.hpp":

  cdef cppclass multipers_interface_input "aida::multipers_interface_input<int>":
    multipers_interface_input(const vector[pair[double,double]]&, const vector[pair[double,double]]&, const vector[vector[int]]&) except + nogil
    multipers_interface_input() except + nogil
    vector[pair[double,double]] col_degrees
    vector[pair[double,double]] row_degrees
    vector[vector[int]] matrix

  cdef cppclass multipers_interface_output "aida::multipers_interface_output<int>":
    multipers_interface_output() except + nogil
    vector[multipers_interface_input] summands

  cdef cppclass AIDA_functor "aida::AIDA_functor":
    AIDA_functor() except + nogil
    multipers_interface_output multipers_interface(multipers_interface_input&) except + nogil
    aida_config config


def _is_available():
    return True


def aida(s, bint sort=True, bint verbose=False, bint progress=False):
    """
    Decomposes (a minimal presentation of a) 2-parameter persistence module as
    a direct sum of indecomposables.

    From [Decomposing Multiparameter Persistence Modules](https://doi.org/10.4230/LIPIcs.SoCG.2025.41).
    """
    from multipers.slicer import is_slicer

    if not is_slicer(s):
        raise ValueError(f"Input has to be a slicer. Got {type(s)=}.")
    if not s.is_minpres:
        raise ValueError(f"AIDA takes a minimal presentation as an input. Got {s.minpres_degree=}.")
    if s.num_parameters != 2 or not s.is_minpres:
        raise ValueError(
            f"AIDA is only compatible with 2-parameter minimal presentations. "
            f"Got {s.num_parameters=} and {s.is_minpres=}."
        )

    cdef bint is_squeezed = s.is_squeezed
    cdef int degree = s.minpres_degree

    if sort:
        s = s.to_colexical()

    F = np.asarray(s.get_filtrations(view=False))
    D = s.get_dimensions()
    cdef double[:, :] row_degree_ = np.asarray(F[D == degree], dtype=np.float64)
    cdef double[:, :] col_degree_ = np.asarray(F[D == degree + 1], dtype=np.float64)
    cdef vector[pair[double, double]] row_degree = view_2_2Darray(row_degree_)
    cdef vector[pair[double, double]] col_degree = view_2_2Darray(col_degree_)

    cdef int64_t i, j
    i, j = np.searchsorted(D, [degree + 1, degree + 2])
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
    _Slicer = mp.Slicer(s, return_type_only=True, dtype=np.float64)
    out = [_Slicer() for _ in range(stuff2.summands.size())]
    dim_container_ = s.get_dimensions().copy()
    cdef int32_t[:] dim_container = np.asarray(dim_container_, dtype=np.int32)
    cdef list boundary_container
    cdef vector[pair[double, double]] FR
    cdef vector[pair[double, double]] FG
    cdef vector[vector[int]] B
    cdef object filtration_values

    for i in range(stuff2.summands.size()):
        FR = stuff2.summands[i].col_degrees
        FG = stuff2.summands[i].row_degrees
        B = stuff2.summands[i].matrix

        for j in range(FG.size()):
            dim_container[j] = degree
        for j in range(FG.size(), FG.size() + FR.size()):
            dim_container[j] = degree + 1

        boundary_container = [[] for _ in range(FG.size())]
        boundary_container.extend(vect_vect_boundary_to_numpy_slices(B))

        if FR.size() == 0:
            filtration_values = vect_pair_double_to_array(FG)
        else:
            filtration_values = np.concatenate(
                [vect_pair_double_to_array(FG), vect_pair_double_to_array(FR)],
                dtype=np.float64,
            )

        s_summand = _Slicer(
            boundary_container,
            dim_container[: FG.size() + FR.size()],
            filtration_values,
        )
        if is_squeezed:
            s_summand.filtration_grid = s.filtration_grid
            s_summand._clean_filtration_grid()
        out[i] = s_summand

    return out
