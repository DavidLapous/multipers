
from libcpp.utility cimport pair
from libc.stdint cimport intptr_t, uint16_t, uint32_t, int32_t, int64_t
from libcpp.vector cimport vector
from libcpp cimport bool, float

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
    # vector[pair[double,double]] col_degrees
    # vector[pair[double,double]] row_degrees
    # vector[vector[int]] matrix

  cdef cppclass AIDA_functor "aida::AIDA_functor":
    AIDA_functor() except + nogil
    multipers_interface_output multipers_interface(multipers_interface_input&) except + nogil
    aida_config config



