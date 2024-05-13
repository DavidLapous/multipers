# This file is part of the Gudhi Library - https://gudhi.inria.fr/ - which is released under MIT.
# See file LICENSE or go to https://gudhi.inria.fr/licensing/ for full license details.
# Author(s):       Vincent Rouvreau
#
# Copyright (C) 2016 Inria
#
# Modification(s):
#  - 2022 David Loiseaux, Hannah Schreiber: adapt for multipersistence. 
#   - YYYY/MM Author: Description of the modification

from cython cimport numeric
from libcpp.vector cimport vector
from libcpp.utility cimport pair
from libcpp cimport bool
from libcpp.string cimport string
from libcpp.map cimport map

from libc.stdint cimport intptr_t, int32_t

__author__ = "Vincent Rouvreau"
__copyright__ = "Copyright (C) 2016 Inria"
__license__ = "MIT"


from multipers.filtrations cimport *


ctypedef int dimension_type
ctypedef vector[int] simplex_type
ctypedef vector[simplex_type] simplex_list
ctypedef vector[pair[pair[int,int], pair[double, double]]] edge_list
ctypedef vector[int] euler_char_list

ctypedef vector[unsigned int] boundary_type
ctypedef vector[boundary_type] boundary_matrix

cdef extern from "Simplex_tree_multi_interface.h" namespace "Gudhi::multiparameter::python_interface":

  cdef cppclass Simplex_tree_multi_simplex_handle[F=*]:
    pass

  cdef cppclass Simplex_tree_multi_simplices_iterator[F=*]:
    Simplex_tree_multi_simplices_iterator() nogil
    Simplex_tree_multi_simplex_handle& operator*() nogil
    Simplex_tree_multi_simplices_iterator operator++() nogil
    bint operator!=(Simplex_tree_multi_simplices_iterator) nogil

  cdef cppclass Simplex_tree_multi_skeleton_iterator[F=*]:
    Simplex_tree_multi_skeleton_iterator() nogil
    Simplex_tree_multi_simplex_handle& operator*() nogil
    Simplex_tree_multi_skeleton_iterator operator++() nogil
    bint operator!=(Simplex_tree_multi_skeleton_iterator) nogil

  cdef cppclass Simplex_tree_multi_boundary_iterator[F=*]:
    Simplex_tree_multi_boundary_iterator() nogil
    Simplex_tree_multi_simplex_handle& operator*() nogil
    Simplex_tree_multi_boundary_iterator operator++() nogil
    bint operator!=(Simplex_tree_multi_boundary_iterator) nogil


  cdef cppclass Simplex_tree_multi_interface[F=*, value_type=*]:
    ctypedef pair[simplex_type, F*] simplex_filtration_type
    Simplex_tree_multi_interface() nogil
    Simplex_tree_multi_interface(Simplex_tree_multi_interface&) nogil
    F* simplex_filtration(const vector[int]& simplex) nogil
    void assign_simplex_filtration(vector[int]& simplex, const F& filtration) noexcept nogil
    void initialize_filtration() nogil
    int num_vertices() nogil
    int num_simplices() nogil
    void set_dimension(int dimension) nogil
    dimension_type dimension() nogil
    dimension_type upper_bound_dimension() nogil
    bool find_simplex(vector[int]& simplex) nogil
    bool insert(vector[int]& simplex, F& filtration) noexcept nogil
    # vector[simplex_filtration_type] get_star(const vector[int]& simplex) nogil
    # vector[simplex_filtration_type] get_cofaces(const vector[int]& simplex, int dimension) nogil
    void expansion(int max_dim)  except + nogil
    void remove_maximal_simplex(simplex_type simplex) nogil
    # bool prune_above_filtration(filtration_type filtration) nogil
    bool prune_above_dimension(int dimension) nogil
    bool make_filtration_non_decreasing()  except + nogil
    # void compute_extended_filtration() nogil
    # Simplex_tree_multi_interface* collapse_edges(int nb_collapse_iteration)  except + nogil
    void reset_filtration(const F& filtration, int dimension) nogil
    bint operator==(Simplex_tree_multi_interface) nogil
    # Iterators over Simplex tree
    pair[simplex_type,F*] get_simplex_and_filtration(Simplex_tree_multi_simplex_handle f_simplex) nogil
    Simplex_tree_multi_simplices_iterator[F] get_simplices_iterator_begin() nogil
    Simplex_tree_multi_simplices_iterator[F] get_simplices_iterator_end() nogil
    vector[Simplex_tree_multi_simplex_handle[F]].const_iterator get_filtration_iterator_begin() nogil
    vector[Simplex_tree_multi_simplex_handle[F]].const_iterator get_filtration_iterator_end() nogil
    Simplex_tree_multi_skeleton_iterator get_skeleton_iterator_begin(int dimension) nogil
    Simplex_tree_multi_skeleton_iterator get_skeleton_iterator_end(int dimension) nogil
    pair[Simplex_tree_multi_boundary_iterator, Simplex_tree_multi_boundary_iterator] get_boundary_iterators(vector[int] simplex)  except + nogil
    # Expansion with blockers
    ctypedef bool (*blocker_func_t)(vector[int], void *user_data)
    void expansion_with_blockers_callback(int dimension, blocker_func_t user_func, void *user_data)

    ## MULTIPERS STUFF
    void set_keys_to_enumerate() nogil const
    int get_key(const simplex_type) nogil
    void set_key(simplex_type, int) nogil
    void fill_lowerstar(const F&, int) nogil
    simplex_list get_simplices_of_dimension(int) nogil
    edge_list get_edge_list() nogil
    # euler_char_list euler_char(const vector[filtration_type]&) nogil
    void resize_all_filtrations(int) nogil
    void set_number_of_parameters(int) nogil
    int get_number_of_parameters() nogil
  
    
    void from_std(intptr_t,int, F&) nogil
    void to_std(intptr_t, Line[double],int ) nogil 
    void to_std_linear_projection(intptr_t, vector[double]) nogil
    void squeeze_filtration_inplace(vector[vector[double]] &, bool) nogil
    void squeeze_filtration(intptr_t, vector[vector[double]] &) nogil
    vector[vector[vector[value_type]]] get_filtration_values(const vector[int]&) nogil


    pair[boundary_matrix, vector[Finitely_critical_multi_filtration[value_type]]] simplextree_to_boundary_filtration()
    vector[pair[ vector[vector[value_type]],boundary_matrix]] simplextree_to_scc()
    vector[pair[ vector[vector[vector[value_type]]],boundary_matrix]] kcritical_simplextree_to_scc()

    vector[pair[ vector[vector[vector[value_type]]],boundary_matrix]] function_simplextree_to_scc()
    pair[vector[vector[value_type]],boundary_matrix ] simplextree_to_ordered_bf()
    # vector[map[value_type,int32_t]] build_idx_map(const vector[int]&) nogil
    # pair[vector[vector[int32_t]],vector[vector[int32_t]]] get_pts_indices(const vector[map[value_type,int32_t]]&, const vector[vector[value_type]]&) nogil
    pair[vector[vector[int32_t]],vector[vector[int32_t]]] pts_to_indices(vector[vector[value_type]]&, vector[int]&) nogil

