# This file is part of the Gudhi Library - https://gudhi.inria.fr/ - which is released under MIT.
# See file LICENSE or go to https://gudhi.inria.fr/licensing/ for full license details.
# Author(s):       Vincent Rouvreau
#
# Copyright (C) 2016 Inria
#
# Modification(s):
#   - YYYY/MM Author: Description of the modification

from cython cimport numeric
from libcpp.vector cimport vector
from libcpp.utility cimport pair
from libcpp cimport bool
from libcpp.string cimport string

__author__ = "Vincent Rouvreau"
__copyright__ = "Copyright (C) 2016 Inria"
__license__ = "MIT"

ctypedef int dimension_type
ctypedef vector[double] point_type
ctypedef double filtration_value_type
ctypedef vector[double] filtration_type
ctypedef vector[int] simplex_type


cdef extern from "Simplex_tree_interface.h" namespace "Gudhi":
	cdef cppclass Simplex_tree_options_multidimensional_filtration:
		pass

	cdef cppclass Simplex_tree_multi_simplex_handle "Gudhi::Simplex_tree_interface<Gudhi::Simplex_tree_options_multidimensional_filtration>::Simplex_handle":
		pass

	cdef cppclass Simplex_tree_multi_simplices_iterator "Gudhi::Simplex_tree_interface<Gudhi::Simplex_tree_options_multidimensional_filtration>::Complex_simplex_iterator":
		Simplex_tree_multi_simplices_iterator() nogil
		Simplex_tree_multi_simplex_handle& operator*() nogil
		Simplex_tree_multi_simplices_iterator operator++() nogil
		bint operator!=(Simplex_tree_multi_simplices_iterator) nogil

	cdef cppclass Simplex_tree_multi_skeleton_iterator "Gudhi::Simplex_tree_interface<Gudhi::Simplex_tree_options_multidimensional_filtration>::Skeleton_simplex_iterator":
		Simplex_tree_multi_skeleton_iterator() nogil
		Simplex_tree_multi_simplex_handle& operator*() nogil
		Simplex_tree_multi_skeleton_iterator operator++() nogil
		bint operator!=(Simplex_tree_multi_skeleton_iterator) nogil

	cdef cppclass Simplex_tree_multi_boundary_iterator "Gudhi::Simplex_tree_interface<Gudhi::Simplex_tree_options_multidimensional_filtration>::Boundary_simplex_iterator":
		Simplex_tree_multi_boundary_iterator() nogil
		Simplex_tree_multi_simplex_handle& operator*() nogil
		Simplex_tree_multi_boundary_iterator operator++() nogil
		bint operator!=(Simplex_tree_multi_boundary_iterator) nogil


	cdef cppclass Simplex_tree_multi_interface "Gudhi::Simplex_tree_interface<Gudhi::Simplex_tree_options_multidimensional_filtration>":
		Simplex_tree_multi_interface() nogil
		Simplex_tree_multi_interface(Simplex_tree_multi_interface&) nogil
		filtration_type simplex_filtration(vector[int] simplex) nogil
		void assign_simplex_filtration(vector[int] simplex, filtration_type filtration) nogil
		void initialize_filtration() nogil
		int num_vertices() nogil
		int num_simplices() nogil
		void set_dimension(int dimension) nogil
		dimension_type dimension() nogil
		dimension_type upper_bound_dimension() nogil
		bool find_simplex(vector[int] simplex) nogil
		bool insert(vector[int] simplex, filtration_type filtration) nogil
		vector[pair[simplex_type, filtration_type]] get_star(vector[int] simplex) nogil
		vector[pair[simplex_type, filtration_type]] get_cofaces(vector[int] simplex, int dimension) nogil
		void expansion(int max_dim) nogil except +
		void remove_maximal_simplex(simplex_type simplex) nogil
		bool prune_above_filtration(filtration_type filtration) nogil
		# bool make_filtration_non_decreasing() nogil
		# void compute_extended_filtration() nogil
		Simplex_tree_multi_interface* collapse_edges(int nb_collapse_iteration) nogil except +
		void reset_filtration(filtration_type filtration, int dimension) nogil
		bint operator==(Simplex_tree_multi_interface) nogil
		# Iterators over Simplex tree
		pair[simplex_type, filtration_type] get_simplex_and_filtration(Simplex_tree_multi_simplex_handle f_simplex) nogil
		Simplex_tree_multi_simplices_iterator get_simplices_iterator_begin() nogil
		Simplex_tree_multi_simplices_iterator get_simplices_iterator_end() nogil
		vector[Simplex_tree_multi_simplex_handle].const_iterator get_filtration_iterator_begin() nogil
		vector[Simplex_tree_multi_simplex_handle].const_iterator get_filtration_iterator_end() nogil
		Simplex_tree_multi_skeleton_iterator get_skeleton_iterator_begin(int dimension) nogil
		Simplex_tree_multi_skeleton_iterator get_skeleton_iterator_end(int dimension) nogil
		pair[Simplex_tree_multi_boundary_iterator, Simplex_tree_multi_boundary_iterator] get_boundary_iterators(vector[int] simplex) nogil except +
		# Expansion with blockers
		ctypedef bool (*blocker_func_t)(vector[int], void *user_data)
		void expansion_with_blockers_callback(int dimension, blocker_func_t user_func, void *user_data)

