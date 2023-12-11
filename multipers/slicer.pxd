from libcpp.utility cimport pair 
from libcpp cimport bool, int, float
from libcpp.vector cimport vector
from libcpp cimport tuple

ctypedef float value_type
ctypedef vector[vector[pair[value_type, value_type]]] Barcode 
ctypedef vector[value_type] point_type
ctypedef vector[value_type] one_filtration_type
from libc.stdint cimport intptr_t, uint16_t, uint32_t, int32_t
from cython cimport uint

import numpy as np
python_value_type=np.float32

from libcpp.string cimport string

cdef extern from "Simplex_tree_multi_interface.h" namespace "Gudhi::multiparameter":
	cdef cppclass Simplex_tree_multi_interface "Gudhi::multiparameter::Simplex_tree_multi_interface<Gudhi::multiparameter::Simplex_tree_options_multidimensional_filtration>":
		pass

cdef extern from "gudhi/Simplex_tree/multi_filtrations/Line.h" namespace "Gudhi::multiparameter::multi_filtrations":
	cdef cppclass Line "Gudhi::multiparameter::multi_filtrations::Line<float>":
		Line()   except + nogil
		Line(point_type&)   except + nogil
		Line(point_type&, point_type&)   except + nogil

from multipers.mma_structures cimport Finitely_critical_multi_filtration

cdef extern from "Persistence_slices_interface.h":
	cdef cppclass SimplicialNoVineMatrixTruc "SimplicialNoVineMatrixTruc":
		SimplicialNoVineMatrixTruc()
		SimplicialNoVineMatrixTruc(Simplex_tree_multi_interface*)
		SimplicialNoVineMatrixTruc& operator=(const SimplicialNoVineMatrixTruc&)
		Barcode get_barcode()
		void push_to[Line](const Line&) 
		void set_one_filtration(const one_filtration_type&)
		one_filtration_type get_one_filtration() const 
		void compute_persistence() 
		uint32_t num_generators()
		string to_str()

	cdef cppclass SimplicialVineMatrixTruc "SimplicialVineMatrixTruc<>":
		SimplicialVineMatrixTruc()
		SimplicialVineMatrixTruc(Simplex_tree_multi_interface*)
		SimplicialVineMatrixTruc& operator=(const SimplicialVineMatrixTruc&)
		void vineyard_update()
		Barcode get_barcode()
		void push_to[Line](const Line&) 
		void set_one_filtration(const one_filtration_type&)
		one_filtration_type get_one_filtration()
		void compute_persistence() 
		uint32_t num_generators()
		string to_str()

	cdef cppclass GeneralVineTruc "GeneralVineTruc<>":
		GeneralVineTruc()
		GeneralVineTruc(const vector[vector[unsigned int]]&, const vector[int]&, const vector[Finitely_critical_multi_filtration]&)
		GeneralVineTruc& operator=(const GeneralVineTruc&)
		void vineyard_update()
		Barcode get_barcode()
		void push_to[Line](const Line&) 
		void set_one_filtration(const one_filtration_type&)
		one_filtration_type get_one_filtration()
		void compute_persistence() 
		uint32_t num_generators()
		string to_str()

	cdef cppclass SimplicialVineGraphTruc "SimplicialVineGraphTruc":
		SimplicialVineGraphTruc()
		SimplicialVineGraphTruc(Simplex_tree_multi_interface*)
		SimplicialVineGraphTruc& operator=(const SimplicialVineGraphTruc&)
		void vineyard_update()
		Barcode get_barcode()
		void push_to[Line](const Line&) 
		void set_one_filtration(const one_filtration_type&)
		one_filtration_type get_one_filtration()
		void compute_persistence() 
		uint32_t num_generators()
		string to_str()

