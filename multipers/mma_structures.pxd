from libcpp.utility cimport pair
from libcpp cimport bool
from libcpp.vector cimport vector
from libcpp cimport tuple
from libc.stdint cimport uintptr_t,intptr_t

ctypedef float value_type

ctypedef pair[vector[value_type],vector[value_type]] interval
ctypedef vector[value_type] corner_type
ctypedef vector[vector[value_type]] image_type
ctypedef int dimension_type
ctypedef vector[value_type] point_type
ctypedef pair[vector[point_type], vector[point_type]] corner_list
ctypedef value_type filtration_value_type
ctypedef pair[corner_type, corner_type] bar
ctypedef vector[bar] barcode
ctypedef pair[vector[pair[value_type, value_type]], vector[unsigned int]] plot_interface_type
ctypedef vector[value_type] multipers_bar
ctypedef vector[multipers_bar] multipers_barcode
ctypedef vector[barcode] barcodes
ctypedef vector[int] euler_curve_type
ctypedef vector[value_type] filtration_type
ctypedef vector[filtration_type] multifiltration
ctypedef vector[pair[int,pair[value_type,value_type]]] barcoded
ctypedef vector[unsigned int] boundary_type
ctypedef vector[boundary_type] boundary_matrix
ctypedef pair[pair[value_type,value_type],pair[value_type,value_type]] interval_2

ctypedef vector[Summand] summand_list_type
ctypedef vector[summand_list_type] approx_summand_type
ctypedef vector[int] simplex_type


cdef extern from "gudhi/Simplex_tree/multi_filtrations/Finitely_critical_filtrations.h" namespace "Gudhi::multiparameter::multi_filtrations":
	cdef cppclass Finitely_critical_multi_filtration "Gudhi::multiparameter::multi_filtrations::Finitely_critical_multi_filtration<Gudhi::multiparameter::Simplex_tree_options_multidimensional_filtration::value_type>":
		Finitely_critical_multi_filtration()  except + nogil
		Finitely_critical_multi_filtration(filtration_type) except +
		Finitely_critical_multi_filtration& operator=(const Finitely_critical_multi_filtration&) except +
		filtration_type get_vector() nogil const 
		int size() nogil
		void clear() nogil
		void push_back(value_type) nogil
		@staticmethod
		multifiltration& to_python(vector[Finitely_critical_multi_filtration]&) nogil const 
		@staticmethod
		vector[Finitely_critical_multi_filtration]& from_python(multifiltration&) nogil const 
		vector[value_type]& _convert_back() nogil
		filtration_type __filtration_type__(self):
			return self.get_vector()

ctypedef Finitely_critical_multi_filtration cfiltration_type
ctypedef vector[cfiltration_type] cmultifiltration_type

cdef extern from "gudhi/Simplex_tree/multi_filtrations/Box.h" namespace "Gudhi::multiparameter::mma":
	cdef cppclass Box[value_type]:
		Box()   except +
		Box(const corner_type&, const corner_type&) nogil 
		Box(const pair[corner_type, corner_type]&) nogil  
		void inflate(value_type)  nogil 
		const Finitely_critical_multi_filtration& get_bottom_corner()  nogil 
		const Finitely_critical_multi_filtration& get_upper_corner()  nogil 
		bool contains(corner_type&)  nogil
		pair[Finitely_critical_multi_filtration, Finitely_critical_multi_filtration] get_pair() nogil

cdef extern from "gudhi/Simplex_tree/multi_filtrations/Line.h" namespace "Gudhi::multiparameter::mma":
	cdef cppclass Line[value_type]:
		Line()   except + nogil
		Line(point_type&)   except + nogil
		Line(point_type&, point_type&)   except + nogil

cdef extern from "multiparameter_module_approximation/approximation.h" namespace "Gudhi::multiparameter::mma":
	cdef cppclass Summand:
		Summand() except +
		Summand(vector[Finitely_critical_multi_filtration]&, vector[Finitely_critical_multi_filtration]&, int)  except + nogil
		value_type get_interleaving() nogil
		value_type get_local_weight(const corner_type&, const value_type)  nogil
		void add_bar(value_type, value_type, const corner_type&, corner_type&, corner_type&, const bool, const interval&) nogil
		bool is_empty() nogil
		cmultifiltration_type& get_birth_list() nogil
		cmultifiltration_type& get_death_list() nogil
		void complete_birth(const value_type)  nogil
		void complete_death(const value_type)  nogil
		dimension_type get_dimension()  nogil const 
		void set_dimension(int) nogil
		bool contains(const corner_type&)  nogil const
		Box[value_type] get_bounds() nogil const
		void rescale(const vector[value_type]&) nogil





cdef extern from "multiparameter_module_approximation/utilities.h" namespace "Gudhi::multiparameter::mma":
	cdef cppclass MultiDiagram_point:
		MultiDiagram_point()   except + nogil
		MultiDiagram_point(dimension_type , corner_type , corner_type )   except + nogil
		filtration_type get_birth()    nogil const
		filtration_type get_death()  nogil const 
		dimension_type get_dimension() nogil  const 

cdef extern from "multiparameter_module_approximation/utilities.h" namespace "Gudhi::multiparameter::mma":
	cdef cppclass MultiDiagram:
		MultiDiagram()   except + nogil
		barcode get_points(const dimension_type) const  
		multipers_barcode to_multipers(const dimension_type) nogil const   
		vector[MultiDiagram_point].const_iterator begin()  
		vector[MultiDiagram_point].const_iterator end()  
		unsigned int size() const  
		MultiDiagram_point& at(unsigned int)  nogil

cdef extern from "multiparameter_module_approximation/utilities.h" namespace "Gudhi::multiparameter::mma":
	cdef cppclass MultiDiagrams:
		MultiDiagrams()  except + nogil
		vector[vector[vector[value_type]]] to_multipers() nogil const    
		MultiDiagram& at(const unsigned int)  nogil
		unsigned int size() nogil const  
		vector[MultiDiagram].const_iterator begin()
		vector[MultiDiagram].const_iterator end()
		plot_interface_type _for_python_plot(dimension_type, value_type)  nogil
		barcodes get_points()  nogil

cdef extern from "multiparameter_module_approximation/approximation.h" namespace "Gudhi::multiparameter::mma":
	cdef cppclass Module:
		Module()  except + nogil
		void resize(unsigned int)  nogil
		Summand& at(unsigned int)  nogil
		vector[Summand].iterator begin()
		vector[Summand].iterator end() 
		void clean(const bool)  nogil
		void fill(const value_type)  nogil
		# vector[image_type] get_vectorization(const value_type,const value_type, unsigned int,unsigned int,const Box&)
		# image_type get_vectorization_in_dimension(const int,const value_type,unsigned int,unsigned int,const Box&)
		void add_summand(Summand)  nogil
		unsigned int size() const  
		Box[value_type] get_box() const  
		Box[value_type] get_bounds() nogil const
		void set_box(Box[value_type])  nogil
		int get_dimension() const 
		vector[corner_list] get_corners_of_dimension(unsigned int)  nogil
		image_type get_vectorization_in_dimension(const dimension_type, const value_type, const value_type, const bool, Box[value_type]&, unsigned int, unsigned int)  nogil
		vector[image_type] get_vectorization(const value_type, const value_type, const bool, Box[value_type], unsigned int, unsigned int)  nogil
		MultiDiagram get_barcode(Line[value_type]&, const dimension_type, const bool)  nogil
		MultiDiagrams get_barcodes(const vector[Finitely_critical_multi_filtration]& , const dimension_type, const bool )  nogil
		image_type get_landscape(const dimension_type,const unsigned int,Box[value_type],const vector[unsigned int]&)  nogil
		vector[image_type] get_landscapes(const dimension_type,const vector[unsigned int],Box[value_type],const vector[unsigned int]&)  nogil
		euler_curve_type euler_curve(const vector[Finitely_critical_multi_filtration]&) nogil
		void rescale(vector[value_type]&, int) nogil
		void translate(vector[value_type]&, int) nogil
		vector[vector[value_type]] compute_pixels(vector[vector[value_type]], vector[int], Box[value_type], value_type, value_type, bool,int) nogil





