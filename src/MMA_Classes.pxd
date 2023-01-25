from libcpp.utility cimport pair
from libcpp cimport bool
from libcpp.vector cimport vector
from libcpp cimport tuple

ctypedef pair[vector[double],vector[double]] interval
ctypedef vector[double] corner_type
ctypedef pair[vector[corner_type], vector[corner_type]] corner_list
ctypedef vector[vector[double]] image_type
ctypedef int dimension_type
ctypedef vector[double] point_type
ctypedef double filtration_value_type
ctypedef pair[corner_type, corner_type] bar
ctypedef vector[bar] barcode
ctypedef pair[vector[pair[double, double]], vector[unsigned int]] plot_interface_type
ctypedef vector[double] multipers_bar
ctypedef vector[multipers_bar] multipers_barcode
ctypedef vector[barcode] barcodes
ctypedef vector[int] euler_curve_type


cdef extern from "mma_cpp/vineyards_trajectories.h" namespace "Vineyard":
	cdef cppclass Box:
		Box()   except +
		Box(const corner_type&, const corner_type&) nogil except +
		Box(const pair[corner_type, corner_type]&) nogil  except +
		void inflate(double)  nogil
		const corner_type& get_bottom_corner()  nogil
		const corner_type& get_upper_corner()  nogil
		bool contains(corner_type&)  nogil


cdef extern from "mma_cpp/approximation.h" namespace "Vineyard":
	cdef cppclass Summand:
		Summand() except +
		Summand(vector[corner_type]&, vector[corner_type]&, int) nogil except +
		double get_interleaving() nogil
		double get_local_weight(const corner_type&, const double)  nogil
		void add_bar(double, double, const corner_type&, corner_type&, corner_type&, const bool, const interval&) nogil
		bool is_empty() nogil
		const vector[corner_type]& get_birth_list() nogil
		const vector[corner_type]& get_death_list() nogil
		void complete_birth(const double)  nogil
		void complete_death(const double)  nogil
		dimension_type get_dimension() nogil
		void set_dimension(int) nogil
		bool contains(const corner_type&)  nogil



cdef extern from "mma_cpp/line_filtration_translation.h" namespace "Vineyard":
	cdef cppclass Line:
		Line()   except +
		Line(point_type&)   except +
		Line(point_type&, point_type&)   except +

cdef extern from "mma_cpp/utilities.h" namespace "Vineyard":
	cdef cppclass MultiDiagram_point:
		MultiDiagram_point()  nogil except +
		MultiDiagram_point(dimension_type , corner_type , corner_type ) nogil  except +
		corner_type get_birth()  nogil const
		corner_type get_death()  nogil const
		dimension_type get_dimension()  nogil const

cdef extern from "mma_cpp/utilities.h" namespace "Vineyard":
	cdef cppclass MultiDiagram:
		MultiDiagram()  nogil except +
		barcode get_points(const dimension_type) const  
		multipers_barcode to_multipers(const dimension_type) const  
		vector[MultiDiagram_point].const_iterator begin()  
		vector[MultiDiagram_point].const_iterator end()  
		unsigned int size() const  
		MultiDiagram_point& at(unsigned int)  nogil

cdef extern from "mma_cpp/utilities.h" namespace "Vineyard":
	cdef cppclass MultiDiagrams:
		MultiDiagrams() nogil except +
		vector[vector[vector[double]]] to_multipers() nogil const   
		MultiDiagram& at(const unsigned int)  nogil
		unsigned int size() nogil const 
		vector[MultiDiagram].const_iterator begin()
		vector[MultiDiagram].const_iterator end()
		plot_interface_type _for_python_plot(dimension_type, double)  nogil
		barcodes get_points()  nogil

cdef extern from "mma_cpp/approximation.h" namespace "Vineyard":
	cdef cppclass Module:
		Module() nogil except +
		void resize(unsigned int)  nogil
		Summand& at(unsigned int)  nogil
		vector[Summand].iterator begin()
		vector[Summand].iterator end() 
		void clean(const bool)  nogil
		void fill(const double)  nogil
		# vector[image_type] get_vectorization(const double,const double, unsigned int,unsigned int,const Box&)
		# image_type get_vectorization_in_dimension(const int,const double,unsigned int,unsigned int,const Box&)
		void add_summand(Summand)  nogil
		unsigned int size() const  
		Box get_box() const  
		void set_box(Box &box)  nogil
		int get_dimension() const 
		vector[corner_list] get_corners_of_dimension(unsigned int)  nogil
		image_type get_vectorization_in_dimension(const dimension_type, const double, const double, const bool, Box&, unsigned int, unsigned int)  nogil
		vector[image_type] get_vectorization(const double, const double, const bool, Box, unsigned int, unsigned int)  nogil
		MultiDiagram get_barcode(const Line&, const dimension_type, const bool)  nogil
		MultiDiagrams get_barcodes(const vector[point_type]& , const dimension_type, const bool )  nogil
		image_type get_landscape(const dimension_type,const unsigned int,const Box&,const vector[unsigned int]&)  nogil
		vector[image_type] get_landscapes(const dimension_type,const vector[unsigned int],const Box&,const vector[unsigned int]&)  nogil
		euler_curve_type euler_curve(const vector[corner_type]&)  nogil
