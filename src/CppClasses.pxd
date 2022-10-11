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

cdef extern from "vineyards_trajectories.h" namespace "Vineyard":
	cdef cppclass Box:
		Box() except +
		Box(const corner_type&, const corner_type&) except +
		Box(const pair[corner_type, corner_type]&) except +
		void inflate(double)
		const corner_type& get_bottom_corner()
		const corner_type& get_upper_corner()
		bool contains(corner_type&)


cdef extern from "approximation.h" namespace "Vineyard":
	cdef cppclass Summand:
		Summand() except +
		Summand(vector[corner_type]&, vector[corner_type]&, int) except +
		double get_interleaving()
		double get_local_weight(const corner_type&, const double)
		void add_bar(double, double, const corner_type&, corner_type&, corner_type&, const bool, const interval&)
		bool is_empty()
		const vector[corner_type]& get_birth_list()
		const vector[corner_type]& get_death_list()
		void complete_birth(const double)
		void complete_death(const double)
		dimension_type get_dimension()
		void set_dimension(int)



cdef extern from "line_filtration_translation.h" namespace "Vineyard":
	cdef cppclass Line:
		Line() except +
		Line(point_type&) except +
		Line(point_type&, point_type&) except +

cdef extern from "utilities.h" namespace "Vineyard":
	cdef cppclass MultiDiagram_point:
		MultiDiagram_point() except +
		MultiDiagram_point(dimension_type , corner_type , corner_type ) except +
		corner_type get_birth() const
		corner_type get_death() const
		dimension_type get_dimension() const

cdef extern from "utilities.h" namespace "Vineyard":
	cdef cppclass MultiDiagram:
		MultiDiagram() except +
		barcode get_points(const dimension_type) const
		multipers_barcode to_multipers(const dimension_type) const
		vector[MultiDiagram_point].const_iterator begin()
		vector[MultiDiagram_point].const_iterator end()
		unsigned int size() const
		MultiDiagram_point& at(unsigned int)

cdef extern from "utilities.h" namespace "Vineyard":
	cdef cppclass MultiDiagrams:
		MultiDiagrams() except +
		vector[vector[vector[double]]] to_multipers() const
		MultiDiagram& at(const unsigned int)
		unsigned int size() const
		vector[MultiDiagram].const_iterator begin()
		vector[MultiDiagram].const_iterator end()
		plot_interface_type _for_python_plot(dimension_type, double)

cdef extern from "approximation.h" namespace "Vineyard":
	cdef cppclass Module:
		Module() except +
		void resize(unsigned int)
		Summand& at(unsigned int)
		vector[Summand].iterator begin()
		vector[Summand].iterator end()
		void clean(const bool)
		void fill(const double)
		# vector[image_type] get_vectorization(const double,const double, unsigned int,unsigned int,const Box&)
		# image_type get_vectorization_in_dimension(const int,const double,unsigned int,unsigned int,const Box&)
		void add_summand(Summand)
		unsigned int size() const
		Box get_box() const
		void set_box(Box &box)
		dimension_type get_dimension() const
		vector[corner_list] get_corners_of_dimension(unsigned int)
		image_type get_vectorization_in_dimension(const dimension_type, const double, const double, const bool, Box, unsigned int, unsigned int)
		vector[image_type] get_vectorization(const double, const double, const bool, Box, unsigned int, unsigned int)
		MultiDiagram get_barcode(const Line &line, const dimension_type, const bool)
		MultiDiagrams get_barcodes(const vector[point_type] , const dimension_type, const bool )
		image_type get_landscape(const dimension_type,const unsigned int,const Box&,const vector[unsigned int]&)
		vector[image_type] get_landscapes(const dimension_type,const vector[unsigned int],const Box&,const vector[unsigned int]&)
