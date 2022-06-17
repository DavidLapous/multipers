from libcpp.utility cimport pair
from libcpp cimport bool
from libcpp.vector cimport vector

ctypedef pair[vector[double],vector[double]] interval
ctypedef vector[double] corner_type
ctypedef pair[vector[corner_type], vector[corner_type]] corner_list
ctypedef vector[vector[double]] image_type
ctypedef int dimension_type

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
		int get_dimension()
		void set_dimension(int)

cdef extern from "approximation.h" namespace "Vineyard":
	cdef cppclass Module:
		Module() except +
		void resize(unsigned int)
		Summand& at(unsigned int)
		vector[Summand].iterator begin()
		vector[Summand].iterator end()
		void clean(const bool)
		void fill(const double)
		vector[image_type] get_vectorization(const double,unsigned int,unsigned int,const Box&)
		image_type get_vectorization_in_dimension(const int,const double,unsigned int,unsigned int,const Box&)
		void add_summand(Summand)
		unsigned int size()
		Box get_box()
		void set_box(Box &box)
		unsigned int get_dimension()
		vector[corner_list] get_corners_of_dimension(unsigned int)
		image_type get_vectorization_in_dimension(const dimension_type, double, unsigned int, unsigned int)
		vector[image_type] get_vectorization(double, unsigned int, unsigned int)

