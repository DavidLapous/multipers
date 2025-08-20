from libcpp.utility cimport pair
from libcpp cimport bool
from libcpp.vector cimport vector
from libcpp cimport tuple
from libc.stdint cimport uintptr_t,intptr_t
from cpython cimport Py_buffer


cdef extern from "gudhi/Persistence_slices_interface.h" namespace "multipers::tmp_interface":
    cdef cppclass One_critical_filtration[T]:
        ## Copied from cython vector
        # ctypedef size_t size_type
        # ctypedef ptrdiff_t difference_type
        # ctypedef T value_type

        One_critical_filtration()  except + nogil
        One_critical_filtration(int)  except + nogil
        One_critical_filtration(int, T)  except + nogil
        One_critical_filtration(vector[T]&)  except + nogil
        One_critical_filtration(vector[T].iterator, vector[T].iterator, int)  except + nogil
        One_critical_filtration[T]& operator=(const One_critical_filtration[T]&) except + nogil
        T& operator()(size_t, size_t) nogil
        size_t num_parameters() nogil
        size_t num_generators() nogil
        size_t num_entries() nogil
        void set_num_generators(size_t) nogil
        @staticmethod
        One_critical_filtration inf(int)
        @staticmethod
        One_critical_filtration minus_inf(int)
        @staticmethod
        One_critical_filtration nan(int)
        bool is_plus_inf() nogil
        bool is_minus_inf() nogil
        bool is_nan() nogil
        bool is_finite() nogil
        bool add_generator(vector[T] &x) nogil
        bool add_generator(vector[T] x) nogil
        void add_guaranteed_generator(vector[T] &x) nogil
        void simplify() nogil
        void remove_empty_generators(bool) nogil
        bool push_to_least_common_upper_bound(vector[T] &, bool) nogil
        bool push_to_least_common_upper_bound(const One_critical_filtration &, bool) nogil
        bool pull_to_greatest_common_lower_bound(vector[T] &, bool) nogil
        # bool pull_to_greatest_common_lower_bound(const One_critical_filtration &, bool) nogil

        # specific to Dynamic_multi_parameter_filtration
        void force_generator_size_to_number_of_parameters(size_t) nogil

cdef extern from "gudhi/Persistence_slices_interface.h" namespace "multipers::tmp_interface":
    cdef cppclass Multi_critical_filtration[T]:
        ## Copied from cython vector
        # ctypedef size_t size_type
        # ctypedef ptrdiff_t difference_type
        # ctypedef T value_type

        Multi_critical_filtration()  except + nogil
        Multi_critical_filtration(int)  except + nogil
        Multi_critical_filtration(int, T)  except + nogil
        Multi_critical_filtration(vector[T]&)  except + nogil
        Multi_critical_filtration(vector[T].iterator, vector[T].iterator, int)  except + nogil
        Multi_critical_filtration[T]& operator=(const Multi_critical_filtration[T]&) except + nogil
        T& operator()(size_t, size_t) nogil
        size_t num_parameters() nogil
        size_t num_generators() nogil
        size_t num_entries() nogil
        void set_num_generators(size_t) nogil
        @staticmethod
        Multi_critical_filtration inf(int)
        @staticmethod
        Multi_critical_filtration minus_inf(int)
        @staticmethod
        Multi_critical_filtration nan(int)
        bool is_plus_inf() nogil
        bool is_minus_inf() nogil
        bool is_nan() nogil
        bool is_finite() nogil
        bool add_generator(vector[T] &x) nogil
        bool add_generator(vector[T] x) nogil
        void add_guaranteed_generator(vector[T] &x) nogil
        void simplify() nogil
        void remove_empty_generators(bool) nogil
        # "push_to_least_common_upper_bound<std::vector<T>>"
        bool push_to_least_common_upper_bound(vector[T] &, bool) nogil
        bool push_to_least_common_upper_bound(const Multi_critical_filtration[T] &, bool) nogil
        bool pull_to_greatest_common_lower_bound(vector[T] &, bool) nogil
        # bool pull_to_greatest_common_lower_bound(const Multi_critical_filtration[T] &, bool) nogil

        # specific to Dynamic_multi_parameter_filtration
        void force_generator_size_to_number_of_parameters(size_t) nogil


cdef extern from "gudhi/Multi_persistence/Point.h" namespace "Gudhi::multi_persistence":
    cdef cppclass Point[T]:
        ctypedef size_t size_type
        ctypedef ptrdiff_t difference_type
        ctypedef T value_type
        ctypedef T& reference
        ctypedef T* pointer

        cppclass const_iterator
        cppclass iterator:
            iterator() except +
            iterator(iterator&) except +
            value_type& operator*()
            iterator operator++()
            iterator operator--()
            iterator operator++(int)
            iterator operator--(int)
            iterator operator+(size_type)
            iterator operator-(size_type)
            difference_type operator-(iterator)
            difference_type operator-(const_iterator)
            bint operator==(iterator)
            bint operator==(const_iterator)
            bint operator!=(iterator)
            bint operator!=(const_iterator)
            bint operator<(iterator)
            bint operator<(const_iterator)
            bint operator>(iterator)
            bint operator>(const_iterator)
            bint operator<=(iterator)
            bint operator<=(const_iterator)
            bint operator>=(iterator)
            bint operator>=(const_iterator)
        cppclass const_iterator:
            const_iterator() except +
            const_iterator(iterator&) except +
            const_iterator(const_iterator&) except +
            operator=(iterator&) except +
            const value_type& operator*()
            const_iterator operator++()
            const_iterator operator--()
            const_iterator operator++(int)
            const_iterator operator--(int)
            const_iterator operator+(size_type)
            const_iterator operator-(size_type)
            difference_type operator-(iterator)
            difference_type operator-(const_iterator)
            bint operator==(iterator)
            bint operator==(const_iterator)
            bint operator!=(iterator)
            bint operator!=(const_iterator)
            bint operator<(iterator)
            bint operator<(const_iterator)
            bint operator>(iterator)
            bint operator>(const_iterator)
            bint operator<=(iterator)
            bint operator<=(const_iterator)
            bint operator>=(iterator)
            bint operator>=(const_iterator)

        cppclass const_reverse_iterator
        cppclass reverse_iterator:
            reverse_iterator() except +
            reverse_iterator(reverse_iterator&) except +
            value_type& operator*()
            reverse_iterator operator++()
            reverse_iterator operator--()
            reverse_iterator operator++(int)
            reverse_iterator operator--(int)
            reverse_iterator operator+(size_type)
            reverse_iterator operator-(size_type)
            difference_type operator-(iterator)
            difference_type operator-(const_iterator)
            bint operator==(reverse_iterator)
            bint operator==(const_reverse_iterator)
            bint operator!=(reverse_iterator)
            bint operator!=(const_reverse_iterator)
            bint operator<(reverse_iterator)
            bint operator<(const_reverse_iterator)
            bint operator>(reverse_iterator)
            bint operator>(const_reverse_iterator)
            bint operator<=(reverse_iterator)
            bint operator<=(const_reverse_iterator)
            bint operator>=(reverse_iterator)
            bint operator>=(const_reverse_iterator)
        cppclass const_reverse_iterator:
            const_reverse_iterator() except +
            const_reverse_iterator(reverse_iterator&) except +
            operator=(reverse_iterator&) except +
            const value_type& operator*()
            const_reverse_iterator operator++()
            const_reverse_iterator operator--()
            const_reverse_iterator operator++(int)
            const_reverse_iterator operator--(int)
            const_reverse_iterator operator+(size_type)
            const_reverse_iterator operator-(size_type)
            difference_type operator-(iterator)
            difference_type operator-(const_iterator)
            bint operator==(reverse_iterator)
            bint operator==(const_reverse_iterator)
            bint operator!=(reverse_iterator)
            bint operator!=(const_reverse_iterator)
            bint operator<(reverse_iterator)
            bint operator<(const_reverse_iterator)
            bint operator>(reverse_iterator)
            bint operator>(const_reverse_iterator)
            bint operator<=(reverse_iterator)
            bint operator<=(const_reverse_iterator)
            bint operator>=(reverse_iterator)
            bint operator>=(const_reverse_iterator)

        Point() except + nogil
        Point(size_type) except + nogil
        Point(size_type, const T &) except + nogil
        Point(const vector[T]&) except + nogil
        Point& operator=(Point&) except + nogil
        reference at(size_type) except +
        reference operator[](size_type)
        reference front()
        reference back()
        pointer data()
        const value_type* const_data "data"()
        iterator begin()
        const_iterator const_begin "begin"()
        const_iterator cbegin()
        iterator end()
        const_iterator const_end "end"()
        const_iterator cend()
        reverse_iterator rbegin()
        const_reverse_iterator const_rbegin "rbegin"()
        const_reverse_iterator crbegin()
        reverse_iterator rend()
        const_reverse_iterator const_rend "rend"()
        const_reverse_iterator crend()
        size_type size()

    bint operator<(const Point &, const Point &)
    bint operator<=(const Point &, const Point &)
    bint operator>(const Point &, const Point &)
    bint operator>=(const Point &, const Point &)
    bint operator==(const Point &, const Point &)
    bint operator!=(const Point &, const Point &)
    Point operator-(const Point &)
    Point operator-(Point, const Point &)
    Point operator-(Point, double)
    Point operator-(Point, float)
    Point operator-(Point, int)
    Point operator-(double, Point)
    Point operator-(float, Point)
    Point operator-(int, Point)
    Point operator+(Point, const Point &)
    Point operator+(Point, double)
    Point operator+(Point, float)
    Point operator+(Point, int)
    Point operator+(double, Point)
    Point operator+(float, Point)
    Point operator+(int, Point)
    Point operator*(Point, const Point &)
    Point operator*(Point, double)
    Point operator*(Point, float)
    Point operator*(Point, int)
    Point operator*(double, Point)
    Point operator*(float, Point)
    Point operator*(int, Point)
    Point operator/(Point, const Point &)
    Point operator/(Point, double)
    Point operator/(Point, float)
    Point operator/(Point, int)
    Point operator/(double, Point)
    Point operator/(float, Point)
    Point operator/(int, Point)

cdef extern from "gudhi/Multi_persistence/Box.h" namespace "Gudhi::multi_persistence":
    cdef cppclass Box[T=*]:
        ctypedef Point[T] corner_type
        Box()   except +
        # Box(corner_type&, corner_type&) nogil 
        # Box(pair[Point[T], Point[T]]&) nogil 
        Box(vector[T]&, vector[T]&) nogil 
        Box(pair[vector[T], vector[T]]&) nogil  
        void inflate(T)  nogil 
        # const corner_type& get_lower_corner()  nogil 
        # const corner_type& get_upper_corner()  nogil 
        const vector[T]& get_lower_corner()  nogil 
        const vector[T]& get_upper_corner()  nogil 
        # bool contains(corner_type&)  nogil
        bool contains(vector[T]&)  nogil
        # pair[corner_type, corner_type] get_bounding_corners() nogil
        pair[vector[T], vector[T]] get_bounding_corners() nogil

cdef extern from "gudhi/Multi_persistence/Line.h" namespace "Gudhi::multi_persistence":
    cdef cppclass Line[T=*]:
        ctypedef Point[T] point_type
        Line()   except + nogil
        # Line(point_type&)   except + nogil
        # Line(point_type&, point_type&)   except + nogil
        Line(vector[T]&)   except + nogil
        Line(vector[T]&, vector[T]&)   except + nogil





# ------ useful types:
# ctypedef One_critical_filtration[float] Generator
# ctypedef Multi_critical_filtration[float] kcritical
