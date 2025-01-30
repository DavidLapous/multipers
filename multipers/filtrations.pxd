from libcpp.utility cimport pair
from libcpp cimport bool
from libcpp.vector cimport vector
from libcpp cimport tuple
from libc.stdint cimport uintptr_t,intptr_t
from cpython cimport Py_buffer


cdef extern from "gudhi/One_critical_filtration.h" namespace "Gudhi::multi_filtration":
    cdef cppclass One_critical_filtration[T=*]:
        ## Copied from cython vector
        ctypedef size_t size_type
        ctypedef ptrdiff_t difference_type
        ctypedef T value_type

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
        value_type& operator[](size_type)
        #vector& operator=(vector&)
        void assign(size_type, const value_type&)
        void assign[InputIt](InputIt, InputIt) except +
        value_type& at(size_type) except +
        value_type& back()
        iterator begin()
        const_iterator const_begin "begin"()
        const_iterator cbegin()
        size_type capacity()
        void clear() nogil
        bint empty() nogil
        iterator end()
        const_iterator const_end "end"()
        const_iterator cend()
        iterator erase(iterator)
        iterator erase(iterator, iterator)
        value_type& front()
        iterator insert(iterator, const value_type&) except +
        iterator insert(iterator, size_type, const value_type&) except +
        iterator insert[InputIt](iterator, InputIt, InputIt) except +
        size_type max_size()
        void pop_back()
        void push_back(value_type&) except + nogil
        reverse_iterator rbegin()
        const_reverse_iterator const_rbegin "rbegin"()
        const_reverse_iterator crbegin()
        reverse_iterator rend()
        const_reverse_iterator const_rend "rend"()
        const_reverse_iterator crend()
        void reserve(size_type) except + nogil
        void resize(size_type) except + nogil
        void resize(size_type, value_type&) except +
        # size_type size()
        size_type num_parameters() nogil
        size_type num_generators() nogil
        void swap(vector&)

        # C++11 methods
        value_type* data()
        const value_type* const_data "data"()
        void shrink_to_fit() except +
        iterator emplace(const_iterator, ...) except +
        value_type& emplace_back(...) except +

        ## end of copied from cython vector

        One_critical_filtration()  except + nogil
        One_critical_filtration(vector[value_type]&) except + nogil
        One_critical_filtration(One_critical_filtration&) except + nogil

        One_critical_filtration(int) nogil
        One_critical_filtration& operator=(const One_critical_filtration&) except +
        @staticmethod
        vector[value_type]& vector[value_type]() nogil
        
        void push_to_least_common_upper_bound(One_critical_filtration[T]&) nogil
        void pull_to_greatest_common_lower_bound(One_critical_filtration[T]&) nogil

        bool is_finite() nogil

        
cdef extern from "gudhi/Multi_critical_filtration.h" namespace "Gudhi::multi_filtration":
    cdef cppclass Multi_critical_filtration[T=*]:
        ctypedef size_t size_type
        ctypedef One_critical_filtration[T] filtration_type
        Multi_critical_filtration()  except + nogil
        Multi_critical_filtration(One_critical_filtration[T]) except +
        Multi_critical_filtration[T]& operator=(const Multi_critical_filtration[T]&) except +
        size_t num_parameters() noexcept nogil
        size_t num_generators() noexcept  nogil
        void add_guaranteed_generator(One_critical_filtration[T]) nogil
        void add_generator(One_critical_filtration[T]) nogil
        void reserve(size_t) noexcept nogil
        void simplify() nogil
        void set_num_generators(size_t) nogil
        One_critical_filtration[T]& operator[](int) nogil

        void push_to_least_common_upper_bound(One_critical_filtration[T]&) except + nogil
        void pull_to_greatest_common_lower_bound(One_critical_filtration[T]&) except + nogil

cdef extern from "gudhi/Multi_persistence/Box.h" namespace "Gudhi::multi_persistence":
    cdef cppclass Box[T=*]:
        ctypedef vector[T] corner_type
        Box()   except +
        Box( vector[T]&,  vector[T]&) nogil 
        Box( pair[vector[T], vector[T]]&) nogil  
        void inflate(T)  nogil 
        const One_critical_filtration[T]& get_lower_corner()  nogil 
        const One_critical_filtration[T]& get_upper_corner()  nogil 
        bool contains(vector[T]&)  nogil
        pair[One_critical_filtration[T], One_critical_filtration[T]] get_bounding_corners() nogil

cdef extern from "gudhi/Multi_persistence/Line.h" namespace "Gudhi::multi_persistence":
    cdef cppclass Line[T=*]:
        ctypedef One_critical_filtration[T] point_type
        Line()   except + nogil
        Line(One_critical_filtration[T]&)   except + nogil
        Line(One_critical_filtration[T]&, One_critical_filtration[T]&)   except + nogil





# ------ useful types:
# ctypedef One_critical_filtration[float] Generator
# ctypedef Multi_critical_filtration[float] kcritical
