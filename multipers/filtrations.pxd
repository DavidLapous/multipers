from libcpp.utility cimport pair
from libcpp cimport bool
from libcpp.vector cimport vector
from libcpp cimport tuple
from libc.stdint cimport uintptr_t,intptr_t
from cpython cimport Py_buffer


cdef extern from "gudhi/Simplex_tree/multi_filtrations/Finitely_critical_filtrations.h" namespace "Gudhi::multiparameter::multi_filtrations":
    cdef cppclass Finitely_critical_multi_filtration[T=*]:
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
        size_type num_parameters()
        size_type num_generators()
        void swap(vector&)

        # C++11 methods
        value_type* data()
        const value_type* const_data "data"()
        void shrink_to_fit() except +
        iterator emplace(const_iterator, ...) except +
        value_type& emplace_back(...) except +

        ## end of copied from cython vector

        Finitely_critical_multi_filtration()  except + nogil
        Finitely_critical_multi_filtration(vector[value_type]&) except + nogil
        Finitely_critical_multi_filtration(Finitely_critical_multi_filtration&) except + nogil

        Finitely_critical_multi_filtration(int) nogil
        Finitely_critical_multi_filtration& operator=(const Finitely_critical_multi_filtration&) except +
        @staticmethod
        vector[vector[value_type]] to_python(vector[Finitely_critical_multi_filtration]&) nogil const 
        @staticmethod
        vector[value_type]& vector[value_type]() nogil
        # overloading += not yet supported.
        # Finitely_critical_multi_filtration[T]& operator+=(Finitely_critical_multi_filtration[T]&, const Finitely_critical_multi_filtration[T]&)
        #
        # Finitely_critical_multi_filtration[T]& operator-=(Finitely_critical_multi_filtration[T]&, const Finitely_critical_multi_filtration[T]&)
        # Finitely_critical_multi_filtration[T]& operator*=(Finitely_critical_multi_filtration[T]&, const Finitely_critical_multi_filtration[T]&)
        
        void push_to(Finitely_critical_multi_filtration[T]&) nogil
        void pull_to(Finitely_critical_multi_filtration[T]&) nogil

        

    cdef cppclass KCriticalFiltration[T=*]:
        ctypedef size_t size_type
        ctypedef Finitely_critical_multi_filtration[T] filtration_type
        KCriticalFiltration()  except + nogil
        KCriticalFiltration(Finitely_critical_multi_filtration[T]) except +
        KCriticalFiltration[T]& operator=(const KCriticalFiltration[T]&) except +
        size_type num_parameters()
        size_type num_generators()
        void clear() nogil
        void push_back(T) nogil
        void add_point(Finitely_critical_multi_filtration[T]) nogil
        void reserve(size_t) nogil
        void set_num_generators(size_t) nogil
        Finitely_critical_multi_filtration[T]& operator[](int) nogil
        # @staticmethod
        # multifiltration& to_python(vector[KCriticalFiltration]&) nogil const 
        # @staticmethod
        # vector[KCriticalFiltration]& from_python(multifiltration&) nogil const 
        # vector[value_type]& _convert_back() nogil
        # filtration_type __filtration_type__(self):
        #     return self.get_vector()
        # KCriticalFiltration[T]& operator+=(KCriticalFiltration[T]&, const KCriticalFiltration[T]&)
        #
        # KCriticalFiltration[T]& operator-=(KCriticalFiltration[T]&, const KCriticalFiltration[T]&)
        # KCriticalFiltration[T]& operator*=(KCriticalFiltration[T]&, const KCriticalFiltration[T]&)

        void push_to(Finitely_critical_multi_filtration[T]&) except + nogil
        void pull_to(Finitely_critical_multi_filtration[T]&) except + nogil

cdef extern from "gudhi/Simplex_tree/multi_filtrations/Box.h" namespace "Gudhi::multiparameter::multi_filtrations":
    cdef cppclass Box[T=*]:
        ctypedef vector[T] corner_type
        Box()   except +
        Box( vector[T]&,  vector[T]&) nogil 
        Box( pair[vector[T], vector[T]]&) nogil  
        void inflate(T)  nogil 
        const Finitely_critical_multi_filtration[T]& get_bottom_corner()  nogil 
        const Finitely_critical_multi_filtration[T]& get_upper_corner()  nogil 
        bool contains(vector[T]&)  nogil
        pair[Finitely_critical_multi_filtration[T], Finitely_critical_multi_filtration[T]] get_pair() nogil

cdef extern from "gudhi/Simplex_tree/multi_filtrations/Line.h" namespace "Gudhi::multiparameter::multi_filtrations":
    cdef cppclass Line[T=*]:
        ctypedef Finitely_critical_multi_filtration[T] point_type
        Line()   except + nogil
        Line(Finitely_critical_multi_filtration[T]&)   except + nogil
        Line(Finitely_critical_multi_filtration[T]&, Finitely_critical_multi_filtration[T]&)   except + nogil





# ------ useful types:
# ctypedef Finitely_critical_multi_filtration[float] onecritical
# ctypedef KCriticalFiltration[float] kcritical
