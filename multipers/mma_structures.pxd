from libcpp.utility cimport pair
from libcpp cimport bool
from libcpp.vector cimport vector
from libc.stdint cimport uintptr_t,intptr_t



from multipers.filtrations cimport *
ctypedef vector[unsigned int] boundary_type
ctypedef vector[boundary_type] boundary_matrix
ctypedef vector[int] simplex_type

cdef extern from "gudhi/Multi_persistence/Summand.h" namespace "Gudhi::multi_persistence":
    cdef cppclass Summand[T=*]:
        ctypedef vector[T] corner_type
        ctypedef T value_type
        ctypedef pair[vector[T],vector[T]] interval
        T T_inf
        T T_m_inf
        Summand() except +
        Summand(int) except +
        Summand(vector[T]&, vector[T]&, int, int)  except + nogil
        int get_number_of_parameters() nogil const
        int get_number_of_birth_corners() nogil const
        int get_number_of_death_corners() nogil const
        vector[T] compute_birth_list() nogil
        vector[T] compute_death_list() nogil
        int get_dimension()  nogil const 
        Box[T] compute_bounds() nogil const
        bool operator==(const Summand&) nogil




# cdef extern from "multiparameter_module_approximation/utilities.h" namespace "Gudhi::multiparameter::mma":
#     cdef cppclass MultiDiagram_point[T=*]:
#         # ctypedef T value_type
#         # ctypedef One_critical_filtration[double] filtration_type
#         MultiDiagram_point()   except + nogil
#         # MultiDiagram_point(int , T , T )   except + nogil
#         const T& get_birth()    nogil const
#         const T& get_death()  nogil const 
#         int get_dimension() nogil  const 

# cdef extern from "multiparameter_module_approximation/utilities.h" namespace "Gudhi::multiparameter::mma":
#     cdef cppclass MultiDiagram[T=*, value_type=*]:
#         MultiDiagram()   except + nogil
#         # ctypedef pair[vector[T], vector[T]] bar
#         # ctypedef vector[bar] barcode
#         # ctypedef vector[float] multipers_bar
#         # ctypedef vector[multipers_bar] multipers_barcode

#         vector[pair[vector[value_type],vector[value_type]]] get_points(const int) const  
#         vector[vector[double]] to_multipers(const int) nogil const   
#         vector[MultiDiagram_point[T]].const_iterator begin()  
#         vector[MultiDiagram_point[T]].const_iterator end()  
#         unsigned int size() const  
#         MultiDiagram_point[T]& at(unsigned int)  nogil

# cdef extern from "multiparameter_module_approximation/utilities.h" namespace "Gudhi::multiparameter::mma":
#     cdef cppclass MultiDiagrams[T=*, value_type=*]:
#         MultiDiagrams()  except + nogil
#         ctypedef pair[vector[pair[double, double]], vector[unsigned int]] plot_interface_type
#         # ctypedef vector[T] corner_type
#         # ctypedef pair[corner_type, corner_type] bar
#         # ctypedef vector[bar] barcode
#         # ctypedef vector[T] multipers_bar
#         # ctypedef vector[multipers_bar] multipers_barcode
#         # ctypedef vector[barcode] barcodes
#         vector[vector[vector[double]]] to_multipers() nogil const    
#         MultiDiagram[T, value_type]& at(const unsigned int)  nogil
#         unsigned int size() nogil const  
#         vector[MultiDiagram[T, value_type]].const_iterator begin()
#         vector[MultiDiagram[T, value_type]].const_iterator end()
#         plot_interface_type _for_python_plot(int, double)  nogil
#         vector[vector[pair[vector[value_type], vector[value_type]]]] get_points()  nogil

cdef extern from "gudhi/Multi_persistence/Module.h" namespace "Gudhi::multi_persistence":
    cdef cppclass Module[T=*]:
        cppclass Summand_of_dimension_range:
            cppclass const_it "const_iterator":
                const Summand[T]& operator*()
                const_it operator++()
                bool operator!=(const_it)
            const_it begin()
            const_it end()

        cppclass Bar:
            T operator[](size_t) nogil const

        Module()  except + nogil
        Module(const vector[intptr_t]&)
        Summand[T]& get_summand(unsigned int)  nogil
        vector[Summand[T]].iterator begin()
        vector[Summand[T]].iterator end() 
        Summand_of_dimension_range get_summand_of_dimension_range(int) nogil const
        bool operator==(const Module[T]&) nogil
        void clean(const bool)  nogil
        void add_summand(Summand[T])  nogil
        void add_summand(Summand[T], int)  nogil
        unsigned int size() const  
        Box[T] get_box() const  
        Box[T] compute_bounds() nogil const
        void set_box(Box[T])  nogil
        int get_max_dimension() const 
        vector[vector[Bar]] get_barcode_from_line(Line[T]&, const int)  nogil
        vector[vector[Bar]] get_barcodes_from_set_of_lines(const vector[Line[T]]& , const int, )  nogil
        void rescale(vector[T]&, int) nogil
        void translate(vector[T]&, int) nogil
        void evaluate_in_grid(const vector[vector[T]]&) except + nogil

cdef extern from "gudhi/Multi_persistence/module_helpers.h" namespace "Gudhi::multi_persistence":
    vector[vector[T]] compute_module_landscape[T](const Module[T]&, const int,const unsigned int,Box[T],const vector[unsigned int]&)  nogil
    vector[vector[vector[T]]] compute_set_of_module_landscapes[T](const Module[T]&, const int,const vector[unsigned int],Box[T],const vector[unsigned int]&, int)  nogil
    vector[vector[vector[T]]] compute_set_of_module_landscapes[T](const Module[T]&, const int,const vector[unsigned int],const vector[vector[T]], int)  nogil
    vector[int] compute_module_euler_curve[T](const Module[T]&, const vector[One_critical_filtration[T]]&) nogil
    vector[vector[T]] compute_module_pixels[T](const Module[T]&, vector[vector[T]], vector[int], Box[T], T, T, bool,int) nogil
    vector[vector[vector[int]]] project_module_into_grid[T](const Module[T]&, vector[vector[T]]) nogil
    vector[vector[vector[size_t]]] compute_module_lower_and_upper_generators_of[T](const Module[T]&, vector[vector[T]],bool, int) nogil
    void compute_module_distances_to[T](const Module[T]&, T*,vector[vector[T]],bool, int) nogil
    vector[T] compute_module_interleavings[T](const Module[T]&, Box[T]) nogil
    Module[T] build_permuted_module[T](const Module[T]&, const vector[int]&) nogil




cdef inline list[tuple[list[double],list[double]]] _bc2py(vector[pair[vector[double],vector[double]]] bc):
    return bc
