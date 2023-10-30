from libc.stdint cimport uint16_t
from libcpp.vector cimport vector
from libcpp cimport bool, float


ctypedef float dtype
ctypedef uint16_t index_type

cdef extern from "tensor/tensor.h" namespace "tensor":
	cdef cppclass static_tensor_view[float, uint16_t]:
		static_tensor_view()   except + nogil
		static_tensor_view(dtype*,const  vector[index_type]&) except + nogil
		const vector[index_type]& get_resolution() 
