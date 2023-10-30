# cimport multipers.tensor as mt
from libc.stdint cimport intptr_t, uint16_t, uint32_t, int32_t, int64_t
from libcpp.vector cimport vector
from libcpp cimport bool, int, float
import numpy as np
cimport numpy as cnp
cnp.import_array()

ctypedef fused some_int:
	int32_t
	int64_t
	int

ctypedef fused some_float:
	float
	double


## TODO : optimize. The filtrations is a list of filtrations, but thats not easily convertible to a cython type without copies
import cython
@cython.boundscheck(False)
@cython.wraparound(False)
def integrate_measure(
	some_float[:,:] pts, 
	some_int[:] weights, 
	filtrations
):
	resolution = np.asarray([len(f) for f in filtrations])
	cdef int num_pts = pts.shape[0]
	cdef int num_parameters = pts.shape[1]
	assert weights.shape[0] == num_pts
	out = np.zeros(shape=resolution, dtype=np.int32) ## dim cannot be known at compiletime
	# cdef some_float[:] filtration_of_parameter
	# cdef cnp.ndarray indices = np.zeros(shape=num_parameters, dtype=int)
	for i in range(num_pts): ## this is slow...
		# for parameter in range(num_parameters):
		# 	indices[parameter] = np.asarray(filtrations[parameter]) >= pts[i,parameter]
		indices = (
			# np.asarray(
			# 	<some_float[:]>(&filtrations[parameter][0])
			# 	) ## I'm not sure why it doesn't work
			np.asarray(filtrations[parameter])
			>=
			pts[i,parameter] for parameter in range(num_parameters)
		) ## iterator so that below there is no copy

		out[np.ix_(*indices)] += weights[i] # This is slow...
	return out

## for benchmark purposes
def integrate_measure_python(pts, weights, filtrations):
	resolution = tuple([len(f) for f in filtrations])
	out = np.zeros(shape=resolution, dtype=pts.dtype)
	num_pts = pts.shape[0]
	num_parameters = pts.shape[1]
	for i in range(num_pts): #this is slow.
		indices = (filtrations[parameter]>=pts[i][parameter] for parameter in range(num_parameters))
		out[np.ix_(*indices)] += weights[i]
	return out