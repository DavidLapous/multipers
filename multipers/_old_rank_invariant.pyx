from libcpp.vector cimport vector
from libc.stdint cimport intptr_t
from libcpp cimport bool
from libcpp cimport int
from libcpp.utility cimport pair
from typing import Optional,Iterable,Callable

ctypedef vector[vector[vector[vector[int]]]] rank2
ctypedef vector[vector[int]] grid2D
ctypedef vector[grid2D] grid3D
ctypedef vector[grid3D] grid4D
ctypedef vector[grid4D] grid5D


ctypedef pair[vector[vector[int]],vector[int]] signed_measure_type
ctypedef pair[vector[vector[double]],vector[int]] signed_measure_double_type








cdef extern from "multi_parameter_rank_invariant/_old_rank_invariant.h" namespace "Gudhi::multiparameter::rank_invariant":
	rank2 get_2drank_invariant(const intptr_t, const vector[int]&, const int) nogil
	grid3D get_2Dhilbert(const intptr_t, const vector[int]&, const vector[int], bool)  except + nogil
	vector[signed_measure_type] get_signed_measure(const intptr_t, const vector[int]&, int, vector[int], bool)  except + nogil
	grid4D get_3Dhilbert(const intptr_t, const vector[int]&, const vector[int])  except + nogil
	grid5D get_4Dhilbert(const intptr_t, const vector[int]&, const vector[int])  except + nogil
	grid2D get_euler2d(const intptr_t, const vector[int]&, bool, bool)  except + nogil
	grid3D get_euler3d(const intptr_t, const vector[int]&, bool, bool)  except + nogil
	grid4D get_euler4d(const intptr_t, const vector[int]&, bool, bool)  except + nogil
	

cdef extern from "multi_parameter_rank_invariant/function_rips.h" namespace "Gudhi::multiparameter::rank_invariant::degree_rips":
	signed_measure_double_type degree_rips_hilbert_signed_measure(intptr_t, int,int)  except + nogil


from multipers.simplex_tree_multi import SimplexTreeMulti # Typing hack

cimport numpy as cnp
cnp.import_array()
import numpy as np

# # TODO : make a st python flag for coordinate_st, with grid resolution.
def rank_invariant2d(simplextree:SimplexTreeMulti, grid_shape:np.ndarray|list, int degree):
	cdef intptr_t ptr = simplextree.thisptr
	cdef int c_degree = degree
	cdef vector[int] c_grid_shape = grid_shape
	cdef vector[vector[vector[vector[int]]]] out
	with nogil:
		out = get_2drank_invariant(ptr, c_grid_shape, c_degree)
	return np.asarray(out, dtype=int)



def _hilbert2d(
		simplextree:SimplexTreeMulti, 
		grid_shape:np.ndarray|list, 
		vector[int] degrees, 
		bool mobius_inversion
	):
	# assert simplextree.num_parameters == 2
	cdef intptr_t ptr = simplextree.thisptr
	cdef vector[int] c_grid_shape = grid_shape
	cdef grid3D out
	with nogil:
		out = get_2Dhilbert(ptr, c_grid_shape, degrees, mobius_inversion)
	return np.array(out, dtype=int)

# cdef _sm_2d(simplextree:SimplexTreeMulti, grid_shape:np.ndarray|list, int degree):
# 	# assert simplextree.num_parameters == 2
# 	cdef intptr_t ptr = simplextree.thisptr
# 	cdef int c_degree = degree
# 	cdef vector[int] c_grid_shape = grid_shape
# 	cdef signed_measure_type out
# 	with nogil:
# 		out = get_2D_SM(ptr, c_grid_shape, c_degree)
# 	return np.array(out.first, dtype=int), np.array(out.second, dtype=int)

cdef _hilbert3d(simplextree:SimplexTreeMulti, grid_shape:np.ndarray|list, vector[int] degrees):
	# assert simplextree.num_parameters == 3
	cdef intptr_t ptr = simplextree.thisptr
	cdef vector[int] c_grid_shape = grid_shape
	cdef grid4D out
	with nogil:
		out = get_3Dhilbert(ptr, c_grid_shape, degrees)
	return np.array(out, dtype=int)

cdef _hilbert4d(simplextree:SimplexTreeMulti, grid_shape:np.ndarray|list, vector[int] degrees):
	# assert simplextree.num_parameters == 3
	cdef intptr_t ptr = simplextree.thisptr
	cdef vector[int] c_grid_shape = grid_shape
	cdef grid5D out
	with nogil:
		out = get_4Dhilbert(ptr, c_grid_shape, degrees)
	return np.array(out, dtype=int)


def hilbert(simplextree:SimplexTreeMulti, grid_shape:np.ndarray|list, vector[int] degrees):
	assert len(grid_shape) >= simplextree.num_parameters, "Grid shape not valid"
	if simplextree.num_parameters == 2:
		out = _hilbert2d(simplextree, grid_shape, degrees, False)
		return out
	if simplextree.num_parameters == 3:
		return _hilbert3d(simplextree, grid_shape, degrees)
	if simplextree.num_parameters == 4:
		return _hilbert4d(simplextree, grid_shape, degrees)
	raise Exception(f"Number of parameter has to be 2,3, or 4.")

def euler2d(simplextree:SimplexTreeMulti, grid_shape:np.ndarray|list, bool inverse=True, bool zero_pad=False):
	cdef intptr_t ptr = simplextree.thisptr
	cdef vector[int] c_grid_shape = grid_shape
	cdef vector[vector[int]] out
	with nogil:
		out = get_euler2d(ptr, c_grid_shape, inverse, zero_pad)
	return np.array(out, dtype=int)
def euler3d(simplextree:SimplexTreeMulti, grid_shape:np.ndarray|list, bool inverse=True, bool zero_pad=False):
	cdef intptr_t ptr = simplextree.thisptr
	cdef vector[int] c_grid_shape = grid_shape
	cdef grid3D out
	with nogil:
		out = get_euler3d(ptr, c_grid_shape, inverse, zero_pad)
	return np.array(out, dtype=int)
def euler4d(simplextree:SimplexTreeMulti, grid_shape:np.ndarray|list, bool inverse=True, bool zero_pad=False):
	cdef intptr_t ptr = simplextree.thisptr
	cdef vector[int] c_grid_shape = grid_shape
	cdef grid4D out
	with nogil:
		out = get_euler4d(ptr, c_grid_shape,inverse, zero_pad)
	return np.array(out, dtype=int)

def euler(simplextree:SimplexTreeMulti, grid_shape:np.ndarray|list, degree:int=None, bool inverse=False, bool zero_pad=False):
	if simplextree.num_parameters == 2:
		return euler2d(simplextree, grid_shape, inverse, zero_pad)
	if simplextree.num_parameters == 3:
		return euler3d(simplextree, grid_shape, inverse, zero_pad)
	if simplextree.num_parameters == 4:
		return euler4d(simplextree, grid_shape, inverse, zero_pad)
	raise Exception(f"Number of parameter has to be 2,3, or 4.")


def signed_measure(
	simplextree:SimplexTreeMulti,
	grid_shape:Optional[np.ndarray]|list|int|None=None, 
	degree:Optional[int]|None=None,
	degrees:Optional[Iterable[int]]=None,
	grid_conversion=None,
	infer_grid=True,
	bool unsparse=False,
	invariant:str | None="rank_invariant",
	plot:bool=False,
	grid_strategy:str='exact',
	mass_default:str|None|Iterable[float]='auto',
	**get_filtration_grid_kwargs,
	):
	"""
	Computes a discrete signed measure from various invariants.

	Parameters
	----------
	- simplextree : SimplexTreeMulti
		The multifiltered complex on which to compute the invariant.
		The filtrations values are assumed to be the coordinates in that filtration, i.e. integers
	- grid_conversion : (num_parameter) x (filtration values)
		Reconverts the coordinate signed measure in that grid.
		Default behavior is searching the grid in the simplextree.
	- grid_shape : Iterable[int] or Int or None:
		The coordinate grid shape. 
		If int, every parameter gets this resolution. 
		If None and grid_conversion is also None, the grid is infered from the filtration values.
	- degree : int|None
		If the invariant is hilbert or the rank invariant, the homological degree.
		If the invariant is euler, this parameter is ignored.
	- degrees :
		Compute multiple signed measures of different degrees at once.
	- mass_default='auto' : filtration value or 'auto' or None
		where to put the mass default to ensure having a zero mass measure. If None, the mass default is not corrected.
	- unsparse = False : bool
		Unsparse the output.
	- invariant = None : str
		The invariant to use to compute the signed measure. 
		Possible options : 'euler' or 'hilbert' or 'rank_invariant'
	- plot : bool
		Will plot the signed measure if possible, i.e. if num_parameters is 2
	
	Output
	------

	Default
		- dirac locations : np.ndarray of shape (num_diracs x simplextree.num_parameters)
		- dirac weights : np.ndarray of shape (num_diracs)
	
	if unsparse is true, returns the unsparsified tensor. 
	"""
	cdef bool zero_pad = not (mass_default is None)
	if degree is None and degrees is None:
		degrees=[None]
	elif degree is not None and degrees is None:
		degrees = [degree]
	elif degree is not None and degrees is not None:
		degrees = np.concatenate([[degree], degrees])
	
	_unique_degrees = np.unique(degrees)
	assert (_unique_degrees[-1] is None and len(_unique_degrees) == 1) or np.issubdtype(_unique_degrees.dtype,int), f"One invariant at the time ! Asked for computations of {degrees}"

	if _unique_degrees[0] is None:
		assert invariant not in ["hilbert", "rank_invariant"], f"Provide a degree to compute {invariant} !"
		invariant=2
		degree=0
	else:
		invariant=1 if invariant is None or invariant is "hilbert" else 3

	if grid_conversion is None and grid_shape is None:
		if len(simplextree.filtration_grid[0]) > 0 and infer_grid:
			grid_conversion = [np.asarray(f) for f in simplextree.filtration_grid]
		else:
			if infer_grid:
				grid_conversion = simplextree.get_filtration_grid(grid_strategy=grid_strategy,**get_filtration_grid_kwargs)
				if zero_pad:
					mass_default_location = (1.1*f[-1] - f[0] for f in grid_conversion) if mass_default=='auto' else mass_default
					grid_conversion = [np.concatenate([f, [xi]]) for f,xi in zip(grid_conversion, mass_default_location)]
				## Copy necessary, as we want a coordinate simplextree
				simplextree_copy = SimplexTreeMulti(simplextree)
				simplextree_copy, simplextree = simplextree, simplextree_copy
				simplextree.grid_squeeze(grid_conversion, coordinate_values=True)
			else:
				grid_shape = np.asarray(simplextree.filtration_bounds()[1], dtype=int)+1

	if grid_conversion is not None and len(simplextree.filtration_grid[0]) == 0 and infer_grid:
		simplextree.grid_squeeze(grid_conversion, coordinate_values=True)
	try:
		int(grid_shape)
		grid_shape = [grid_shape]*simplextree.num_parameters
	except:
		if grid_shape is None:
			grid_shape = [len(f) for f in grid_conversion]
	
	num_degrees = len(degrees)
	cdef intptr_t ptr = simplextree.thisptr
	cdef vector[int] c_grid_shape = grid_shape
	cdef vector[signed_measure_type] _out
	cdef int cinvariant =invariant
	# cdef int cdegree = degree
	if invariant == 2:	degrees=np.array([],dtype=int)
	cdef vector[int] cdegrees = degrees
	num_parameters = simplextree.num_parameters
	if invariant == 3:
		assert simplextree.num_parameters == 2, "Rank invariant computations are limited for 2-parameter simplextree for the moment"
		assert len(degrees) ==1, 'One degree at the time, WIP.'
		from multipers.ml.signed_betti import rank_decomposition_by_rectangles
		from torch import Tensor
		out = rank_invariant2d(simplextree, grid_shape,degrees[0])
		out = rank_decomposition_by_rectangles(out, threshold= zero_pad)
		out = Tensor(out.copy()).to_sparse()
		def _is_trivial(rectangle:np.ndarray):
			birth=rectangle[:num_parameters]
			death=rectangle[num_parameters:]
			return np.all(birth<=death) and not np.array_equal(birth,death)
		coords = np.asarray(out.indices().T, dtype=int)
		weights = np.asarray(out.values(), dtype=int)
		correct_indices = np.asarray([_is_trivial(rectangle) for rectangle in coords])
		if len(correct_indices) == 0:
			pts, weights = np.empty((0, num_parameters)), np.empty((0))
		else:
			pts = np.asarray(coords[correct_indices])
			weights = weights[correct_indices]
		out = [ ## same output format as others
			(pts,weights)
		]
	else:
		with nogil:
			_out = get_signed_measure(ptr, c_grid_shape, cinvariant, cdegrees, zero_pad)
	
		out = [[np.asarray(_out[i].first, dtype=int), np.asarray(_out[i].second, dtype=int)] for i in range(num_degrees)]

	for i,(pts, weights) in enumerate(out):
		if len(pts) == 0:	
			out[i][0]=np.empty(shape=(0,2*num_parameters if invariant == 3 else num_parameters), dtype=int)
	
	if unsparse:
		from torch import sparse_coo_tensor
		if invariant == 3:
			grid_shape = list(grid_shape) + list(grid_shape)
		return [np.asarray(sparse_coo_tensor(indices=pts.T,values=weights, size=tuple(grid_shape)).to_dense(), dtype=weights.dtype) for pts,weights in out]
	
	if grid_conversion is not None:
		for degree,(pts,weights) in enumerate(out):
			coords = np.empty(shape=pts.shape)
			for i in range(coords.shape[1]):
				coords[:,i] = grid_conversion[i % num_parameters][pts[:,i]]
			out[degree]=(coords, weights)

	if plot:
		assert simplextree.num_parameters == 2, "Plot only available for 2-parameter persistence."
		from multipers.plots import plot_signed_measures
		plot_signed_measures(out)
		# fig, axes = plt.subplots(nrows=1,ncols=num_degrees)
		# if num_degrees == 1:
		# 	axes = [axes]
		# for ax, signed_measure in zip(axes,out):
			# plt.sca(ax)
			# # ax.set_aspect('equal')
			# if (invariant != 3):
			# 	color_weights = np.empty(weights.shape, dtype=int)
			# 	color_weights[weights>0] = 1
			# 	color_weights[weights<0] = -1
			# 	plt.scatter(coords[:,0],coords[:,1], c=color_weights, cmap="coolwarm") 
			# else:
			# 	def _plot_rectangle(rectangle:np.ndarray, weight):
			# 		x_axis=rectangle[[0,2]]
			# 		y_axis=rectangle[[1,3]]
			# 		color = "blue" if weight > 0 else "red"
			# 		plt.plot(x_axis, y_axis, c=color)
			# 	for rectangle, weight in zip(coords, weights):
			# 		_plot_rectangle(rectangle=rectangle, weight=weight)
	return out

def degree_rips(simplextree, int num_degrees, int homological_degree):
	cdef intptr_t ptr = simplextree.thisptr
	cdef signed_measure_double_type out
	with nogil:
		out = degree_rips_hilbert_signed_measure(ptr, num_degrees, homological_degree)
	pts, weights = np.asarray(out.first), np.asarray(out.second, dtype=int)
	if len(pts) == 0:	
		pts=np.empty(shape=(0,2), dtype=float)
	return  pts,weights
