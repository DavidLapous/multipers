"""!
@package mma
@brief Files containing the C++ cythonized functions.
@author David Loiseaux, Mathieu Carrière
@copyright Copyright (c) 2022 Inria.
"""

from cython cimport numeric
from libcpp.vector cimport vector
from libcpp.utility cimport pair
#from libcpp.list cimport list as clist
from libcpp cimport bool
from libcpp cimport int
from libcpp.string cimport string
import gudhi as _gd
import matplotlib.pyplot as _plt
from matplotlib.cm import get_cmap as _get_cmap
import sys as _sys
import numpy as _np

from libcpp.vector cimport vector

###########################################################################

ctypedef vector[pair[int,pair[int,int]]] barcode
ctypedef vector[pair[int,pair[double,double]]] barcoded

ctypedef vector[vector[unsigned int]] boundary_matrix

ctypedef pair[pair[double,double],pair[double,double]] interval_2
ctypedef pair[vector[double],vector[double]] interval

ctypedef pair[vector[vector[double]], vector[vector[double]]] corner_list


cdef extern from "vineyards_trajectories.h" namespace "Vineyard":
	#vineyard_alt
	vector[vector[vector[interval]]] compute_vineyard_barcode(boundary_matrix B, vector[vector[double]] filters_list, double precision, pair[vector[double], vector[double]] box, bool threshold, bool multithread, bool verbose)
	#vineyard_alt_dim
	vector[vector[interval]] compute_vineyard_barcode_in_dimension(boundary_matrix B, vector[vector[double]] filters_list, double precision, pair[vector[double], vector[double]] box, unsigned int dimension, bool threshold, bool verbose)

cdef extern from "approximation.h" namespace "Vineyard":
	# Approximation
	vector[vector[corner_list]] compute_vineyard_barcode_approximation(boundary_matrix B, vector[vector[double]] filters_list, double precision, pair[vector[double], vector[double]] box, bool threshold, bool keep_order, bool complete, bool multithread, bool verbose)

cdef extern from "images.h":
	#image_2d_from_boundary_matrix
	vector[vector[vector[double]]] get_2D_image_from_boundary_matrix(boundary_matrix &B, vector[vector[double]] &filters_list, double precision, pair[vector[double], vector[double]] &box, const double delta, const vector[unsigned int] &resolution, const int dimension, bool complete, bool verbose)

cdef extern from "benchmarks.h":
	#time_vineyard_alt
	double time_vineyard_barcode_computation(boundary_matrix B, vector[vector[double]] filters_list, double precision, pair[vector[double], vector[double]] box, bool threshold, bool multithread, bool verbose)

	#time_approximation
	double time_approximated_vineyard_barcode_computation(boundary_matrix B, vector[vector[double]] filters_list, double precision, pair[vector[double], vector[double]] box, bool threshold, bool keep_order, bool complete, bool multithread, bool verbose)

	double time_2D_image_from_boundary_matrix_construction(boundary_matrix &B, vector[vector[double]] &filters_list, double precision, pair[vector[double], vector[double]] &box, const double delta, const vector[unsigned int] &resolution, const unsigned int dimension, bool complete, bool verbose)

cdef extern from "format_python-cpp.h":
	#list_simplicies_to_sparse_boundary_matrix
	vector[vector[unsigned int]] build_sparse_boundary_matrix_from_simplex_list(vector[vector[unsigned int]] list_simplices)
	#list_simplices_ls_filtration_to_sparse_boundary_filtration
	pair[vector[vector[unsigned int]], vector[vector[double]]] build_boundary_matrix_from_simplex_list(vector[vector[unsigned int]] list_simplices, vector[vector[double]] filtrations, vector[unsigned int] filters_to_permute)

###########################################################################

def time_image_2d(B, filters, precision=-1, box=[], bandwidth=-1, resolution=[100,100], dimension=0, complete=True, verbose = False):
	if box == [] and (type(filters) == _np.ndarray):
		box = [[min(filters[:,0]),min(filters[:,1])],[max(filters[:,0]),max(filters[:,1])]]
	if box == [] and (type(filters) == list):
		box = [[min(filters[0]), min(filters[1])],[max(filters[0]), max(filters[1])]]
	if(type(filters) == _np.ndarray):
		#assert filters.shape[1] == 2
		filtration = [filters[:,i] for i in range(filters.shape[1])]
	else:
		filtration = filters
	if precision <=0.00001:
		precision = _np.linalg.norm(_np.array(box[0]) - _np.array(box[1]))/100 # This makes around 100-200 lines
	if bandwidth <= 0.000001:
		bandwidth = 1/_np.min(_np.array(resolution))
	if (type(B) == _gd.simplex_tree.SimplexTree):
		time =	 time_2D_image_from_boundary_matrix_construction(simplextree_to_sparse_boundary(B), filtration, precision, box, bandwidth, resolution, dimension, complete, verbose)
	else:
		time =		 time_2D_image_from_boundary_matrix_construction(B, filtration, precision, box, bandwidth, resolution, dimension, complete, verbose)

	return time

def persistence_image_2d(B, filters, precision=-1, box=[], bandwidth=-1, resolution=[100,100], dimension=-1, complete=True, verbose = False, plot = True, save=False):
	if box == [] and (type(filters) == _np.ndarray):
		box = [[min(filters[:,0]),min(filters[:,1])],[max(filters[:,0]),max(filters[:,1])]]
	if box == [] and (type(filters) == list):
		box = [[min(filters[0]), min(filters[1])],[max(filters[0]), max(filters[1])]]
	if(type(filters) == _np.ndarray):
		#assert filters.shape[1] == 2
		filtration = [filters[:,i] for i in range(filters.shape[1])]
	else:
		filtration = filters
	if precision <=0.00001:
		precision = _np.linalg.norm(_np.array(box[0]) - _np.array(box[1]))/100 # This makes around 100-200 lines
	if bandwidth <= 0.000001:
		bandwidth = 1/_np.min(_np.array(resolution))
	if (type(B) == _gd.simplex_tree.SimplexTree):
		image_vector =	 _np.array(get_2D_image_from_boundary_matrix(simplextree_to_sparse_boundary(B), filtration, precision, box, bandwidth, resolution, dimension, complete, verbose))
	else:
		image_vector =	 _np.array(get_2D_image_from_boundary_matrix(B, filtration, precision, box, bandwidth, resolution, dimension, complete, verbose))
	## Fixes images
	i=0
	for image in image_vector:
		if(plot or save):
			_plt.imshow(_np.flip(image.transpose(),0),extent=[box[0][0], box[1][0], box[0][1], box[1][1]], aspect = (box[1][0]-box[0][0]) / (box[1][1]-box[0][1]))
			_plt.colorbar()
			if save:
				_plt.savefig(save+"_H"+str(i)+".png")
				i+=1
			_plt.show()
	if dimension<0:
		return image_vector
	return image_vector[0]


def time_approx(B, filters, precision, box=[], threshold=False,complete=False,multithread=False, verbose=False, keep_order = False):
	""" Benchmarks the time taken by an approximation call. See \ref approx for parameters documentation.
	@return  Time took by the c++ part (double).
	"""
	if box == [] and (type(filters) == _np.ndarray):
		box = [[min(filters[:,0]),min(filters[:,1])],[max(filters[:,0]),max(filters[:,1])]]
	if box == [] and (type(filters) == list):
		box = [[min(filters[0]), min(filters[1])],[max(filters[0]), max(filters[1])]]
	if(type(filters) == _np.ndarray):
		#assert filters.shape[1] == 2
		filtration = [filters[:,i] for i in range(filters.shape[1])]
	else:
		filtration = filters
	if precision <=0.00001:
		precision = _np.linalg.norm(_np.array(box[0]) - _np.array(box[1]))/100 # This makes around 100-200 lines
	if (type(B) == _gd.simplex_tree.SimplexTree):
		return time_approximated_vineyard_barcode_computation(simplextree_to_sparse_boundary(B), filtration, precision, box, threshold, keep_order, complete, multithread, verbose)
	return time_approximated_vineyard_barcode_computation(B,filtration,precision, box, threshold, keep_order, complete, multithread, verbose)

def approx(B, filters, precision, box = [], threshold=False, complete=True, multithread = False, verbose = False, keep_order = False):
	if box == [] and (type(filters) == _np.ndarray):
		box = [[min(filters[:,0]),min(filters[:,1])],[max(filters[:,0]),max(filters[:,1])]]
	if box == [] and (type(filters) == list):
		box = [[min(filters[0]), min(filters[1])],[max(filters[0]), max(filters[1])]]
	if(type(filters) == _np.ndarray):
		#assert filters.shape[1] == 2
		filtration = [filters[:,i] for i in range(filters.shape[1])]
	else:
		filtration = filters
	if precision <=0.00001:
		precision = _np.linalg.norm(_np.array(box[0]) - _np.array(box[1]))/100 # This makes around 100-200 lines
	if (type(B) == _gd.simplex_tree.SimplexTree):
		return compute_vineyard_barcode_approximation(simplextree_to_sparse_boundary(B), filtration, precision, box, threshold, keep_order, complete, multithread,verbose)
	return compute_vineyard_barcode_approximation(B,filtration,precision, box, threshold, keep_order, complete, multithread,verbose)

def ls_boundary_density(list_simplices, points_filtration, to_permute = []):
	
	if (type(list_simplices) == _gd.simplex_tree.SimplexTree):
		boundary, ls_filter = build_boundary_matrix_from_simplex_list(
			[simplex[0] for simplex in list_simplices.get_simplices()],points_filtration,
			to_permute)
	else:
		boundary, ls_filter = build_boundary_matrix_from_simplex_list(list_simplices, points_filtration, to_permute)
	return boundary, _np.array(ls_filter).transpose()

def time_vine_alt(B, filters, precision, box = [], threshold=False, multithread = False, verbose = False):
	if box == [] and (type(filters) == _np.ndarray):
		box = [[min(filters[:,0]),min(filters[:,1])],[max(filters[:,0]),max(filters[:,1])]]
	if box == [] and (type(filters) == list):
		box = [[min(filters[0]), min(filters[1])],[max(filters[0]), max(filters[1])]]
	if(type(filters) == _np.ndarray):
		assert filters.shape[1] == 2
		filtration = [filters[:,0], filters[:,1]]
	else:
		filtration = filters
	if (type(B) == _gd.simplex_tree.SimplexTree):
		return time_vineyard_barcode_computation(simplextree_to_sparse_boundary(B), filtration, precision, box, threshold, multithread, verbose)
	return time_vineyard_barcode_computation(B,filtration,precision, box, threshold, multithread, verbose)

def vine_alt(B, filters, precision, box = [], dimension = -1, threshold=False, multithread = False, verbose = False):
	if box == [] and (type(filters) == _np.ndarray):
		box = [[min(filters[:,0]),min(filters[:,1])],[max(filters[:,0]),max(filters[:,1])]]
	if box == [] and (type(filters) == list):
		box = [[min(filters[0]), min(filters[1])],[max(filters[0]), max(filters[1])]]
	if(type(filters) == _np.ndarray):
		assert filters.shape[1] == 2
		filtration = [filters[:,0], filters[:,1]]
	else:
		filtration = filters
	if dimension ==-1: # if dimension is not specified we return every dimension
		if (type(B) == _gd.simplex_tree.SimplexTree):
			return compute_vineyard_barcode(simplextree_to_sparse_boundary(B), filtration, precision, box, threshold, multithread, verbose)
		return compute_vineyard_barcode(B,filtration,precision, box, threshold, multithread, verbose)
	if (type(B) == _gd.simplex_tree.SimplexTree):
		return compute_vineyard_barcode_in_dimension(simplextree_to_sparse_boundary(B), filtration, precision, box, dimension, threshold, verbose)
	return compute_vineyard_barcode_in_dimension(B,filtration,precision, box, dimension, threshold, verbose)

def simplextree_to_sparse_boundary(st):
	return build_sparse_boundary_matrix_from_simplex_list([simplex[0] for simplex in st.get_simplices()])

def simplextree_to_sparse_boundary_python(st, verbose=False):
	#we assume here that st has vertex name 0 to n
	max_dim = st.dimension()
	num_simplices = st.num_simplices()
	boundary = [[] for _ in range(num_simplices)]

	n_simplex_of_dim = _np.array([0 for _ in range(max_dim+1)])

	def get_id(s):
		s_dim = len(s)-1
		j = sum(n_simplex_of_dim[0:s_dim])
		for s2 in st.get_skeleton(s_dim):
			if s2[0] == s:
				return j
			if len(s2[0])-1 == s_dim:
				j+=1
		return -1
	for dim in range(max_dim+1):
		for simplex in st.get_skeleton(dim):
			if len(simplex[0])-1 != dim:
				continue
			n_simplex_of_dim[dim] +=1
			simplex_id = get_id(simplex[0])
			if verbose:
				print(simplex[0],simplex_id, n_simplex_of_dim)
			for simplex_in_boundary in st.get_boundaries(simplex[0]):
				boundary[simplex_id] += [get_id(simplex_in_boundary[0])]
	return boundary

def simplextree_to_boundary(st):
	return [[simplex_in_boundary[0] for simplex_in_boundary in st.get_boundaries(simplex[0])] for simplex in st.get_simplices()]

def plot_vine_2d(matrix, filters, precision, box=[], dimension=0, return_barcodes=False, separated = False, multithread = True, save=False, dpi=50):
	if box == [] and (type(filters) == _np.ndarray):
		box = [[min(filters[:,0]),min(filters[:,1])],[max(filters[:,0]),max(filters[:,1])]]
	if box == [] and (type(filters) == list):
		box = [[min(filters[0]), min(filters[1])],[max(filters[0]), max(filters[1])]]
	temp = vine_alt(matrix, filters, precision, box, dimension = dimension, threshold = True, multithread = False)
	#barcodes = _np.array([_np.array([ _np.array([z for z in y]) for y in x]) for x in temp])
	barcodes = temp
	cmap = _get_cmap("Spectral")
	n=len(barcodes)
	#number_of_trivial_features=0
	for matching in range(n):
		trivial = True
		for line in range(len(barcodes[matching])):
			birth = barcodes[matching][line][0]
			death = barcodes[matching][line][1]
			if((birth ==[]) or (death == []) or (death == birth) or (birth[0] == _sys.float_info.max)):	continue
			trivial = False
			if(death[0] != _sys.float_info.max and death[1] != _sys.float_info.max  and birth[0] != _sys.float_info.max):
				_plt.plot([birth[0], death[0]], [birth[1],death[1]], c=cmap((matching)/(n)))
		if(not(trivial)):
			_plt.xlim(box[0][0], box[1][0])
			_plt.ylim(box[0][1], box[1][1])
		#if trivial:
			#number_of_trivial_features+=1
		if separated and not(trivial) :
			_plt.show()
	if(save):	_plt.savefig(save, dpi=dpi)
	_plt.show()
	if(return_barcodes):
		return barcodes

from matplotlib.patches import Rectangle as _Rectangle
"""
Defines a rectangle patch in the format {z | x  ≤ z ≤ y} with color and alpha
"""
def _rectangle(x,y,color, alpha):
	return _Rectangle(x, max(y[0]-x[0],0),max(y[1]-x[1],0), color=color, alpha=alpha)

def _d_inf(a,b):
	if type(a) != _np.ndarray or type(b) != _np.ndarray:
		a = _np.array(a)
		b = _np.array(b)
	return _np.min(_np.abs(b-a))

def plot_approx_2d(B, filters, precision, box = [], dimension=0, return_corners=False, separated=False, min_interleaving = 0, multithread = False, complete=True, alpha=1, verbose = False, save=False, dpi=50, keep_order=False, shapely = True):
	try:
		from shapely.geometry import box as _rectangle_box
		from shapely.ops import unary_union
	except ModuleNotFoundError:
		print("Fallbacking to matplotlib instead of shapely.")
		shapely = False
	if alpha >= 1:
		shapely = False # Not sure which one is quicker in that case.
	if box == [] and (type(filters) == _np.ndarray):
		box = [[min(filters[:,0]),min(filters[:,1])],[max(filters[:,0]),max(filters[:,1])]]
	if box == [] and (type(filters) == list):
		box = [[min(filters[0]), min(filters[1])],[max(filters[0]), max(filters[1])]]

	corners = approx(B,filters,precision,box=box,threshold=1, multithread = multithread, complete = complete, verbose = verbose, keep_order = keep_order)[dimension]
	cmap = _get_cmap("Spectral")
	if not(separated):
		_, ax = _plt.subplots()
		ax.set(xlim=[box[0][0],box[1][0]],ylim=[box[0][1],box[1][1]])
	n_summands = len(corners)

	for i in range(n_summands):
		trivial_summand = True
		list_of_rect = []
		for birth in corners[i][0]:
			for death in corners[i][1]:
				if death[1]>birth[1] and death[0]>birth[0]:
					if trivial_summand and _d_inf(birth,death)>min_interleaving:
						trivial_summand = False
					if shapely:
						list_of_rect.append(_rectangle_box(birth[0], birth[1], death[0],death[1]))
					else:
						list_of_rect.append(_rectangle(birth,death,cmap(i/n_summands),alpha))
		if not(trivial_summand):
			if separated:
				fig,ax= _plt.subplots()
				ax.set(xlim=[box[0][0],box[1][0]],ylim=[box[0][1],box[1][1]])
			if shapely:
				summand_shape = unary_union(list_of_rect)
				xs,ys=summand_shape.exterior.xy
				ax.fill(xs,ys,alpha=alpha, fc=cmap(i/n_summands), ec='None')
			else:
				for rectangle in list_of_rect:
					ax.add_patch(rectangle)
			if separated:
				_plt.show()

	if save:
		_plt.savefig(save, dpi=dpi)
	if not(separated):
		_plt.show()
	if return_corners:
		return corners


