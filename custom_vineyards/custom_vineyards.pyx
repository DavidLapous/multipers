from cython cimport numeric
from libcpp.vector cimport vector
from libcpp.utility cimport pair
from libcpp.list cimport list as clist
from libcpp cimport bool
from libcpp cimport int
from libcpp.string cimport string
import gudhi
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import sys
import numpy

#import numpy as np
#cimport numpy as np
from libcpp.vector cimport vector




ctypedef vector[pair[int,pair[int,int]]] barcode
ctypedef vector[pair[int,pair[double,double]]] barcoded

ctypedef vector[vector[int]] boundary_matrix

ctypedef pair[pair[double,double],pair[double,double]] interval_2
ctypedef pair[vector[double],vector[double]] interval

ctypedef pair[vector[vector[double]], vector[vector[double]]] corner_list

cdef extern from "custom_vineyards.hpp":
	pair[  pair[ pair[vector[pair[int,pair[int,int]]], vector[pair[int,pair[double,double]]]], vector[int]],  pair[ pair[ pair[vector[vector[int]],pair[vector[int],vector[int]]], pair[vector[vector[int]],pair[vector[int],vector[int]]] ], pair[vector[int],vector[int]] ]  ] lower_star_vineyards_update(vector[vector[int]], vector[double], vector[int], bool, vector[vector[int]], vector[vector[int]], vector[int], vector[pair[int,pair[int,int]]], vector[int], vector[int], vector[int], vector[int], vector[int])

	vector[vector[vector[double]]] vineyards(vector[vector[double]], string, int)

cdef extern from "vineyards_trajectories.h":
	#vector<vector<vector<interval_2>>> vineyard_2d(boundary_matrix B,pair<vector<double>,vector<double>> filters_list,double precision,pair<pair<double,double>,pair<double,double>> box = {{0,0},{0,0}},bool verbose = true,bool debug = false)
	vector[vector[vector[interval_2]]] vineyard_2d(boundary_matrix, pair[vector[double], vector[double]], double, interval_2)

	#vector<vector<vector<interval_2>>> vineyard_2d(boundary_matrix B,pair<vector<double>,vector<double>> filters_list,double basepoint, double range, double precision,,bool verbose = true,bool debug = false)
	vector[vector[vector[interval_2]]] vineyard_2d(boundary_matrix, pair[vector[double], vector[double]], double, double, double)

	#vector<barcoded> compute_vineyard_2d(boundary_matrix B, pair<vector<double>,vector<double>> filters_list, double basepoint, double range, double precision)
	vector[barcoded] compute_vineyard_2d( boundary_matrix, pair[vector[double], vector[double]], double, double, double)

	vector[vector[barcoded]] vineyard_3d( boundary_matrix, vector[vector[double]], vector[double], pair[double,double], double)

	#vineyard_alt
	vector[vector[vector[interval]]] vineyard_alt(boundary_matrix B, vector[vector[double]] filters_list, double precision, pair[vector[double], vector[double]] box, bool threshold, bool multithread)
	#vineyard_alt_dim
	vector[vector[interval]] vineyard_alt_dim(boundary_matrix B, vector[vector[double]] filters_list, double precision, pair[vector[double], vector[double]] box, unsigned int dimension, bool threshold, bool multithread)
	#time_vineyard_alt
	double time_vineyard_alt(boundary_matrix B, vector[vector[double]] filters_list, double precision, pair[vector[double], vector[double]] box, bool threshold, vector[uint] nlines, bool multithread)
	#list_simplicies_to_sparse_boundary_matrix
	vector[vector[unsigned int]] list_simplicies_to_sparse_boundary_matrix(vector[vector[unsigned int]] list_simplices)
	#list_simplices_ls_filtration_to_sparse_boundary_filtration
	pair[vector[vector[unsigned int]], vector[vector[double]]]  list_simplices_ls_filtration_to_sparse_boundary_filtration(vector[vector[unsigned int]] list_simplices, vector[vector[double]] points_filtration, vector[uint] filters_to_permute)

cdef extern from "approximation.h":
	# Approximation
	vector[vector[corner_list]] approximation_vineyards(boundary_matrix B, vector[vector[double]] filters_list, double precision, pair[vector[double], vector[double]] box, bool threshold, bool complete, bool multithread, bool verbose)


def approx(B, filters, precision, box = [], threshold=False, complete=True, multithread = False, verbose = False):
	if box == [] and (type(filters) == numpy.ndarray):
		box = [[min(filters[:,0]),min(filters[:,1])],[max(filters[:,0]),max(filters[:,1])]]
	if box == [] and (type(filters) == list):
		box = [[min(filters[0]), min(filters[1])],[max(filters[0]), max(filters[1])]]
	if(type(filters) == numpy.ndarray):
		#assert filters.shape[1] == 2
		filtration = [filters[:,i] for i in range(filters.shape[1])]
	else:
		filtration = filters
	#if dimension <0: # if dimension is not specified we return every dimension
	if (type(B) == gudhi.simplex_tree.SimplexTree):
		return approximation_vineyards(simplextree_to_sparse_boundary(B), filtration, precision, box, threshold,complete, multithread, verbose)
	return approximation_vineyards(B,filtration,precision, box, threshold, complete, multithread, verbose)

	#if (type(B) == gudhi.simplex_tree.SimplexTree):
		#return approximation_vineyards(simplextree_to_sparse_boundary(B), filtration, precision, box, threshold, multithread)[dimension]
	#return approximation_vineyards(B,filtration,precision, box, threshold, multithread)[dimension]



def ls_boundary_density(list_simplices, points_filtration, to_permute = []):
	
	if (type(list_simplices) == gudhi.simplex_tree.SimplexTree):
		boundary, ls_filter = list_simplices_ls_filtration_to_sparse_boundary_filtration(
			[simplex[0] for simplex in list_simplices.get_simplices()],
			points_filtration,
			to_permute)
	else:
		boundary, ls_filter = list_simplices_ls_filtration_to_sparse_boundary_filtration(list_simplices, points_filtration, to_permute)
	return boundary, numpy.array(ls_filter).transpose()



def time_vine_alt(B, filters, precision, box = [], threshold=False, multithread = False, nlines = []):
	if box == [] and (type(filters) == numpy.ndarray):
		box = [[min(filters[:,0]),min(filters[:,1])],[max(filters[:,0]),max(filters[:,1])]]
	if box == [] and (type(filters) == list):
		box = [[min(filters[0]), min(filters[1])],[max(filters[0]), max(filters[1])]]
	if(type(filters) == numpy.ndarray):
		assert filters.shape[1] == 2
		filtration = [filters[:,0], filters[:,1]]
	else:
		filtration = filters
	if (type(B) == gudhi.simplex_tree.SimplexTree):
		return time_vineyard_alt(simplextree_to_sparse_boundary(B), filtration, precision, box, threshold, nlines, multithread)
	return time_vineyard_alt(B,filtration,precision, box, threshold, nlines, multithread)

def vine_alt(B, filters, precision, box = [], dimension = -1, threshold=False, multithread = False):
	if box == [] and (type(filters) == numpy.ndarray):
		box = [[min(filters[:,0]),min(filters[:,1])],[max(filters[:,0]),max(filters[:,1])]]
	if box == [] and (type(filters) == list):
		box = [[min(filters[0]), min(filters[1])],[max(filters[0]), max(filters[1])]]
	if(type(filters) == numpy.ndarray):
		assert filters.shape[1] == 2
		filtration = [filters[:,0], filters[:,1]]
	else:
		filtration = filters
	if dimension ==-1: # if dimension is not specified we return every dimension
		if (type(B) == gudhi.simplex_tree.SimplexTree):
			return vineyard_alt(simplextree_to_sparse_boundary(B), filtration, precision, box, threshold, multithread)
		return vineyard_alt(B,filtration,precision, box, threshold, multithread)
	if (type(B) == gudhi.simplex_tree.SimplexTree):
		return vineyard_alt_dim(simplextree_to_sparse_boundary(B), filtration, precision, box, dimension, threshold, multithread)
	return vineyard_alt_dim(B,filtration,precision, box, dimension, threshold, multithread)



def compute_ls_vineyard_update(structure, filter, dimensions, compute_barcode=True, R=[], U=[], permutation=[], bc=[], bc_inv=[], row_map_R=[], row_map_R_inv=[], row_map_U=[], row_map_U_inv=[]):
	return lower_star_vineyards_update(structure, filter, dimensions, compute_barcode, R, U, permutation, bc, bc_inv, row_map_R, row_map_R_inv, row_map_U, row_map_U_inv)

def ls_vineyards(filtrations, complex, discard):
	return vineyards(filtrations, complex, discard)

def compute_2d_vine(matrix, filters, basepoint, range_, precision):
	return compute_vineyard_2d(matrix,filters,basepoint, range_, precision)



def simplextree_to_sparse_boundary(st):
	return list_simplicies_to_sparse_boundary_matrix([simplex[0] for simplex in st.get_simplices()])


def simplextree_to_sparse_boundary_python(st, verbose=False):
	#we assume here that st has vertex name 0 to n
	max_dim = st.dimension()
	num_simplices = st.num_simplices()
	boundary = [[] for _ in range(num_simplices)]

	n_simplex_of_dim = numpy.array([0 for _ in range(max_dim+1)])

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





def vine_2d_box(simplextree, filters, precision, box=[[0,0],[5,5]]):
	if(type(filters) == numpy.ndarray):
		assert filters.shape[1] == 2
		filtration = [filters[:,0], filters[:,1]]
	else:
		filtration = filters
	if (type(simplextree) == gudhi.simplex_tree.SimplexTree):
		return vineyard_2d(simplextree_to_sparse_boundary(simplextree), filtration, precision, box)
	return vineyard_2d(simplextree, filtration, precision, box)
	#return numpy.array([numpy.array(x) for x in temp])
	#return temp


def vine_2d(simplextree, filters, basepoint, endpoint, precision):
	range_ = endpoint - basepoint
	if (type(simplextree) == gudhi.simplex_tree.SimplexTree):
		return vineyard_2d(simplextree_to_sparse_boundary(simplextree), filters, basepoint, range_, precision)
	return  vineyard_2d(simplextree, filters, basepoint, range_, precision)
	#return numpy.array([numpy.array(x) for x in temp])
	#return temp


def plot_vine_2d(matrix, filters, precision, box=[], dimension=0, return_barcodes=False, separated = False, alt = True, multithread = True, save=False, dpi=50):
	if box == [] and (type(filters) == numpy.ndarray):
		box = [[min(filters[:,0]),min(filters[:,1])],[max(filters[:,0]),max(filters[:,1])]]
	if box == [] and (type(filters) == list):
		box = [[min(filters[0]), min(filters[1])],[max(filters[0]), max(filters[1])]]
	if alt:
		temp = vine_alt(matrix, filters, precision, box, dimension = dimension, threshold = True, multithread = False)
	else:
		temp = vine_2d_box(matrix, filters, precision, box)[dimension] # dimension -> matching -> line
	#barcodes = numpy.array([numpy.array([ numpy.array([z for z in y]) for y in x]) for x in temp])
	barcodes = temp
	cmap = get_cmap("Spectral")
	n=len(barcodes)
	#number_of_trivial_features=0
	for matching in range(n):
		trivial = True
		for line in range(len(barcodes[matching])):
			birth = barcodes[matching][line][0]
			death = barcodes[matching][line][1]
			if((birth ==[]) or (death == []) or (death == birth) or (birth[0] == sys.float_info.max)):	continue
			trivial = False
			if(death[0] != sys.float_info.max and death[1] != sys.float_info.max  and birth[0] != sys.float_info.max):
				plt.plot([birth[0], death[0]], [birth[1],death[1]], c=cmap((matching)/(n)))
		if(not(trivial)):
			plt.xlim(box[0][0], box[1][0])
			plt.ylim(box[0][1], box[1][1])
		#if trivial:
			#number_of_trivial_features+=1
		if separated and not(trivial) :
			plt.show()
	if(save):	plt.savefig(save, dpi=dpi)
	plt.show()
	if(return_barcodes):
		return barcodes




from matplotlib.patches import Rectangle


"""
Defines a rectangle patch in the format {z | x  ≤ z ≤ y} with color and alpha
"""
def _rectangle(x,y,color, alpha):
	return Rectangle(x, max(y[0]-x[0],0),max(y[1]-x[1],0), color=color, alpha=alpha)

def plot_approx_2d(B, filters, precision, box = [], dimension=0, return_corners=False, separated=False, multithread = False, complete=True, alpha=1, verbose = False, save=False, dpi=50):
	if box == [] and (type(filters) == numpy.ndarray):
		box = [[min(filters[:,0]),min(filters[:,1])],[max(filters[:,0]),max(filters[:,1])]]
	if box == [] and (type(filters) == list):
		box = [[min(filters[0]), min(filters[1])],[max(filters[0]), max(filters[1])]]

	corners = approx(B,filters,precision,box=box,threshold=1, multithread = multithread, complete = complete, verbose = verbose)[dimension]
	cmap = get_cmap("Spectral")
	_, ax = plt.subplots()
	ax.set(xlim=[box[0][0],box[1][0]],ylim=[box[0][1],box[1][1]])
	n_summands = len(corners)

	for i in range(n_summands):
		for birth in corners[i][0]:
			for death in corners[i][1]:
				ax.add_patch(_rectangle(birth,death,cmap(i/n_summands),alpha))
		if separated:
			if i < n_summands -1:
				plt.show()
				fig,ax= plt.subplots()
				ax.set(xlim=[box[0][0],box[1][0]],ylim=[box[0][1],box[1][1]])
	if save:
		plt.savefig(save, dpi=dpi)
	plt.show()
	if return_corners:
		return corners


