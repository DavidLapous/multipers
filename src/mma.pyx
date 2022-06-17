"""!
@package mma
@brief Files containing the C++ cythonized functions.
@author David Loiseaux, Mathieu Carrière
@copyright Copyright (c) 2022 Inria.
"""

# distutils: language = c++

###########################################################################
#PYTHON LIBRARIES
import gudhi as _gd
from gudhi.simplex_tree import SimplexTree as _SimplexTree
import matplotlib.pyplot as _plt
from matplotlib.cm import get_cmap as _get_cmap
import sys as _sys
import numpy as _np
from typing import List, Union
from os.path import exists as _exists
from os import remove as _remove
from tqdm import tqdm as _tqdm
from sympy.ntheory import factorint as _factorint


try:
	shapely = True
	from shapely.geometry import box as _rectangle_box
	from shapely.geometry import Polygon as _Polygon
	from shapely.ops import unary_union as _unary_union
except ModuleNotFoundError:
	print("Fallbacking to matplotlib instead of shapely.")
	shapely = False

###########################################################################
#CPP CLASSES
from CppClasses cimport corner_type
from CppClasses cimport corner_list
from CppClasses cimport interval
from CppClasses cimport Summand
from CppClasses cimport Box
from CppClasses cimport Module

###########################################################################
#CYTHON TYPES
from cython cimport numeric
from libcpp.vector cimport vector
from libcpp.utility cimport pair
#from libcpp.list cimport list as clist
from libcpp cimport bool
from libcpp cimport int
from libcpp.string cimport string

###########################################################################
#CYTHON MACROS
ctypedef vector[pair[int,pair[int,int]]] barcode
ctypedef vector[pair[int,pair[double,double]]] barcoded
ctypedef vector[unsigned int] boundary_type
ctypedef vector[boundary_type] boundary_matrix
ctypedef pair[pair[double,double],pair[double,double]] interval_2
ctypedef vector[double] filtration_type
ctypedef vector[Summand] summand_list_type
ctypedef vector[summand_list_type] approx_summand_type
ctypedef vector[filtration_type] image_type

###########################################################################
#CPP TO CYTHON FUNCTIONS
cdef extern from "vineyards_trajectories.h" namespace "Vineyard":
	#vineyard_alt
	vector[vector[vector[interval]]] compute_vineyard_barcode(boundary_matrix &B, vector[vector[double]] &filters_list, double precision, Box &box, bool threshold, bool multithread, bool verbose)
	#vineyard_alt_dim
	vector[vector[interval]] compute_vineyard_barcode_in_dimension(boundary_matrix &B, vector[vector[double]] &filters_list, double precision, Box &box, unsigned int dimension, bool threshold, bool verbose)

cdef extern from "approximation.h" namespace "Vineyard":
	# Approximation
	Module compute_vineyard_barcode_approximation(boundary_matrix &B, vector[vector[double]] &filters_list, double precision, Box &box, bool threshold, bool complete, bool multithread, bool verbose)

cdef extern from "images.h":
	#image_2d_from_boundary_matrix
	vector[vector[vector[double]]] get_2D_image_from_boundary_matrix(boundary_matrix &B, vector[vector[double]] &filters_list, double precision, Box &box, const double bandwidth, const vector[unsigned int] &resolution, const int dimension, bool complete, bool verbose)


cdef extern from "format_python-cpp.h":
	#list_simplicies_to_sparse_boundary_matrix
	vector[vector[unsigned int]] build_sparse_boundary_matrix_from_simplex_list(vector[vector[unsigned int]] list_simplices)
	#list_simplices_ls_filtration_to_sparse_boundary_filtration
	pair[vector[vector[unsigned int]], vector[vector[double]]] build_boundary_matrix_from_simplex_list(vector[vector[unsigned int]] list_simplices, vector[vector[double]] filtrations, vector[unsigned int] filters_to_permute)
	pair[vector[vector[unsigned int]], vector[double]] simplextree_to_boundary_filtration(vector[boundary_type] &simplexList, filtration_type &filtration)
cdef extern from "benchmarks.h":
	#time_vineyard_alt
	double time_vineyard_barcode_computation(boundary_matrix B, vector[vector[double]] filters_list, double precision, Box box, bool threshold, bool multithread, bool verbose)

	#time_approximation
	double time_approximated_vineyard_barcode_computation(boundary_matrix B, vector[vector[double]] filters_list, double precision, Box box, bool threshold, bool complete, bool multithread, bool verbose)

	double time_2D_image_from_boundary_matrix_construction(boundary_matrix &B, vector[vector[double]] &filters_list, double precision, Box &box, const double bandwidth, const vector[unsigned int] &resolution, const unsigned int dimension, bool complete, bool verbose)

###########################################################################
# CYTHON CLASSES
cdef class PySummand:
	cdef Summand sum 

	# def __cinit__(self, vector[corner_type]& births, vector[corner_type]& deaths, int dim):
	# 	self.sum = Summand(births, deaths, dim)

	def get_birth_list(self):
		return self.sum.get_birth_list()

	def get_death_list(self):
		return self.sum.get_death_list()

	def get_dimension(self):
		return self.sum.get_dimension()
	
	cdef set(self, Summand summand):
		self.sum = summand

# cdef class PyBox:
# 	cdef Box box 

# 	def __cinit__(self, corner_type bottomCorner, corner_type topCorner):
# 		self.box = Box(bottomCorner, topCorner)

cdef class PyModule:
	cdef Module cmod
	def __cinit__(self):
		self.cmod = Module()

	cdef set(self, Module m):
		self.cmod = m
	cdef set_box(self, Box box):
		self.cmod.set_box(box)
	def get_module_of_dimension(self, int dim):
		pmodule = PyModule()
		pmodule.set_box(self.cmod.get_box())
		for summand in self.cmod:
			if summand.get_dimension() == dim:
				pmodule.cmod.add_summand(summand)
		return pmodule
		# summand_list = []
		# for summand in self.cmod:
		# 	if summand.get_dimension() == dim:
		# 		tmp_summand = PySummand()
		# 		tmp_summand.set(summand)
		# 		summand_list.append(tmp_summand)
		# return summand_list
	def size(self):
		return self.cmod.size()
	def get_bottom(self):
		return self.cmod.get_box().get_bottom_corner()
	def get_top(self):
		return self.cmod.get_box().get_upper_corner()
	def len(self):
		return len(self.mod)

	def __getitem__(self, unsigned int i):
		summand = PySummand()
		summand.set(self.cmod.at(i))
		return summand

	
	def plot(self, int dimension=-1,**kwargs):
	# *,box = None, separated=False, alpha=1, save=False, xlabel=None, ylabel=None,min_interleaving = 0,complete=True):
		if (kwargs.get('box')):
			box = kwargs.pop('box')
		else:
			box = [self.get_bottom(), self.get_top()]
		if (len(box[0]) != 2):
			print("Filtration size :", len(box[0]), " != 2")
			return
		num = 0
		if(dimension < 0):
			for dimension in range(self.cmod.get_dimension()+1):
					self.plot(dimension,box=box,**kwargs)
			return
		corners = self.cmod.get_corners_of_dimension(dimension)
		plot2d(corners, box=box, dimension=dimension, **kwargs)

	def image(self, resolution:List=[100,100], bandwidth=1, dimension = -1, plot=True, save=False, dpi=200, **kwargs):
		if (len(self.get_bottom()) != 2):
			print("Non 2 dimensional images not yet implemented in python !")
			return
		if dimension < 0:
			image_vector = self.cmod.get_vectorization(bandwidth, resolution[0], resolution[1])
		else:
			image_vector = [self.cmod.get_vectorization_in_dimension(dimension, bandwidth, resolution[0], resolution[1])]
		if kwargs.get('box'):
			kwargs.pop('box')
			print("box not yet implemented :(")

		box = [self.get_bottom(), self.get_top()]
		if plot:
			i=0
			for image in image_vector:	
				_plt.imshow(_np.flip(_np.array(image).transpose(),0),extent=[box[0][0], box[1][0], box[0][1], box[1][1]], aspect = (box[1][0]-box[0][0]) / (box[1][1]-box[0][1]))
				if (kwargs.get('colorbar') or kwargs.get('cb')):
					_plt.colorbar()
				if save:
					_plt.savefig(save+"_H"+str(i)+".png", dpi=dpi)
				i+=1
				if dimension < 0 :
					_plt.title(f"H_{i} 2-persistence image")
					_plt.show(**kwargs)
			if dimension >= 0:
				_plt.title(f"H_{dimension} 2-persistence image")
				_plt.plot(**kwargs)
		return image_vector



###########################################################################
# PYTHON FUNCTIONS USING CYTHON

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
	if (type(B) == _SimplexTree):
		time = time_2D_image_from_boundary_matrix_construction(simplextree_to_sparse_boundary(B), filtration, precision, Box(box), bandwidth, resolution, dimension, complete, verbose)
	else:
		time = time_2D_image_from_boundary_matrix_construction(B, filtration, precision, Box(box), bandwidth, resolution, dimension, complete, verbose)

	return time

def persistence_image_2d(B, filters, precision=-1, box=[], bandwidth=-1, resolution=[100,100], dimension=-1, complete=True, verbose = False, plot = True, save=False, dpi=200):
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
	if (type(B) == _SimplexTree):
		image_vector = _np.array(get_2D_image_from_boundary_matrix(simplextree_to_sparse_boundary(B), filtration, precision, Box(box), bandwidth, resolution, dimension, complete, verbose))
	else:
		image_vector = _np.array(get_2D_image_from_boundary_matrix(B, filtration, precision, Box(box), bandwidth, resolution, dimension, complete, verbose))
	## Fixes images
	i=0
	if(plot or save):
		for image in image_vector:
			_plt.imshow(_np.flip(image.transpose(),0),extent=[box[0][0], box[1][0], box[0][1], box[1][1]], aspect = (box[1][0]-box[0][0]) / (box[1][1]-box[0][1]))
			_plt.colorbar()
			if save:
				_plt.savefig(save+"_H"+str(i)+".png", dpi=dpi)
				i+=1
			if plot:
				_plt.show()
			else:
				_plt.close()
	if dimension<0:
		return image_vector
	return image_vector[0]


def time_approx(B, filters, precision, box=[], threshold=False,complete=False,multithread=False, verbose=False):
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
	if (type(B) == _SimplexTree):
		return time_approximated_vineyard_barcode_computation(simplextree_to_sparse_boundary(B), filtration, precision, Box(box), threshold, complete, multithread, verbose)
	return time_approximated_vineyard_barcode_computation(B,filtration,precision, Box(box), threshold, complete, multithread, verbose)

def approx(B, filters, precision:float=0.1, box = [[],[]], threshold=True, complete=True, multithread = False, verbose = False)->PyModule:
	if(type(filters) == _np.ndarray):
		#assert filters.shape[1] == 2
		filtration = [filters[:,i] for i in range(filters.shape[1])]
	else:
		filtration = filters

	if type(B) == _SimplexTree:
		if verbose:
			print("Converting simplextree to boundary matrix...")
		boundary = simplextree_to_sparse_boundary(B)
	else:
		boundary = B
	approx_mod = PyModule()
	approx_mod.set(compute_vineyard_barcode_approximation(boundary,filtration,precision, Box(box), threshold, complete, multithread,verbose))
	return approx_mod



def ls_boundary_density(list_simplices, points_filtration, to_permute = []):
	
	if (type(list_simplices) == _SimplexTree):
		boundary, ls_filter = build_boundary_matrix_from_simplex_list(
			[simplex[0] for simplex in list_simplices.get_simplices()],points_filtration,
			to_permute)
	else:
		boundary, ls_filter = build_boundary_matrix_from_simplex_list(list_simplices, points_filtration, to_permute)
	return boundary, _np.array(ls_filter).transpose()

def splx2bf(simplextree):
	if (type(simplextree) != _SimplexTree):
		print(simplextree, "has to be a simplextree.")
	boundaries = [s for s,f in simplextree.get_simplices()]
	filtration = [f for s,f in simplextree.get_simplices()]
	return simplextree_to_boundary_filtration(boundaries,filtration)
	

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
	if (type(B) == _SimplexTree):
		return time_vineyard_barcode_computation(simplextree_to_sparse_boundary(B), filtration, precision, Box(box), threshold, multithread, verbose)
	return time_vineyard_barcode_computation(B,filtration,precision, Box(box), threshold, multithread, verbose)

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
		if (type(B) == _SimplexTree):
			return compute_vineyard_barcode(simplextree_to_sparse_boundary(B), filtration, precision, Box(box), threshold, multithread, verbose)
		return compute_vineyard_barcode(B,filtration,precision, Box(box), threshold, multithread, verbose)
	if (type(B) == _SimplexTree):
		return compute_vineyard_barcode_in_dimension(simplextree_to_sparse_boundary(B), filtration, precision, Box(box), dimension, threshold, verbose)
	return compute_vineyard_barcode_in_dimension(B,filtration,precision, Box(box), dimension, threshold, verbose)

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

# def plot2d(B, filters, precision=0.1, box = [], dimension=-1, return_corners=False, separated=False, min_interleaving = 0, multithread = False, complete=True, alpha=1, verbose = False, save=False, dpi=200, shapely = True):
# 	if alpha >= 1:
# 		shapely = False # Not sure which one is quicker in that case.
# 	if box == [] and (type(filters) == _np.ndarray):
# 		box = [[min(filters[:,0]),min(filters[:,1])],[max(filters[:,0]),max(filters[:,1])]]
# 	if box == [] and (type(filters) == list):
# 		box = [[min(filters[0]), min(filters[1])],[max(filters[0]), max(filters[1])]]

# 	module = approx(B,filters,precision,box=box,threshold=1, multithread = multithread, complete = complete, verbose = verbose)
# 	if dimension < 0:
# 		maxDim = module[-1].get_dimension()
# 		for i in range(maxDim + 1):
# 			plot_approx(module.get_module_of_dimension(i), precision=precision, box=box, return_corners=return_corners, separated = separated, min_interleaving=min_interleaving, alpha=alpha, dpi=dpi, shapely=shapely, save = save)
# 	else:
# 		plot_approx(module.get_module_of_dimension(dimension), precision=precision, box=box, return_corners=return_corners, separated = separated, min_interleaving=min_interleaving, alpha=alpha, dpi=dpi, shapely=shapely, save = save)



def plot2d(corners, box = [],*,dimension=-1, separated=False, min_interleaving = 0, alpha=1, verbose = False, save=False, dpi=200, shapely = True, xlabel=None, ylabel=None, **kwargs):
	assert(len(box) == 0, "You have to provide the box") # TODO : retrieve the box from the module.
	if (kwargs.get('cmap')):
		cmap = _get_cmap(kwargs.pop('cmap'))
	else:
		cmap = _get_cmap("Spectral")
	if not(separated):
		fig, ax = _plt.subplots()
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
				summand_shape = _unary_union(list_of_rect)
				if type(summand_shape) == _Polygon:
					xs,ys=summand_shape.exterior.xy
					ax.fill(xs,ys,alpha=alpha, fc=cmap(i/n_summands), ec='None')
				else:
					for polygon in summand_shape.geoms:
						xs,ys=polygon.exterior.xy
						ax.fill(xs,ys,alpha=alpha, fc=cmap(i/n_summands), ec='None')
			else:
				for rectangle in list_of_rect:
					ax.add_patch(rectangle)
			if separated:
				if xlabel:
					_plt.xlabel(xlabel)
				if ylabel:
					_plt.ylabel(ylabel)
				if dimension>=0:
					_plt.title(f"H_{dimension} 2-persistence")
				_plt.show()
	if save:
		_plt.savefig(save, dpi=dpi)
	if not(separated):
		if xlabel != None:
			_plt.xlabel(xlabel)
		if ylabel != None:
			_plt.ylabel(ylabel)
		if dimension>=0:
			_plt.title(f"H_{dimension} 2-persistence")
		_plt.show()
	for kw in kwargs:
		print(kw, "argument non implemented, ignoring.")
	return fig, ax

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
	if (type(B) == _SimplexTree):
		time = time_2D_image_from_boundary_matrix_construction(simplextree_to_sparse_boundary(B), filtration, precision, Box(box), bandwidth, resolution, dimension, complete, verbose)
	else:
		time = time_2D_image_from_boundary_matrix_construction(B, filtration, precision, Box(box), bandwidth, resolution, dimension, complete, verbose)

	return time

def time_approx(B, filters, precision, box=[], threshold=False,complete=False,multithread=False, verbose=False):
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
	if (type(B) == _SimplexTree):
		return time_approximated_vineyard_barcode_computation(simplextree_to_sparse_boundary(B), filtration, precision, Box(box), threshold, complete, multithread, verbose)
	return time_approximated_vineyard_barcode_computation(B,filtration,precision, Box(box), threshold, complete, multithread, verbose)


#######################################################################
# USEFULL PYTHON FUNCTIONS
def convert_to_rivet(simplextree, kde, X,*, dimension=1, verbose = True):
	if _exists("rivet_dataset.txt"):
		_remove("rivet_dataset.txt")
	file = open("rivet_dataset.txt", "a")
	file.write("--datatype bifiltration\n")
	file.write(f"--homology {dimension}\n")
	file.write("--xlabel time of appearance\n")
	file.write("--ylabel density\n\n")

	to_write = ""
	if verbose:
		for s,f in _tqdm(simplextree.get_simplices()):
			for i in s:
				to_write += str(i) + " "
			to_write += "; "+ str(f) + " " + str(_np.max(-kde.score_samples(X[s,:])))+'\n'
	else:
		for s,f in simplextree.get_simplices():
			for i in s:
				to_write += str(i) + " "
			to_write += "; "+ str(f) + " " + str(_np.max(-kde.score_samples(X[s,:])))+'\n'
	file.write(to_write)
	file.close()

def noisy_annulus(r1=1, r2=2, n=50, dim=2, center=None):
	set =[]
	while len(set)<n:
		draw=_np.random.uniform(low=-r2, high=r2, size=dim)
		if _np.linalg.norm(draw) > r1 and _np.linalg.norm(draw) < r2:
			set.append(draw)
	dataset = _np.array(set) if center == None else _np.array(set) + _np.array(center)
	return dataset


def nlines_precision_box(nlines, basepoint, scale, square = False):
	import math
	from random import choice, shuffle
	h = scale
	dim = len(basepoint)
	basepoint = _np.array(basepoint, 'double')
	if square:
		# here we want n^dim-1 lines (n = nlines)
		n=nlines
		basepoint = _np.array(basepoint, 'double')
		deathpoint = basepoint.copy()
		deathpoint+=n*h + - h/2
		deathpoint[-1] = basepoint[-1]+h/2
		return [basepoint,deathpoint]
	factors = _factorint(nlines)
	prime_list=[]
	for prime in factors:
		for i in range(factors[prime]):
			prime_list.append(prime)
	while len(prime_list)<dim-1:
		prime_list.append(1)
	shuffle(prime_list)
	while len(prime_list)>dim-1:
		prime_list[choice(range(dim-1))] *= prime_list.pop()
	deathpoint = basepoint.copy()
	for i in range(dim-1):
		deathpoint[i] = basepoint[i] + prime_list[i] * scale - scale/2
	return [basepoint,deathpoint]
