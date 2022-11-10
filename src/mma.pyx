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
from matplotlib.patches import Rectangle as _Rectangle
from cycler import cycler
import pickle as _pk
from filtration_domination import remove_strongly_filtration_dominated as _remove_strongly_filtration_dominated
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
from CppClasses cimport MultiDiagram_point
from CppClasses cimport Line
from CppClasses cimport Summand
from CppClasses cimport Box
from CppClasses cimport Module
from CppClasses cimport MultiDiagram
from CppClasses cimport MultiDiagrams
from cython.operator import dereference, preincrement
from libc.stdint cimport intptr_t
from libc.stdint cimport uintptr_t


# from simplex_tree_multi cimport Simplex_tree_multi_interface
# from simplex_tree_multi cimport Simplex_tree_multi_simplices_iterator
# from simplex_tree_multi cimport Simplex_tree_multi_simplex_handle
# from simplex_tree_multi cimport Simplex_tree_multi_skeleton_iterator
# from simplex_tree_multi cimport Simplex_tree_multi_boundary_iterator


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
ctypedef vector[filtration_type] multifiltration
ctypedef vector[Summand] summand_list_type
ctypedef vector[summand_list_type] approx_summand_type
ctypedef vector[filtration_type] image_type
ctypedef vector[int] simplex_type
ctypedef int dimension_type

###########################################################################
#PYX MODULES
include "simplex_tree_multi.pyx"


###########################################################################
#CPP TO CYTHON FUNCTIONS

cdef extern from "approximation.h" namespace "Vineyard":
	# Approximation
	Module compute_vineyard_barcode_approximation(boundary_matrix &B, vector[vector[double]] &filters_list, double precision, Box &box, bool threshold, bool complete, bool multithread, bool verbose)


cdef extern from "format_python-cpp.h":
	#list_simplicies_to_sparse_boundary_matrix
	vector[vector[unsigned int]] build_sparse_boundary_matrix_from_simplex_list(vector[vector[unsigned int]] list_simplices)
	#list_simplices_ls_filtration_to_sparse_boundary_filtration
	pair[vector[vector[unsigned int]], vector[vector[double]]] build_boundary_matrix_from_simplex_list(vector[vector[unsigned int]] list_simplices, vector[vector[double]] filtrations, vector[unsigned int] filters_to_permute)
	# pair[vector[vector[unsigned int]], vector[double]] simplextree_to_boundary_filtration(vector[boundary_type] &simplexList, filtration_type &filtration)
	pair[boundary_matrix, multifiltration] simplextree_to_boundary_filtration(uintptr_t)

	pair[vector[vector[unsigned int]], vector[double]] __old__simplextree_to_boundary_filtration(vector[boundary_type]&, filtration_type&)

	string __old__simplextree2rivet(const uintptr_t, const vector[filtration_type]&)
	void simplextree2rivet(const string&, const uintptr_t, const vector[filtration_type]&)



###########################################################################
# CYTHON CLASSES

cdef class PySummand:
	cdef Summand sum 
	# def __cinit__(self, vector[corner_type]& births, vector[corner_type]& deaths, int dim):
	# 	self.sum = Summand(births, deaths, dim)

	def get_birth_list(self)->list:
		return self.sum.get_birth_list()

	def get_death_list(self)->list:
		return self.sum.get_death_list()

	def get_dimension(self)->int:
		return self.sum.get_dimension()
	
	cdef set(self, Summand& summand):
		self.sum = summand
		return self

cdef class PyBox:
	cdef Box box
	def __cinit__(self, corner_type bottomCorner, corner_type topCorner):
		self.box = Box(bottomCorner, topCorner)
	def get_dimension(self):
		dim = self.box.get_bottom_corner().size()
		if dim == self.box.get_upper_corner().size():	return dim
		else:	print("Bad box definition.")
	def contains(self, x):
		return self.box.contains(x)
	cdef set(self, Box& b):
		self.box = b
		return self

	def get(self):
		return [self.box.get_bottom_corner(),self.box.get_upper_corner()]
	def to_multipers(self):
		#assert (self.get_dimension() == 2) "Multipers only works in dimension  2 !"
		return _np.array(self.get()).flatten(order = 'F')

cdef class PyMultiDiagramPoint:
	cdef MultiDiagram_point point
	cdef set(self, MultiDiagram_point pt):
		self.point = pt
		return self

	def get_dimension(self):
		return self.get_dimension()
	def get_birth(self):
		return self.get_birth()
	def get_death(self):
		return self.get_death()

cdef class PyMultiDiagram:
	cdef MultiDiagram multiDiagram
	cdef set(self, MultiDiagram m):
		self.multiDiagram = m
		return self
	def get_points(self, dimension:int=-1):
		return self.multiDiagram.get_points(dimension)
	def to_multipers(self, dimension:int):
		return self.multiDiagram.to_multipers(dimension)
	def __len__(self):
		return self.multiDiagram.size()
	def __getitem__(self,i:int):
		if  0 <= i < self.multiDiagram.size()  :
			return PyMultiDiagramPoint().set(self.multiDiagram.at(i))
		elif -self.multiDiagram.size() < i < 0:
			return PyMultiDiagramPoint().set(self.multiDiagram.at( self.multiDiagram.size() - i))
		else:
			print("Bad index.")
cdef class PyMultiDiagrams:
	cdef MultiDiagrams multiDiagrams
	cdef set(self,MultiDiagrams m):
		self.multiDiagrams = m
		return self
	def to_multipers(self):
		out = self.multiDiagrams.to_multipers()
		# return out
		return [_np.array(summand, dtype=_np.float64) for summand in out]
	def __getitem__(self,i:int):
		if i >=0 :
			return PyMultiDiagram().set(self.multiDiagrams.at(i))
		else:
			return PyMultiDiagram().set(self.multiDiagrams.at( self.multiDiagrams.size() - i))
	def __len__(self):
		return self.multiDiagrams.size()
	def get_points(self, dimension:int=-1):
		return _np.array([x.get_points(dimension) for x in self.multiDiagrams], dtype=float)
		# return _np.array([PyMultiDiagram().set(x).get_points(dimension) for x in self.multiDiagrams])
	cdef _get_plot_bars(self, dimension:int=-1, min_persistence:float=0):
		return self.multiDiagrams._for_python_plot(dimension, min_persistence);
	def plot(self, dimension:int=-1, min_persistence:float=0):
		"""
		Plots the barcodes.

		Parameters
		----------
		dimension:int=-1
			Only plots the bars of specified dimension. Useful when the multidiagrams contains multiple dimenions
		min_persistence:float=0
			Only plot bars of length greater than this value. Useful to reduce the time to plot.

		Warning
		-------
		If the barcodes are not thresholded, essential barcodes will not be displayed !

		Returns
		-------
		Nothing
		"""
		if len(self) == 0: return
		_cmap = _get_cmap("Spectral")
		multibarcodes_, colors = self._get_plot_bars(dimension, min_persistence)
		n_summands = _np.max(colors)+1
		_plt.rc('axes', prop_cycle = cycler('color', [_cmap(i/n_summands) for i in colors]))
		_plt.plot(*multibarcodes_)
cdef class PyLine:
	cdef Line line

cdef class PyModule:
	cdef Module cmod

	cdef set(self, Module m):
		self.cmod = m
	cdef set_box(self, Box box):
		self.cmod.set_box(box)
		return self
	def get_module_of_dimension(self, dim:int)->PyModule: # TODO : in c++ ?
		pmodule = PyModule()
		pmodule.set_box(self.cmod.get_box())
		for summand in self.cmod:
			if summand.get_dimension() == dim:
				pmodule.cmod.add_summand(summand)
		return pmodule

	def __len__(self)->int:
		return self.cmod.size()
	def get_bottom(self)->list:
		return self.cmod.get_box().get_bottom_corner()
	def get_top(self)->list:
		return self.cmod.get_box().get_upper_corner()
	def get_box(self):
		return [self.get_bottom(), self.get_top()]
	def get_dimension(self)->int:
		return self.cmod.get_dimension()
	def dump(self, path:str=None):
		"""
		Dumps the module into a pickle-able format.
		
		Parameters
		----------
		path:str=None (optional) saves the pickled module in specified path
		
		Returns
		-------
		list of list, encoding the module, which can be retrieved with the function `from_dump`.
		"""
		out = [[] for _ in range(self.cmod.get_dimension()+1)]
		out += [self.get_box()]
		for summand in self.cmod:
			out[summand.get_dimension()].append([summand.get_birth_list(), summand.get_death_list()])
		if path is None:
			return out
		_pk.dump(out, open(path, "wb"))
		return out
	def __getitem__(self, i:int) -> PySummand:
		summand = PySummand()
		if i>=0:
			summand.set(self.cmod.at(i))
		else:
			summand.set(self.cmod.at(self.size() - i))
		return summand
	
	def plot(self, dimension:int=-1,**kwargs)->None:
		"""Shows the module on a plot. Each color corresponds to an apprimation summand of the module, and its shape corresponds to its support.
		Only works with 2-parameter modules.

		Parameters
		----------
		dimension = -1 : integer
			If positive returns only the image of dimension `dimension`.
		box=None : of the form [[b_x,b_y], [d_x,d_y]] where b,d are the bottom and top corner of the rectangle.
			If non-None, will plot the module on this specific rectangle.
		min_persistence =0 : float
			Only plots the summand with a persistence above this threshold.
		separated=False : bool
			If true, plot each summand in a different plot.
		alpha=1 : float
			Transparancy parameter
		save = False : string
			if nontrivial, will save the figure at this path


		Returns
		-------
		The figure of the plot.
		"""
		if (kwargs.get('box')):
			box = kwargs.pop('box')
		else:
			box = [self.get_bottom(), self.get_top()]
		if (len(box[0]) != 2):
			print("Filtration size :", len(box[0]), " != 2")
			return
		num = 0
		if(dimension < 0):
			ndim = self.cmod.get_dimension()+1
			scale = kwargs.pop("scale", 4)
			fig, axes = _plt.subplots(1, ndim, figsize=(ndim*scale,scale))
			for dimension in range(ndim):
				_plt.sca(axes[dimension]) if ndim > 1 else  _plt.sca(axes)
				self.plot(dimension,box=box,**kwargs)
			return
		corners = self.cmod.get_corners_of_dimension(dimension)
		plot2d(corners, box=box, dimension=dimension, **kwargs)
		return

	def barcode(self, basepoint, dimension:int=-1,*, threshold = False): # TODO direction vector interface
		"""Computes the barcode of module along a lines.

		Parameters
		----------
		basepoint  : vector
			basepoint of the lines on which to compute the barcodes, i.e. a point on the line
		dimension = -1 : integer
			Homology dimension on which to compute the bars. If negative, every dimension is computed
		box (default) :
			box on which to compute the barcodes if basepoints is not specified. Default is a linspace of lines crossing that box.
		threshold = False : threshold t
			Resolution of the image(s).

		Warning
		-------
		If the barcodes are not thresholded, essential barcodes will not be plot-able.

		Returns
		-------
		PyMultiDiagrams
			Structure that holds the barcodes. Barcodes can be retrieved with a .get_points() or a .to_multipers() or a .plot().
		"""
		out = PyMultiDiagram()
		out.set(self.cmod.get_barcode(Line(basepoint), dimension, threshold))
		return out
	def barcodes(self, basepoints = None, dimension:int=-1, *,threshold = False, **kwargs):
		"""Computes barcodes of module along a set of lines.

		Parameters
		----------
		basepoints = None : list of vectors
			basepoints of the lines on which to compute the barcodes.
		dimension = -1 : integer
			Homology dimension on which to compute the bars. If negative, every dimension is computed
		box (default) :
			box on which to compute the barcodes if basepoints is not specified. Default is a linspace of lines crossing that box.
		num:int=100
			if basepoints is not specified, defines the number of lines to consider.
		threshold = False : threshold t
			Resolution of the image(s).

		Warning
		-------
		If the barcodes are not thresholded, essential barcodes will not be plot-able.

		Returns
		-------
		PyMultiDiagrams
			Structure that holds the barcodes. Barcodes can be retrieved with a .get_points() or a .to_multipers() or a .plot().
		"""
		out = PyMultiDiagrams()
		if (kwargs.get('box')):
			box = kwargs.pop('box')
		else:
			box = [self.get_bottom(), self.get_top()]
		if (len(box[0]) != 2) and (basepoints == None):
			print("Filtration size :", len(box[0]), " != 2")
			print("Basepoints has to be specified for filtration dimension >= 3 !")
			return
		elif basepoints is None:
			basepoints = _np.linspace([box[0][0] - box[1][1],0], [box[1][0],0], num=kwargs.get("num", 100))
		out.set(self.cmod.get_barcodes(basepoints, dimension, threshold))
		return out
	def __old__multipers_landscape(self, dimension:int,num:int=100,  **kwargs):
		"""
		Computes the Multi parameter landscape, using the multipers library.
		"""
		box = kwargs.pop('box',self.get_box())
		bnds = _np.array(box).flatten(order = 'F')
		
		#Defines on which lines to compute the barcodes
		if (len(box[0]) != 2):
			print("Filtration size :", len(box[0]), " != 2")
			print("Not implemented in multipers !")
			return

		first = [box[0][0] - box[1][1],0]
		last = [box[1][0],0]
		basepoints = kwargs.get("basepoints", _np.linspace(first, last, num=num))
		# Computes barcodes from PyModule in the multipers format
		decomposition = self.barcodes(basepoints, dimension = dimension, threshold = True).to_multipers()
		delta = (last[0] - first[0]) / num
		from multipers import multipersistence_landscape
		# Computes multipers Landscapes
		return multipersistence_landscape(decomposition, bnds, delta,**kwargs)
	def landscape(self, dimension:int, k:int=0,box=None, resolution:List[int]=[100,100], plot=True):
		"""Computes the multiparameter landscape from a PyModule. Python interface only bifiltrations.

		Parameters
		----------
		dimension : integer
			The homology dimension of the landscape.
		k = 0 : int
			the k-th landscape
		resolution = [50,50] : pair of integers
			Resolution of the image.
		box = None : in the format [[a,b], [c,d]]
			If nontrivial, compute the landscape of this box. Default is the PyModule box.
		plot = True : Boolean
			If true, plots the images;
		Returns
		-------
		Numpy array
			The landscape of the module.
		"""
		if box is None:
			cbox = self.cmod.get_box()
			pbox = self.get_box()
		else:
			cbox = PyBox(*box).box
			pbox = box

		out = _np.array(self.cmod.get_landscape(dimension, k, cbox, resolution))
		if plot:
			_plt.figure()
			aspect = (pbox[1][0]-pbox[0][0]) / (pbox[1][1]-pbox[0][1])
			extent = [pbox[0][0], pbox[1][0], pbox[0][1], pbox[1][1]]
			_plt.imshow(out.T, origin="lower", extent=extent, aspect=aspect)
		return out
	def landscapes(self, dimension:int, ks:List[int]=[0],box=None, resolution:List[int]=[100,100], plot=True):
		"""Computes the multiparameter landscape from a PyModule. Python interface only bifiltrations.

		Parameters
		----------
		dimension : integer
			The homology dimension of the landscape.
		ks = 0 : list of int
			the k-th landscape
		resolution = [50,50] : pair of integers
			Resolution of the image.
		box = None : in the format [[a,b], [c,d]]
			If nontrivial, compute the landscape of this box. Default is the PyModule box.
		plot = True : Boolean
			If true, plots the images;
		Returns
		-------
		Numpy array
			The landscapes of the module with parameters ks.
		"""
		if box is None:
			cbox = self.cmod.get_box()
			pbox = self.get_box()
		else:
			cbox = PyBox(*box).box
			pbox = box
		out = _np.array(self.cmod.get_landscapes(dimension, ks, cbox, resolution))
		if plot:
			to_plot = _np.sum(out, axis=0)
			_plt.figure()
			aspect = (pbox[1][0]-pbox[0][0]) / (pbox[1][1]-pbox[0][1])
			extent = [pbox[0][0], pbox[1][0], pbox[0][1], pbox[1][1]]
			_plt.imshow(to_plot.T, origin="lower", extent=extent, aspect=aspect)
		return out


	def image(self, dimension:int = -1, bandwidth:float=0.1, resolution:list=[100,100], normalize:bool=True, plot:bool=True, save:bool=False, dpi:int=200,p:float=1., **kwargs)->_np.ndarray:
		"""Computes a vectorization from a PyModule. Python interface only bifiltrations.

		Parameters
		----------
		dimension = -1 : integer
			If positive returns only the image of dimension `dimension`.
		bandwidth = 0.1 : float
			Image parameter. TODO : different weights
		resolution = [100,100] : pair of integers
			Resolution of the image(s).
		normalize = True : Boolean
			Ensures that the image belongs to [0,1].
		plot = True : Boolean
			If true, plots the images;

		Returns
		-------
		List of Numpy arrays or numpy array
			The list of images, or the image of fixed dimension.
		"""
		if (len(self.get_bottom()) != 2):
			print("Non 2 dimensional images not yet implemented in python !")
			return
		box = kwargs.get("box",[self.get_bottom(),self.get_top()])
		if dimension < 0:
			image_vector = _np.array(self.cmod.get_vectorization(bandwidth, p, normalize, Box(box), resolution[0], resolution[1]))
		else:
			image_vector = _np.array([self.cmod.get_vectorization_in_dimension(dimension, bandwidth, p,normalize,Box(box),  resolution[0], resolution[1])])
		if plot:
			i=0
			n_plots = len(image_vector)
			scale = 4 if not(kwargs.get("size")) else kwargs.get("size")
			fig, axs = _plt.subplots(1,n_plots, figsize=(n_plots*scale,scale))
			aspect = (box[1][0]-box[0][0]) / (box[1][1]-box[0][1])
			extent = [box[0][0], box[1][0], box[0][1], box[1][1]]
			for image in image_vector:
				ax = axs if n_plots <= 1 else axs[i]
				temp = ax.imshow(_np.flip(_np.array(image).transpose(),0),extent=extent, aspect=aspect)
				if (kwargs.get('colorbar') or kwargs.get('cb')):
					_plt.colorbar(temp, ax = ax)
				if dimension < 0 :
					ax.set_title(f"H_{i} 2-persistence image")
				if dimension >= 0:
					ax.set_title(f"H_{dimension} 2-persistence image")
				i+=1

		return image_vector[0] if dimension >=0 else  image_vector

###########################################################################
# PYTHON FUNCTIONS USING CYTHON
def approx(
	st:type(SimplexTree()),
	precision:float = 0.01,
	box = None,
	threshold:bool = False,
	complete:bool = True,
	multithread:bool = False, 
	verbose:bool = False,
	ignore_warning:bool = False, **kwargs)->PyModule:
	"""Computes an interval module approximation of a multiparameter filtration.

	Parameters
	----------
	B : Simplextree or (sparse) boundary matrix
		Stores the full complex of the filtration.
	filters : list of filtrations
		list of 1-dimensional filtrations that encode the multiparameter filtration.
		Given an index i, filters[i] should be the list of filtration values of 
		the simplices, in lexical order, of the i-th filtration.
	precision: positive float
		Trade-off between approximation and computational complexity.
		Upper bound of the module approximation, in bottleneck distance, 
		for interval-decomposable modules.
	box : pair of list of floats
		Defines a rectangle on which to compute the approximation.
		Format : [x,y], where x,y defines the rectangle {z : x ≤ z ≤ y}
	threshold: bool
		When true, intersects the module support with the box.
	verbose: bool
		Prints C++ infos.
	ignore_warning : bool
		Unless set to true, prevents computing on more than 10k lines. Useful to prevent a segmentation fault due to "infinite" recursion.
	Returns
	-------
	PyModule
		An interval decomposable module approximation of the module defined by the
		homology of this multi-filtration.
	"""
# 	if(type(filters) == _np.ndarray):
# 		#assert filters.shape[1] == 2
# 		filtration = [filters[:,i] for i in range(filters.shape[1])]
# 	else:
# 		filtration = filters
#
#
# 	if type(B) == _SimplexTree:
# 		if verbose:
# 			print("Converting simplextree to boundary matrix...")
# 		boundary,_ = splx2bf(B)
# 	else:
# 		boundary = B
	boundary,filtration = splx2bf(st)
	if box is None:
		M = [_np.max(f)+2*precision for f in filtration]
		m = [_np.min(f)-2*precision for f in filtration]
		box = [m,M]
	else:
		m, M = box
	if not ignore_warning:
		prod = 1
		h = M[-1] - m[-1]
		for i, [a,b] in enumerate(zip(m,M)):
			if i == len(M)-1:	continue
			prod *= (b-a + h) / precision
		if prod >= 10_000:
			from warnings import warn
			warn(f"Warning : the number of lines (around {_np.round(prod)}) may be too high. Try to increase the precision parameter, or set `ignore_warning=True` to compute this module. Returning the trivial module.")
			return PyModule()
	approx_mod = PyModule()
	approx_mod.set(compute_vineyard_barcode_approximation(boundary,filtration,precision, Box(box), threshold, complete, multithread,verbose))
	return approx_mod

def from_dump(dump)->PyModule: #Assumes that the input format is the same as the output of a PyModule dump.
	"""Retrieves a PyModule from a previous dump.

	Parameters
	----------
	dump: either the output of the dump function, or a file containing the output of a dump.
		The dumped module to retrieve

	Returns
	-------
	PyModule
		The retrieved module.
	"""
	mod = PyModule()
	if type(dump) is str:
		dump = _pk.load(open(dump, "rb"))
	mod.cmod.set_box(Box(dump[-1]))
	for dim,summands in enumerate(dump[:-1]):
		for summand in summands:
			mod.cmod.add_summand(Summand(summand[0], summand[1], dim))

	return mod




def splx2bf_old(simplextree:_SimplexTree):
	boundaries = [s for s,f in simplextree.get_simplices()]
	filtration = [f for s,f in simplextree.get_simplices()]
	return __old__simplextree_to_boundary_filtration(boundaries,filtration)
	

def splx2bf(simplextree:Union[type(SimplexTree()), type(_gd.SimplexTree())]):
	"""Computes a (sparse) boundary matrix, with associated filtration. Can be used as an input of approx afterwards.
	
	Parameters
	----------
	simplextree: Gudhi or mma simplextree
		The simplextree defining the filtration to convert to boundary-filtration.
	
	Returns
	-------
	B:List of lists of ints
		The boundary matrix.
	F: List of 1D filtration
		The filtrations aligned with B; the i-th simplex of this simplextree has boundary B[i] and filtration(s) F[i].
	
	"""
	if type(simplextree) == type(SimplexTree()):
		return simplextree_to_boundary_filtration(simplextree.thisptr)
	else:
		temp_st = from_gudhi(simplextree, dimension=1)
		b,f=simplextree_to_boundary_filtration(temp_st.thisptr)
		temp_st
		return b, f[0]


def simplextree_to_sparse_boundary(st:_SimplexTree):
	return build_sparse_boundary_matrix_from_simplex_list([simplex[0] for simplex in st.get_simplices()])



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

	

def plot2d(corners, box = [],*,dimension=-1, separated=False, min_persistence = 0, alpha=1, verbose = False, save=False, dpi=200, shapely = True, xlabel=None, ylabel=None, **kwargs):
	cmap = _get_cmap(kwargs.pop('cmap', "Spectral"))
	if not(separated):
		# fig, ax = _plt.subplots()
		ax = _plt.gca()
		ax.set(xlim=[box[0][0],box[1][0]],ylim=[box[0][1],box[1][1]])
	n_summands = len(corners)
	for i in range(n_summands):
		trivial_summand = True
		list_of_rect = []
		for birth in corners[i][0]:
			for death in corners[i][1]:
				death[0] = min(death[0],box[1][0])
				death[1] = min(death[1],box[1][1])
				if death[1]>birth[1] and death[0]>birth[0]:
					if trivial_summand and _d_inf(birth,death)>min_persistence:
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
	if not(separated):
		if xlabel != None:
			_plt.xlabel(xlabel)
		if ylabel != None:
			_plt.ylabel(ylabel)
		if dimension>=0:
			_plt.title(f"H_{dimension} 2-persistence")
	for kw in kwargs:
		print(kw, "argument non implemented, ignoring.")
	return



#######################################################################
# USEFULL PYTHON FUNCTIONS

def __old__convert_to_rivet(simplextree:_SimplexTree, kde, X,*, dimension=1, verbose = True)->None:
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

def __old__splx2rivet(simplextree:_SimplexTree, F, **kwargs):
	"""Converts an input of approx to a file (rivet_dataset.txt) that can be opened by rivet.

	Parameters
	----------
	simplextree: gudhi.SimplexTree
		A gudhi simplextree defining the chain complex
	bifiltration: pair of filtrations. Same format as the approx function.
		list of 1-dimensional filtrations that encode the multiparameter filtration.
		Given an index i, filters[i] should be the list of filtration values of
		the simplices, in lexical order, of the i-th filtration.

	Returns
	-------
	Nothing.
	The file created is located at <current_working_directory>/rivet_dataset.txt; and can be directly imported to rivet.
	"""
	if(type(F) == _np.ndarray):
		#assert filters.shape[1] == 2
		G = [F[:,i] for i in range(F.shape[1])]
	else:
		G = F
	if _exists("rivet_dataset.txt"):
		_remove("rivet_dataset.txt")
	file = open("rivet_dataset.txt", "a")
	file.write("--datatype bifiltration\n")
	file.write("--xbins " + str(kwargs.get("xbins", 0))+"\n")
	file.write("--ybins " + str(kwargs.get("xbins", 0))+"\n")
	file.write("--xlabel "+ kwargs.get("xlabel", "")+"\n")
	file.write("--ylabel " + kwargs.get("ylabel", "") + "\n\n")

	simplices = __old__simplextree2rivet(simplextree.thisptr, G).decode("UTF-8")
	file.write(simplices)
	#return simplices

def splx2rivet(simplextree:_SimplexTree, F, **kwargs):
	"""Converts an input of approx to a file (rivet_dataset.txt) that can be opened by rivet.

	Parameters
	----------
	simplextree: gudhi.SimplexTree
		A gudhi simplextree defining the chain complex
	bifiltration: pair of filtrations. Same format as the approx function.
		list of 1-dimensional filtrations that encode the multiparameter filtration.
		Given an index i, filters[i] should be the list of filtration values of
		the simplices, in lexical order, of the i-th filtration.

	Returns
	-------
	Nothing.
	The file created is located at <current_working_directory>/rivet_dataset.txt; and can be directly imported to rivet.
	"""
	from os import getcwd
	if(type(F) == _np.ndarray):
		#assert filters.shape[1] == 2
		G = [F[:,i] for i in range(F.shape[1])]
	else:
		G = F
	path = getcwd()+"/rivet_dataset.txt"
	if _exists(path):
		_remove(path)
	file = open(path, "a")
	file.write("--datatype bifiltration\n")
	file.write("--xbins " + str(kwargs.get("xbins", 0))+"\n")
	file.write("--ybins " + str(kwargs.get("xbins", 0))+"\n")
	file.write("--xlabel "+ kwargs.get("xlabel", "")+"\n")
	file.write("--ylabel " + kwargs.get("ylabel", "") + "\n\n")
	file.close()

	simplextree2rivet(path.encode("UTF-8"), simplextree.thisptr, G)
	return


def noisy_annulus(r1:float=1, r2:float=2, n1:int=1000,n2:int=200, dim:int=2, center:Union[_np.ndarray, list]=None, **kwargs)->_np.ndarray:
	"""Generates a noisy annulus dataset.

	Parameters
	----------
	r1 : float.
		Lower radius of the annulus.
	r2 : float.
		Upper radius of the annulus.
	n1 : int
		Number of points in the annulus.
	n2 : int
		Number of points in the square.
	dim : int
		Dimension of the annulus.
	center: list or array
		center of the annulus.

	Returns
	-------
	numpy array
		Dataset. size : (n1+n2) x dim

	"""
	from numpy.random import uniform
	from numpy.linalg import norm

	set =[]
	while len(set)<n1:
		draw=uniform(low=-r2, high=r2, size=dim)
		if norm(draw) > r1 and norm(draw) < r2:
			set.append(draw)
	annulus = _np.array(set) if center == None else _np.array(set) + _np.array(center)
	diffuse_noise = uniform(size=(n2,dim), low=-1.1*r2,high=1.1*r2)
	if center is not None:	diffuse_noise += _np.array(center)
	return _np.vstack([annulus, diffuse_noise])

def test_module(**kwargs):
	"""Generates a module from a noisy annulus.

	Parameters
	----------
	r1 : float.
		Lower radius of the annulus.
	r2 : float.
		Upper radius of the annulus.
	n1 : int
		Number of points in the annulus.
	n2 : int
		Number of points in the square.
	dim : int
		Dimension of the annulus.
	center: list or array
		center of the annulus.

	Returns
	-------
	PyModule
		The associated module.

	"""
	points = noisy_annulus(**kwargs)
	points = _np.unique(points, axis=0)
	from sklearn.neighbors import KernelDensity
	kde = KernelDensity(bandwidth = 1).fit(points)
	ac = _gd.AlphaComplex(points = points)
	st = ac.create_simplex_tree(max_alpha_square=10)
	points = [ac.get_point(i) for i,_ in enumerate(points)]
	b,f1 = splx2bf(st)
	f2 = kde.score_samples(points)
	mod = approx(b, [f1,f2], **kwargs)
	return mod

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


def collapse_2pers(tree:type(SimplexTree()), max_dimension:int=None, num:int=1):
	"""Strong collapse of 1 critical clique complex, compatible with 2-parameter filtration.

	Parameters
	----------
	tree:SimplexTree
		The complex to collapse
	F2: 1-dimensional array, or callable
		Second filtration. The first one being in the simplextree.
	max_dimension:int
		Max simplicial dimension of the complex. Unless specified, 
	num:int
		The number of collapses to do.
	Returns
	-------
	reduced_tree:SimplexTree
		A simplex tree that has the same homology over this bifiltration.

	"""
	if num <= 0:
		return tree
	max_dimension = tree.dimension() if max_dimension is None else max_dimension
	edges = [(tuple(splx), (f1, f2)) for splx,(f1,f2) in tree.get_skeleton(1) if len(splx) == 2]
	edges = _remove_strongly_filtration_dominated(edges)
	reduced_tree = SimplexTree()
	for splx, f in tree.get_skeleton(0):
		reduced_tree.insert(splx, f)
	for e, (f1, f2) in edges:
		reduced_tree.insert(e, [f1,f2])
	reduced_tree.expansion(max_dimension)
	return collapse_2pers(reduced_tree, max_dimension, num-1)


#
# ############################################################
# # SimplexTree
# # from simplex_tree_multi cimport Simplex_tree_multi_interface
# #
# # cdef ntruc():
# #
# # 	return Simplex_tree_multi_interface().num_vertices()
# # def truc():
# # 	return ntruc()
#
# cdef bool callback(vector[int] simplex, void *blocker_func):
# 	return (<object>blocker_func)(simplex)
#
# # SimplexTree python interface
# cdef class SimplexTree:
# 	"""The simplex tree is an efficient and flexible data structure for
# 	representing general (filtered) simplicial complexes. The data structure
# 	is described in Jean-Daniel Boissonnat and Clément Maria. The Simplex
# 	Tree: An Efficient Data Structure for General Simplicial Complexes.
# 	Algorithmica, pages 1–22, 2014.
#
# 	This class is a filtered, with keys, and non contiguous vertices version
# 	of the simplex tree.
# 	"""
# 	# unfortunately 'cdef public Simplex_tree_multi_interface* thisptr' is not possible
# 	# Use intptr_t instead to cast the pointer
# 	cdef public intptr_t thisptr
#
# 	# Get the pointer casted as it should be
# 	cdef Simplex_tree_multi_interface* get_ptr(self) nogil:
# 		return <Simplex_tree_multi_interface*>(self.thisptr)
#
# 	# cdef Simplex_tree_persistence_interface * pcohptr
#
# 	# Fake constructor that does nothing but documenting the constructor
# 	def __init__(self, other = None):
# 		"""SimplexTree constructor.
#
# 		:param other: If `other` is `None` (default value), an empty `SimplexTree` is created.
# 			If `other` is a `SimplexTree`, the `SimplexTree` is constructed from a deep copy of `other`.
# 		:type other: SimplexTree (Optional)
# 		:returns: An empty or a copy simplex tree.
# 		:rtype: SimplexTree
#
# 		:raises TypeError: In case `other` is neither `None`, nor a `SimplexTree`.
# 		:note: If the `SimplexTree` is a copy, the persistence information is not copied. If you need it in the clone,
# 			you have to call :func:`compute_persistence` on it even if you had already computed it in the original.
# 		"""
#
# 	# The real cython constructor
# 	def __cinit__(self, other = None):
# 		if other:
# 			if isinstance(other, SimplexTree):
# 				self.thisptr = _get_copy_intptr(other)
# 			else:
# 				raise TypeError("`other` argument requires to be of type `SimplexTree`, or `None`.")
# 		else:
# 			self.thisptr = <intptr_t>(new Simplex_tree_multi_interface())
#
# 	def __dealloc__(self):
# 		cdef Simplex_tree_multi_interface* ptr = self.get_ptr()
# 		if ptr != NULL:
# 			del ptr
# 		# if self.pcohptr != NULL:
# 		#     del self.pcohptr
#
# 	def __is_defined(self):
# 		"""Returns true if SimplexTree pointer is not NULL.
# 			"""
# 		return self.get_ptr() != NULL
#
# 	# def __is_persistence_defined(self):
# 	#     """Returns true if Persistence pointer is not NULL.
# 	#      """
# 	#     return self.pcohptr != NULL
#
# 	def copy(self)->SimplexTree:
# 		"""
# 		:returns: A simplex tree that is a deep copy of itself.
# 		:rtype: SimplexTree
#
# 		:note: The persistence information is not copied. If you need it in the clone, you have to call
# 			:func:`compute_persistence` on it even if you had already computed it in the original.
# 		"""
# 		stree = SimplexTree()
# 		stree.thisptr = _get_copy_intptr(self)
# 		return stree
#
# 	def __deepcopy__(self):
# 		return self.copy()
#
# 	def filtration(self, simplex)->filtration_type:
# 		"""This function returns the filtration value for a given N-simplex in
# 		this simplicial complex, or +infinity if it is not in the complex.
#
# 		:param simplex: The N-simplex, represented by a list of vertex.
# 		:type simplex: list of int
# 		:returns:  The simplicial complex filtration value.
# 		:rtype:  float
# 		"""
# 		return self.get_ptr().simplex_filtration(simplex)
#
# 	def assign_filtration(self, simplex, filtration):
# 		"""This function assigns a new filtration value to a
# 		given N-simplex.
#
# 		:param simplex: The N-simplex, represented by a list of vertex.
# 		:type simplex: list of int
# 		:param filtration:  The new filtration value.
# 		:type filtration:  float
#
# 		.. note::
# 			Beware that after this operation, the structure may not be a valid
# 			filtration anymore, a simplex could have a lower filtration value
# 			than one of its faces. Callers are responsible for fixing this
# 			(with more :meth:`assign_filtration` or
# 			:meth:`make_filtration_non_decreasing` for instance) before calling
# 			any function that relies on the filtration property, like
# 			:meth:`persistence`.
# 		"""
# 		self.get_ptr().assign_simplex_filtration(simplex, filtration)
#
#
# 	def num_vertices(self)->int:
# 		"""This function returns the number of vertices of the simplicial
# 		complex.
#
# 		:returns:  The simplicial complex number of vertices.
# 		:rtype:  int
# 		"""
# 		return self.get_ptr().num_vertices()
#
# 	def num_simplices(self)->int:
# 		"""This function returns the number of simplices of the simplicial
# 		complex.
#
# 		:returns:  the simplicial complex number of simplices.
# 		:rtype:  int
# 		"""
# 		return self.get_ptr().num_simplices()
#
# 	def dimension(self)->dimension_type:
# 		"""This function returns the dimension of the simplicial complex.
#
# 		:returns:  the simplicial complex dimension.
# 		:rtype:  int
#
# 		.. note::
#
# 			This function is not constant time because it can recompute
# 			dimension if required (can be triggered by
# 			:func:`remove_maximal_simplex`
# 			or
# 			:func:`prune_above_filtration`
# 			methods).
# 		"""
# 		return self.get_ptr().dimension()
#
# 	def upper_bound_dimension(self)->dimension_type:
# 		"""This function returns a valid dimension upper bound of the
# 		simplicial complex.
#
# 		:returns:  an upper bound on the dimension of the simplicial complex.
# 		:rtype:  int
# 		"""
# 		return self.get_ptr().upper_bound_dimension()
#
# 	def set_dimension(self, dimension)->None:
# 		"""This function sets the dimension of the simplicial complex.
#
# 		:param dimension: The new dimension value.
# 		:type dimension: int
#
# 		.. note::
#
# 			This function must be used with caution because it disables
# 			dimension recomputation when required
# 			(this recomputation can be triggered by
# 			:func:`remove_maximal_simplex`
# 			or
# 			:func:`prune_above_filtration`
# 			).
# 		"""
# 		self.get_ptr().set_dimension(<int>dimension)
#
# 	def find(self, simplex)->bool:
# 		"""This function returns if the N-simplex was found in the simplicial
# 		complex or not.
#
# 		:param simplex: The N-simplex to find, represented by a list of vertex.
# 		:type simplex: list of int
# 		:returns:  true if the simplex was found, false otherwise.
# 		:rtype:  bool
# 		"""
# 		return self.get_ptr().find_simplex(simplex)
#
# 	def insert(self, simplex, filtration:list=[0.0])->bool:
# 		"""This function inserts the given N-simplex and its subfaces with the
# 		given filtration value (default value is '0.0'). If some of those
# 		simplices are already present with a higher filtration value, their
# 		filtration value is lowered.
#
# 		:param simplex: The N-simplex to insert, represented by a list of
# 			vertex.
# 		:type simplex: list of int
# 		:param filtration: The filtration value of the simplex.
# 		:type filtration: float
# 		:returns:  true if the simplex was not yet in the complex, false
# 			otherwise (whatever its original filtration value).
# 		:rtype:  bool
# 		"""
# 		return self.get_ptr().insert(simplex, <filtration_type>filtration)
#
# 	def get_simplices(self):
# 		"""This function returns a generator with simplices and their given
# 		filtration values.
#
# 		:returns:  The simplices.
# 		:rtype:  generator with tuples(simplex, filtration)
# 		"""
# 		cdef Simplex_tree_multi_simplices_iterator it = self.get_ptr().get_simplices_iterator_begin()
# 		cdef Simplex_tree_multi_simplices_iterator end = self.get_ptr().get_simplices_iterator_end()
# 		cdef Simplex_tree_multi_simplex_handle sh = dereference(it)
#
# 		while it != end:
# 			yield self.get_ptr().get_simplex_and_filtration(dereference(it))
# 			preincrement(it)
#
# 	def get_filtration(self):
# 		"""This function returns a generator with simplices and their given
# 		filtration values sorted by increasing filtration values.
#
# 		:returns:  The simplices sorted by increasing filtration values.
# 		:rtype:  generator with tuples(simplex, filtration)
# 		"""
# 		cdef vector[Simplex_tree_multi_simplex_handle].const_iterator it = self.get_ptr().get_filtration_iterator_begin()
# 		cdef vector[Simplex_tree_multi_simplex_handle].const_iterator end = self.get_ptr().get_filtration_iterator_end()
#
# 		while it != end:
# 			yield self.get_ptr().get_simplex_and_filtration(dereference(it))
# 			preincrement(it)
#
# 	def get_skeleton(self, dimension):
# 		"""This function returns a generator with the (simplices of the) skeleton of a maximum given dimension.
#
# 		:param dimension: The skeleton dimension value.
# 		:type dimension: int
# 		:returns:  The (simplices of the) skeleton of a maximum dimension.
# 		:rtype:  generator with tuples(simplex, filtration)
# 		"""
# 		cdef Simplex_tree_multi_skeleton_iterator it = self.get_ptr().get_skeleton_iterator_begin(dimension)
# 		cdef Simplex_tree_multi_skeleton_iterator end = self.get_ptr().get_skeleton_iterator_end(dimension)
#
# 		while it != end:
# 			yield self.get_ptr().get_simplex_and_filtration(dereference(it))
# 			preincrement(it)
#
# 	def get_star(self, simplex):
# 		"""This function returns the star of a given N-simplex.
#
# 		:param simplex: The N-simplex, represented by a list of vertex.
# 		:type simplex: list of int
# 		:returns:  The (simplices of the) star of a simplex.
# 		:rtype:  list of tuples(simplex, filtration)
# 		"""
# 		cdef simplex_type csimplex
# 		for i in simplex:
# 			csimplex.push_back(i)
# 		cdef vector[pair[simplex_type, filtration_type]] star \
# 			= self.get_ptr().get_star(csimplex)
# 		ct = []
# 		for filtered_simplex in star:
# 			v = []
# 			for vertex in filtered_simplex.first:
# 				v.append(vertex)
# 			ct.append((v, filtered_simplex.second))
# 		return ct
#
# 	def get_cofaces(self, simplex, codimension):
# 		"""This function returns the cofaces of a given N-simplex with a
# 		given codimension.
#
# 		:param simplex: The N-simplex, represented by a list of vertex.
# 		:type simplex: list of int
# 		:param codimension: The codimension. If codimension = 0, all cofaces
# 			are returned (equivalent of get_star function)
# 		:type codimension: int
# 		:returns:  The (simplices of the) cofaces of a simplex
# 		:rtype:  list of tuples(simplex, filtration)
# 		"""
# 		cdef vector[int] csimplex
# 		for i in simplex:
# 			csimplex.push_back(i)
# 		cdef vector[pair[simplex_type, filtration_type]] cofaces \
# 			= self.get_ptr().get_cofaces(csimplex, <int>codimension)
# 		ct = []
# 		for filtered_simplex in cofaces:
# 			v = []
# 			for vertex in filtered_simplex.first:
# 				v.append(vertex)
# 			ct.append((v, filtered_simplex.second))
# 		return ct
#
# 	def get_boundaries(self, simplex):
# 		"""This function returns a generator with the boundaries of a given N-simplex.
# 		If you do not need the filtration values, the boundary can also be obtained as
# 		:code:`itertools.combinations(simplex,len(simplex)-1)`.
#
# 		:param simplex: The N-simplex, represented by a list of vertex.
# 		:type simplex: list of int.
# 		:returns:  The (simplices of the) boundary of a simplex
# 		:rtype:  generator with tuples(simplex, filtration)
# 		"""
# 		cdef pair[Simplex_tree_multi_boundary_iterator, Simplex_tree_multi_boundary_iterator] it =  self.get_ptr().get_boundary_iterators(simplex)
#
# 		while it.first != it.second:
# 			yield self.get_ptr().get_simplex_and_filtration(dereference(it.first))
# 			preincrement(it.first)
#
# 	def remove_maximal_simplex(self, simplex):
# 		"""This function removes a given maximal N-simplex from the simplicial
# 		complex.
#
# 		:param simplex: The N-simplex, represented by a list of vertex.
# 		:type simplex: list of int
#
# 		.. note::
#
# 			The dimension of the simplicial complex may be lower after calling
# 			remove_maximal_simplex than it was before. However,
# 			:func:`upper_bound_dimension`
# 			method will return the old value, which
# 			remains a valid upper bound. If you care, you can call
# 			:func:`dimension`
# 			to recompute the exact dimension.
# 		"""
# 		self.get_ptr().remove_maximal_simplex(simplex)
#
# 	def prune_above_filtration(self, filtration)->bool:
# 		"""Prune above filtration value given as parameter.
#
# 		:param filtration: Maximum threshold value.
# 		:type filtration: float
# 		:returns: The filtration modification information.
# 		:rtype: bool
#
#
# 		.. note::
#
# 			Note that the dimension of the simplicial complex may be lower
# 			after calling
# 			:func:`prune_above_filtration`
# 			than it was before. However,
# 			:func:`upper_bound_dimension`
# 			will return the old value, which remains a
# 			valid upper bound. If you care, you can call
# 			:func:`dimension`
# 			method to recompute the exact dimension.
# 		"""
# 		return self.get_ptr().prune_above_filtration(filtration)
#
# 	def expansion(self, max_dim):
# 		"""Expands the simplex tree containing only its one skeleton
# 		until dimension max_dim.
#
# 		The expanded simplicial complex until dimension :math:`d`
# 		attached to a graph :math:`G` is the maximal simplicial complex of
# 		dimension at most :math:`d` admitting the graph :math:`G` as
# 		:math:`1`-skeleton.
# 		The filtration value assigned to a simplex is the maximal filtration
# 		value of one of its edges.
#
# 		The simplex tree must contain no simplex of dimension bigger than
# 		1 when calling the method.
#
# 		:param max_dim: The maximal dimension.
# 		:type max_dim: int
# 		"""
# 		cdef int maxdim = max_dim
# 		with nogil:
# 			self.get_ptr().expansion(maxdim)
#
# 	# def make_filtration_non_decreasing(self):
# 	#     """This function ensures that each simplex has a higher filtration
# 	#     value than its faces by increasing the filtration values.
# 	#
# 	#     :returns: True if any filtration value was modified,
# 	#         False if the filtration was already non-decreasing.
# 	#     :rtype: bool
# 	#     """
# 	#     return self.get_ptr().make_filtration_non_decreasing()

#
# 	def reset_filtration(self, filtration, min_dim = 0):
# 		"""This function resets the filtration value of all the simplices of dimension at least min_dim. Resets all the
# 		simplex tree when `min_dim = 0`.
# 		`reset_filtration` may break the filtration property with `min_dim > 0`, and it is the user's responsibility to
# 		make it a valid filtration (using a large enough `filt_value`, or calling `make_filtration_non_decreasing`
# 		afterwards for instance).
#
# 		:param filtration: New threshold value.
# 		:type filtration: float.
# 		:param min_dim: The minimal dimension. Default value is 0.
# 		:type min_dim: int.
# 		"""
# 		self.get_ptr().reset_filtration(filtration, min_dim)
#
# 	# def extend_filtration(self):
# 	#     """ Extend filtration for computing extended persistence. This function only uses the filtration values at the
# 	#     0-dimensional simplices, and computes the extended persistence diagram induced by the lower-star filtration
# 	#     computed with these values.
# 	#
# 	#     .. note::
# 	#
# 	#         Note that after calling this function, the filtration values are actually modified within the simplex tree.
# 	#         The function :func:`extended_persistence` retrieves the original values.
# 	#
# 	#     .. note::
# 	#
# 	#         Note that this code creates an extra vertex internally, so you should make sure that the simplex tree does
# 	#         not contain a vertex with the largest possible value (i.e., 4294967295).
# 	#
# 	#     This `notebook <https://github.com/GUDHI/TDA-tutorial/blob/master/Tuto-GUDHI-extended-persistence.ipynb>`_
# 	#     explains how to compute an extension of persistence called extended persistence.
# 	#     """
# 	#     self.get_ptr().compute_extended_filtration()
#
# 	# def extended_persistence(self, homology_coeff_field=11, min_persistence=0):
# 	#     """This function retrieves good values for extended persistence, and separate the diagrams into the Ordinary,
# 	#     Relative, Extended+ and Extended- subdiagrams.
# 	#
# 	#     :param homology_coeff_field: The homology coefficient field. Must be a prime number. Default value is 11. Max is 46337.
# 	#     :type homology_coeff_field: int
# 	#     :param min_persistence: The minimum persistence value (i.e., the absolute value of the difference between the
# 	#         persistence diagram point coordinates) to take into account (strictly greater than min_persistence).
# 	#         Default value is 0.0. Sets min_persistence to -1.0 to see all values.
# 	#     :type min_persistence: float
# 	#     :returns: A list of four persistence diagrams in the format described in :func:`persistence`. The first one is
# 	#         Ordinary, the second one is Relative, the third one is Extended+ and the fourth one is Extended-.
# 	#         See https://link.springer.com/article/10.1007/s10208-008-9027-z and/or section 2.2 in
# 	#         https://link.springer.com/article/10.1007/s10208-017-9370-z for a description of these subtypes.
# 	#
# 	#     .. note::
# 	#
# 	#         This function should be called only if :func:`extend_filtration` has been called first!
# 	#
# 	#     .. note::
# 	#
# 	#         The coordinates of the persistence diagram points might be a little different than the
# 	#         original filtration values due to the internal transformation (scaling to [-2,-1]) that is
# 	#         performed on these values during the computation of extended persistence.
# 	#
# 	#     This `notebook <https://github.com/GUDHI/TDA-tutorial/blob/master/Tuto-GUDHI-extended-persistence.ipynb>`_
# 	#     explains how to compute an extension of persistence called extended persistence.
# 	#     """
# 	#     cdef vector[pair[int, pair[double, double]]] persistence_result
# 	#     if self.pcohptr != NULL:
# 	#         del self.pcohptr
# 	#     self.pcohptr = new Simplex_tree_persistence_interface(self.get_ptr(), False)
# 	#     self.pcohptr.compute_persistence(homology_coeff_field, -1.)
# 	#     return self.pcohptr.compute_extended_persistence_subdiagrams(min_persistence)
#
# 	def expansion_with_blocker(self, max_dim, blocker_func):
# 		"""Expands the Simplex_tree containing only a graph. Simplices corresponding to cliques in the graph are added
# 		incrementally, faces before cofaces, unless the simplex has dimension larger than `max_dim` or `blocker_func`
# 		returns `True` for this simplex.
#
# 		The function identifies a candidate simplex whose faces are all already in the complex, inserts it with a
# 		filtration value corresponding to the maximum of the filtration values of the faces, then calls `blocker_func`
# 		with this new simplex (represented as a list of int). If `blocker_func` returns `True`, the simplex is removed,
# 		otherwise it is kept. The algorithm then proceeds with the next candidate.
#
# 		.. warning::
# 			Several candidates of the same dimension may be inserted simultaneously before calling `blocker_func`, so
# 			if you examine the complex in `blocker_func`, you may hit a few simplices of the same dimension that have
# 			not been vetted by `blocker_func` yet, or have already been rejected but not yet removed.
#
# 		:param max_dim: Expansion maximal dimension value.
# 		:type max_dim: int
# 		:param blocker_func: Blocker oracle.
# 		:type blocker_func: Callable[[List[int]], bool]
# 		"""
# 		self.get_ptr().expansion_with_blockers_callback(max_dim, callback, <void*>blocker_func)
#
# 	# def persistence(self, homology_coeff_field=11, min_persistence=0, persistence_dim_max = False):
# 	#     """This function computes and returns the persistence of the simplicial complex.
# 	#
# 	#     :param homology_coeff_field: The homology coefficient field. Must be a
# 	#         prime number. Default value is 11. Max is 46337.
# 	#     :type homology_coeff_field: int
# 	#     :param min_persistence: The minimum persistence value to take into
# 	#         account (strictly greater than min_persistence). Default value is
# 	#         0.0.
# 	#         Set min_persistence to -1.0 to see all values.
# 	#     :type min_persistence: float
# 	#     :param persistence_dim_max: If true, the persistent homology for the
# 	#         maximal dimension in the complex is computed. If false, it is
# 	#         ignored. Default is false.
# 	#     :type persistence_dim_max: bool
# 	#     :returns: The persistence of the simplicial complex.
# 	#     :rtype:  list of pairs(dimension, pair(birth, death))
# 	#     """
# 	#     self.compute_persistence(homology_coeff_field, min_persistence, persistence_dim_max)
# 	#     return self.pcohptr.get_persistence()
#
# 	# def compute_persistence(self, homology_coeff_field=11, min_persistence=0, persistence_dim_max = False):
# 	#     """This function computes the persistence of the simplicial complex, so it can be accessed through
# 	#     :func:`persistent_betti_numbers`, :func:`persistence_pairs`, etc. This function is equivalent to :func:`persistence`
# 	#     when you do not want the list :func:`persistence` returns.
# 	#
# 	#     :param homology_coeff_field: The homology coefficient field. Must be a
# 	#         prime number. Default value is 11. Max is 46337.
# 	#     :type homology_coeff_field: int
# 	#     :param min_persistence: The minimum persistence value to take into
# 	#         account (strictly greater than min_persistence). Default value is
# 	#         0.0.
# 	#         Sets min_persistence to -1.0 to see all values.
# 	#     :type min_persistence: float
# 	#     :param persistence_dim_max: If true, the persistent homology for the
# 	#         maximal dimension in the complex is computed. If false, it is
# 	#         ignored. Default is false.
# 	#     :type persistence_dim_max: bool
# 	#     :returns: Nothing.
# 	#     """
# 	#     if self.pcohptr != NULL:
# 	#         del self.pcohptr
# 	#     cdef bool pdm = persistence_dim_max
# 	#     cdef int coef = homology_coeff_field
# 	#     cdef double minp = min_persistence
# 	#     with nogil:
# 	#         self.pcohptr = new Simplex_tree_persistence_interface(self.get_ptr(), pdm)
# 	#         self.pcohptr.compute_persistence(coef, minp)
# 	#
# 	# def betti_numbers(self):
# 	#     """This function returns the Betti numbers of the simplicial complex.
# 	#
# 	#     :returns: The Betti numbers ([B0, B1, ..., Bn]).
# 	#     :rtype:  list of int
# 	#
# 	#     :note: betti_numbers function requires
# 	#         :func:`compute_persistence`
# 	#         function to be launched first.
# 	#     """
# 	#     assert self.pcohptr != NULL, "compute_persistence() must be called before betti_numbers()"
# 	#     return self.pcohptr.betti_numbers()
# 	#
# 	# def persistent_betti_numbers(self, from_value, to_value):
# 	#     """This function returns the persistent Betti numbers of the
# 	#     simplicial complex.
# 	#
# 	#     :param from_value: The persistence birth limit to be added in the
# 	#         numbers (persistent birth <= from_value).
# 	#     :type from_value: float
# 	#     :param to_value: The persistence death limit to be added in the
# 	#         numbers (persistent death > to_value).
# 	#     :type to_value: float
# 	#
# 	#     :returns: The persistent Betti numbers ([B0, B1, ..., Bn]).
# 	#     :rtype:  list of int
# 	#
# 	#     :note: persistent_betti_numbers function requires
# 	#         :func:`compute_persistence`
# 	#         function to be launched first.
# 	#     """
# 	#     assert self.pcohptr != NULL, "compute_persistence() must be called before persistent_betti_numbers()"
# 	#     return self.pcohptr.persistent_betti_numbers(<double>from_value, <double>to_value)
# 	#
# 	# def persistence_intervals_in_dimension(self, dimension):
# 	#     """This function returns the persistence intervals of the simplicial
# 	#     complex in a specific dimension.
# 	#
# 	#     :param dimension: The specific dimension.
# 	#     :type dimension: int
# 	#     :returns: The persistence intervals.
# 	#     :rtype:  numpy array of dimension 2
# 	#
# 	#     :note: intervals_in_dim function requires
# 	#         :func:`compute_persistence`
# 	#         function to be launched first.
# 	#     """
# 	#     assert self.pcohptr != NULL, "compute_persistence() must be called before persistence_intervals_in_dimension()"
# 	#     piid = np.array(self.pcohptr.intervals_in_dimension(dimension))
# 	#     # Workaround https://github.com/GUDHI/gudhi-devel/issues/507
# 	#     if len(piid) == 0:
# 	#         return np.empty(shape = [0, 2])
# 	#     return piid
# 	#
# 	# def persistence_pairs(self):
# 	#     """This function returns a list of persistence birth and death simplices pairs.
# 	#
# 	#     :returns: A list of persistence simplices intervals.
# 	#     :rtype:  list of pair of list of int
# 	#
# 	#     :note: persistence_pairs function requires
# 	#         :func:`compute_persistence`
# 	#         function to be launched first.
# 	#     """
# 	#     assert self.pcohptr != NULL, "compute_persistence() must be called before persistence_pairs()"
# 	#     return self.pcohptr.persistence_pairs()
# 	#
# 	# def write_persistence_diagram(self, persistence_file):
# 	#     """This function writes the persistence intervals of the simplicial
# 	#     complex in a user given file name.
# 	#
# 	#     :param persistence_file: Name of the file.
# 	#     :type persistence_file: string
# 	#
# 	#     :note: intervals_in_dim function requires
# 	#         :func:`compute_persistence`
# 	#         function to be launched first.
# 	#     """
# 	#     assert self.pcohptr != NULL, "compute_persistence() must be called before write_persistence_diagram()"
# 	#     self.pcohptr.write_output_diagram(persistence_file.encode('utf-8'))
# 	#
# 	# def lower_star_persistence_generators(self):
# 	#     """Assuming this is a lower-star filtration, this function returns the persistence pairs,
# 	#     where each simplex is replaced with the vertex that gave it its filtration value.
# 	#
# 	#     :returns: First the regular persistence pairs, grouped by dimension, with one vertex per extremity,
# 	#         and second the essential features, grouped by dimension, with one vertex each
# 	#     :rtype: Tuple[List[numpy.array[int] of shape (n,2)], List[numpy.array[int] of shape (m,)]]
# 	#
# 	#     :note: lower_star_persistence_generators requires that `persistence()` be called first.
# 	#     """
# 	#     assert self.pcohptr != NULL, "lower_star_persistence_generators() requires that persistence() be called first."
# 	#     gen = self.pcohptr.lower_star_generators()
# 	#     normal = [np.array(d).reshape(-1,2) for d in gen.first]
# 	#     infinite = [np.array(d) for d in gen.second]
# 	#     return (normal, infinite)
# 	#
# 	# def flag_persistence_generators(self):
# 	#     """Assuming this is a flag complex, this function returns the persistence pairs,
# 	#     where each simplex is replaced with the vertices of the edges that gave it its filtration value.
# 	#
# 	#     :returns: First the regular persistence pairs of dimension 0, with one vertex for birth and two for death;
# 	#         then the other regular persistence pairs, grouped by dimension, with 2 vertices per extremity;
# 	#         then the connected components, with one vertex each;
# 	#         finally the other essential features, grouped by dimension, with 2 vertices for birth.
# 	#     :rtype: Tuple[numpy.array[int] of shape (n,3), List[numpy.array[int] of shape (m,4)], numpy.array[int] of shape (l,), List[numpy.array[int] of shape (k,2)]]
# 	#
# 	#     :note: flag_persistence_generators requires that `persistence()` be called first.
# 	#     """
# 	#     assert self.pcohptr != NULL, "flag_persistence_generators() requires that persistence() be called first."
# 	#     gen = self.pcohptr.flag_generators()
# 	#     if len(gen.first) == 0:
# 	#         normal0 = np.empty((0,3))
# 	#         normals = []
# 	#     else:
# 	#         l = iter(gen.first)
# 	#         normal0 = np.array(next(l)).reshape(-1,3)
# 	#         normals = [np.array(d).reshape(-1,4) for d in l]
# 	#     if len(gen.second) == 0:
# 	#         infinite0 = np.empty(0)
# 	#         infinites = []
# 	#     else:
# 	#         l = iter(gen.second)
# 	#         infinite0 = np.array(next(l))
# 	#         infinites = [np.array(d).reshape(-1,2) for d in l]
# 	#     return (normal0, normals, infinite0, infinites)
# 	#
# 	# def collapse_edges(self, nb_iterations = 1):
# 	#     """Assuming the complex is a graph (simplices of higher dimension are ignored), this method implicitly
# 	#     interprets it as the 1-skeleton of a flag complex, and replaces it with another (smaller) graph whose
# 	#     expansion has the same persistent homology, using a technique known as edge collapses
# 	#     (see :cite:`edgecollapsearxiv`).
# 	#
# 	#     A natural application is to get a simplex tree of dimension 1 from :class:`~gudhi.RipsComplex`,
# 	#     then collapse edges, perform :meth:`expansion()` and finally compute persistence
# 	#     (cf. :download:`rips_complex_edge_collapse_example.py <../example/rips_complex_edge_collapse_example.py>`).
# 	#
# 	#     :param nb_iterations: The number of edge collapse iterations to perform. Default is 1.
# 	#     :type nb_iterations: int
# 	#     """
# 	#     # Backup old pointer
# 	#     cdef Simplex_tree_multi_interface* ptr = self.get_ptr()
# 	#     cdef int nb_iter = nb_iterations
# 	#     with nogil:
# 	#         # New pointer is a new collapsed simplex tree
# 	#         self.thisptr = <intptr_t>(ptr.collapse_edges(nb_iter))
# 	#         # Delete old pointer
# 	#         del ptr
#
# 	def __eq__(self, other:SimplexTree):
# 		"""Test for structural equality
# 		:returns: True if the 2 simplex trees are equal, False otherwise.
# 		:rtype: bool
# 		"""
# 		return dereference(self.get_ptr()) == dereference(other.get_ptr())
#
# cdef intptr_t _get_copy_intptr(SimplexTree stree) nogil:
# 	return <intptr_t>(new Simplex_tree_multi_interface(dereference(stree.get_ptr())))
#

