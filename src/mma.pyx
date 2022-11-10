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

###########################################################################
#CYTHON TYPES
from cython cimport numeric
from libcpp.vector cimport vector
from libcpp.utility cimport pair
#from libcpp.list cimport list as clist
from libcpp cimport bool
from libcpp cimport int
from libcpp.string cimport string
ctypedef  size_t uintptr_t

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

cdef extern from "approximation.h" namespace "Vineyard":
	# Approximation
	Module compute_vineyard_barcode_approximation(boundary_matrix &B, vector[vector[double]] &filters_list, double precision, Box &box, bool threshold, bool complete, bool multithread, bool verbose)


cdef extern from "format_python-cpp.h":
	#list_simplicies_to_sparse_boundary_matrix
	vector[vector[unsigned int]] build_sparse_boundary_matrix_from_simplex_list(vector[vector[unsigned int]] list_simplices)
	#list_simplices_ls_filtration_to_sparse_boundary_filtration
	pair[vector[vector[unsigned int]], vector[vector[double]]] build_boundary_matrix_from_simplex_list(vector[vector[unsigned int]] list_simplices, vector[vector[double]] filtrations, vector[unsigned int] filters_to_permute)
	# pair[vector[vector[unsigned int]], vector[double]] simplextree_to_boundary_filtration(vector[boundary_type] &simplexList, filtration_type &filtration)
	pair[boundary_matrix, filtration_type] simplextree_to_boundary_filtration(uintptr_t)

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
	B:Union[list,_SimplexTree], 
	filters:Union[_np.ndarray, list], 
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
	if(type(filters) == _np.ndarray):
		#assert filters.shape[1] == 2
		filtration = [filters[:,i] for i in range(filters.shape[1])]
	else:
		filtration = filters

	if type(B) == _SimplexTree:
		if verbose:
			print("Converting simplextree to boundary matrix...")
		boundary,_ = simplextree_to_boundary_filtration(B.thisptr)
	else:
		boundary = B
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
	

def splx2bf(simplextree:_SimplexTree):
	"""Computes a (sparse) boundary matrix, with associated filtration. Can be used as an input of approx afterwards.
	
	Parameters
	----------
	simplextree:SimplexTree
		The simplextree defining the filtration to convert to boundary-filtration.
	
	Returns
	-------
	B:List of lists of ints
		The boundary matrix.
	F: List of filtration values
		The filtrations aligned with B; the i-th simplex of this simplextree has boundary B[i] and filtration F[i].
	
	"""
	return simplextree_to_boundary_filtration(simplextree.thisptr)


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


def collapse_2pers(tree:_SimplexTree, F2, max_dimension:int=None, num:int=1):
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
	if type(F2) is list:
		F2 = _np.array(F2)
	if type(F2) is _np.ndarray:
		F = lambda x : _np.max(F2[x])
	else:
		# We assume here that F2 is callable, and output the simplex values.
		F=F2
	tree_edges = [(tuple(splx), (f1, F(splx))) for splx,f1 in tree.get_skeleton(1) if len(splx) == 2]
	reduced_edges = _remove_strongly_filtration_dominated(tree_edges)
	reduced_tree = _gd.SimplexTree()
	for splx, f in tree.get_skeleton(0):
		reduced_tree.insert(splx, f)
	for e, (f1, f2) in reduced_edges:
		reduced_tree.insert(e, f1)
	reduced_tree.expansion(max_dimension)
	return collapse_2pers(reduced_tree, F, max_dimension, num-1)
