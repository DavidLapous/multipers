"""!
@package mma
@brief Files containing the C++ cythonized functions.
@author David Loiseaux
@copyright Copyright (c) 2022 Inria.
"""

# distutils: language = c++

###########################################################################
## PYTHON LIBRARIES
import gudhi as gd
import numpy as np
from typing import List
import pickle as pk
import matplotlib.pyplot as plt

###########################################################################
## CPP CLASSES
from libc.stdint cimport intptr_t
from libc.stdint cimport uintptr_t

###########################################################################
## CYTHON TYPES
from libcpp.vector cimport vector
from libcpp.utility cimport pair
#from libcpp.list cimport list as clist
from libcpp cimport bool
from libcpp cimport int
from typing import Iterable

#########################################################################
## Multipersistence Module Approximation Classes
from multipers.multiparameter_module_approximation cimport *



#########################################################################
## Small hack for typing
from gudhi import SimplexTree
from multipers.simplex_tree_multi import SimplexTreeMulti
# cimport numpy as cnp
# cnp.import_array()

###################################### MMA
cdef extern from "multiparameter_module_approximation/approximation.h" namespace "Gudhi::multiparameter::mma":
	Module compute_vineyard_barcode_approximation(boundary_matrix, vector[Finitely_critical_multi_filtration] , value_type precision, Box[value_type] &, bool threshold, bool complete, bool multithread, bool verbose) nogil


cdef class PySummand:
	"""
	Stores a Summand of a PyModule
	"""
	cdef Summand sum
	# def __cinit__(self, vector[corner_type]& births, vector[corner_type]& deaths, int dim):
	# 	self.sum = Summand(births, deaths, dim)

	def get_birth_list(self): 
		return np.asarray(Finitely_critical_multi_filtration.to_python(self.sum.get_birth_list()))

	def get_death_list(self):
		return np.asarray(Finitely_critical_multi_filtration.to_python(self.sum.get_death_list()))
	@property
	def degree(self)->int:
		return self.sum.get_dimension()
	
	cdef set(self, Summand& summand):
		self.sum = summand
		return self
	def get_bounds(self):
		cdef pair[Finitely_critical_multi_filtration,Finitely_critical_multi_filtration] cbounds
		with nogil:
			cbounds = self.sum.get_bounds().get_pair()
		# return np.array(<value_type[:self.num_parameters]>(&cbounds.first[0])),np.array(<value_type[:self.num_parameters]>(&cbounds.second[0]))
		return np.asarray(cbounds.first._convert_back()), np.asarray(cbounds.second._convert_back())

cdef get_summand_filtration_values(Summand summand):
	births = np.asarray(Finitely_critical_multi_filtration.to_python(summand.get_birth_list()))
	deaths = np.asarray(Finitely_critical_multi_filtration.to_python(summand.get_death_list()))
	pts = np.concatenate([births,deaths],axis=0)
	num_parameters = pts.shape[1]
	out = [np.unique(pts[:,parameter]) for parameter in range(num_parameters)]
	out = [f[:-1] if f[-1] == np.inf else f for f in out]
	out = [f[1:]  if f[0] == -np.inf else f for f in out]
	return out

cdef class PyBox:
	cdef Box[value_type] box
	def __cinit__(self, corner_type bottomCorner, corner_type topCorner):
		self.box = Box[value_type](bottomCorner, topCorner)
	@property
	def num_parameters(self):
		cdef size_t dim = self.box.get_bottom_corner().size()
		if dim == self.box.get_upper_corner().size():	return dim
		else:	print("Bad box definition.")
	def contains(self, x):
		return self.box.contains(x)
	cdef set(self, Box[value_type]& b):
		self.box = b
		return self

	def get(self):
		return [self.box.get_bottom_corner().get_vector(),self.box.get_upper_corner().get_vector()]
	def to_multipers(self):
		#assert (self.get_dimension() == 2) "Multipers only works in dimension  2 !"
		return np.array(self.get()).flatten(order = 'F')

cdef class PyMultiDiagramPoint:
	cdef MultiDiagram_point point
	cdef set(self, MultiDiagram_point &pt):
		self.point = pt
		return self

	def get_degree(self):
		return self.point.get_dimension()
	def get_birth(self):
		return self.point.get_birth()
	def get_death(self):
		return self.point.get_death()

cdef class PyMultiDiagram:
	"""
	Stores the diagram of a PyModule on a line
	"""
	cdef MultiDiagram multiDiagram
	cdef set(self, MultiDiagram m):
		self.multiDiagram = m
		return self
	def get_points(self, degree:int=-1) -> np.ndarray:
		out = self.multiDiagram.get_points(degree)
		if len(out) == 0 and len(self) == 0:
			return np.empty() # TODO Retrieve good number of parameters if there is no points in diagram
		if len(out) == 0:
			return np.empty((0,2,self.multiDiagram.at(0).get_dimension())) # gets the number of parameters
		return np.array(out)
	def to_multipers(self, dimension:int):
		return self.multiDiagram.to_multipers(dimension)
	def __len__(self) -> int:
		return self.multiDiagram.size()
	def __getitem__(self,i:int) -> PyMultiDiagramPoint:
		return PyMultiDiagramPoint().set(self.multiDiagram.at(i % self.multiDiagram.size()))
cdef class PyMultiDiagrams:
	"""
	Stores the barcodes of a PyModule on multiple lines
	"""
	cdef MultiDiagrams multiDiagrams
	cdef set(self,MultiDiagrams m):
		self.multiDiagrams = m
		return self
	def to_multipers(self):
		out = self.multiDiagrams.to_multipers()
		# return out
		return [np.array(summand, dtype=np.float64) for summand in out]
	def __getitem__(self,i:int):
		if i >=0 :
			return PyMultiDiagram().set(self.multiDiagrams.at(i))
		else:
			return PyMultiDiagram().set(self.multiDiagrams.at( self.multiDiagrams.size() - i))
	def __len__(self):
		return self.multiDiagrams.size()
	def get_points(self, degree:int=-1):
		return self.multiDiagrams.get_points()
		# return np.array([x.get_points(dimension) for x in self.multiDiagrams], dtype=float)
		# return np.array([PyMultiDiagram().set(x).get_points(dimension) for x in self.multiDiagrams])
	cdef _get_plot_bars(self, dimension:int=-1, min_persistence:float=0):
		return self.multiDiagrams._for_python_plot(dimension, min_persistence);
	def plot(self, degree:int=-1, min_persistence:float=0):
		"""
		Plots the barcodes.

		Parameters
		----------
		degree:int=-1
			Only plots the bars of specified homology degree. Useful when the multidiagrams contains multiple dimenions
		min_persistence:float=0
			Only plot bars of length greater than this value. Useful to reduce the time to plot.

		Warning
		-------
		If the barcodes are not thresholded, essential barcodes will not be displayed !

		Returns
		-------
		Nothing
		"""
		from cycler import cycler
		import matplotlib
		import matplotlib.pyplot as plt
		if len(self) == 0: return
		_cmap = matplotlib.colormaps["Spectral"]
		multibarcodes_, colors = self._get_plot_bars(degree, min_persistence)
		n_summands = np.max(colors)+1 if len(colors)>0 else 1

		plt.rc('axes', prop_cycle = cycler('color', [_cmap(i/n_summands) for i in colors]))
		plt.plot(*multibarcodes_)
cdef class PyLine:
	cdef Line[value_type] line

cdef class PyModule:
	"""
	Stores a representation of a n-persistence module.
	"""
	cdef Module cmod

	cdef set(self, Module m):
		self.cmod = m
	def set_box(self, PyBox pybox):
		cdef Box[value_type] cbox = pybox.box
		with nogil:
			self.cmod.set_box(cbox)
		return self
	def get_module_of_degree(self, int degree)->PyModule: # TODO : in c++ ?
		pmodule = PyModule()
		cdef Box[value_type] c_box = self.cmod.get_box()
		pmodule.cmod.set_box(c_box) 
		with nogil:
			for summand in self.cmod:
				if summand.get_dimension() == degree:
					pmodule.cmod.add_summand(summand)
		return pmodule
	def get_module_of_degrees(self, degrees:Iterable[int])->PyModule: # TODO : in c++ ?
		pmodule = PyModule()
		cdef Box[value_type] c_box = self.cmod.get_box()
		pmodule.cmod.set_box(c_box)
		cdef vector[int] cdegrees = degrees
		with nogil:
			for summand in self.cmod:
				for d in cdegrees:
					if d == summand.get_dimension():
						pmodule.cmod.add_summand(summand)
		return pmodule
	def _compute_pixels(self,coordinates:np.ndarray,
			degrees=None, box=None, value_type delta=.1, 
			value_type p=1., bool normalize=False, int n_jobs=0):
		if degrees is not None: assert np.all(degrees[:-1] <= degrees[1:]), "Degrees have to be sorted"
		cdef vector[int] cdegrees = np.arange(self.max_degree +1) if degrees is None else degrees
		pybox = PyBox(*self.get_box()) if box is None else PyBox(*box)
		cdef Box[value_type] cbox = pybox.box
		cdef vector[vector[value_type]] ccoords = coordinates
		cdef vector[vector[value_type]] out 
		with nogil:
			out = self.cmod.compute_pixels(ccoords, cdegrees, cbox, delta, p, normalize, n_jobs)
		return np.asarray(out)
	def __len__(self)->int:
		return self.cmod.size()
	def get_bottom(self)->list:
		return self.cmod.get_box().get_bottom_corner().get_vector()
	def get_top(self)->list:
		return self.cmod.get_box().get_upper_corner().get_vector()
	def get_box(self):
		return np.asarray([self.get_bottom(), self.get_top()])
	@property
	def max_degree(self)->int:
		return self.cmod.get_dimension()
	@property
	def num_parameters(self)->int:
		cdef size_t dim = self.cmod.get_box().get_bottom_corner().size()
		assert dim == self.cmod.get_box().get_upper_corner().size(), "Bad box definition, cannot infer num_parameters."
		return dim
	def dump(self, path:str|None=None):
		"""
		Dumps the module into a pickle-able format.
		
		Parameters
		----------
		path:str=None (optional) saves the pickled module in specified path
		
		Returns
		-------
		list of list, encoding the module, which can be retrieved with the function `from_dump`.
		"""
		## TODO : optimize, but not really used.
		return dump_cmod(self.cmod) 
	def __getstate__(self):
		return self.dump()
	def __setstate__(self,dump):
		cdef Module cmod = cmod_from_dump(dump)
		self.cmod = cmod
		return
	def __getitem__(self, i) -> PySummand:
		if i == slice(None):
			return self
		summand = PySummand()
		summand.set(self.cmod.at(i % self.cmod.size()))
		return summand
	def get_bounds(self):
		cdef pair[Finitely_critical_multi_filtration,Finitely_critical_multi_filtration] cbounds
		with nogil:
			cbounds = self.cmod.get_bounds().get_pair()
		# return np.array(<value_type[:self.num_parameters]>(&cbounds.first[0])),np.array(<value_type[:self.num_parameters]>(&cbounds.second[0]))
		return np.asarray(cbounds.first._convert_back()), np.asarray(cbounds.second._convert_back())
	def rescale(self,rescale_factors, int degree=-1):
		"""
		Rescales the fitlration values of the summands by this rescaling vector.
		"""
		cdef vector[value_type] crescale_factors = rescale_factors
		with nogil:
			self.cmod.rescale(crescale_factors,degree)
	def translate(self,translation, int degree=-1):
		"""
		Translates the module in the filtration space by this vector.
		"""
		cdef vector[value_type] ctranslation = translation
		with nogil:
			self.cmod.translate(ctranslation,degree)

	def get_filtration_values(self, unique=True):
		"""
		Retrieves all filtration values of the summands of the module.
		
		Output format 
		-------------

		list of filtration values for parameter.
		"""
		if len(self) ==0:
			return np.empty((self.num_parameters,0))
		values = [get_summand_filtration_values(summand) for summand in self.cmod]
		values = [np.concatenate([f[parameter] for f in values], axis=0) for  parameter in range(self.num_parameters)]
		if unique:
			return [np.unique(f) for f in values]
		return values

	def plot(self, degree:int=-1,**kwargs)->None:
		"""Shows the module on a plot. Each color corresponds to an apprimation summand of the module, and its shape corresponds to its support.
		Only works with 2-parameter modules.

		Parameters
		----------
		degree = -1 : integer
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
		from multipers.plots import plot2d_PyModule
		if (kwargs.get('box')):
			box = kwargs.pop('box')
		else:
			box = [self.get_bottom(), self.get_top()]
		if (len(box[0]) != 2):
			print("Filtration size :", len(box[0]), " != 2")
			return
		num = 0
		if(degree < 0):
			ndim = self.cmod.get_dimension()+1
			scale = kwargs.pop("scale", 4)
			fig, axes = plt.subplots(1, ndim, figsize=(ndim*scale,scale))
			for degree in range(ndim):
				plt.sca(axes[degree]) if ndim > 1 else  plt.sca(axes)
				self.plot(degree,box=box,**kwargs)
			return
		corners = self.cmod.get_corners_of_dimension(degree)
		plot2d_PyModule(corners, box=box, dimension=degree, **kwargs)
		return

	def barcode(self, basepoint, degree:int,*, threshold = False): # TODO direction vector interface
		"""Computes the barcode of module along a lines.

		Parameters
		----------
		basepoint  : vector
			basepoint of the lines on which to compute the barcodes, i.e. a point on the line
		degree = -1 : integer
			Homology degree on which to compute the bars. If negative, every dimension is computed
		box (default) :
			box on which to compute the barcodes if basepoints is not specified. Default is a linspace of lines crossing that box.
		threshold = False : 
			Thre

		Warning
		-------
		If the barcodes are not thresholded, essential barcodes will not be plot-able.

		Returns
		-------
		PyMultiDiagrams
			Structure that holds the barcodes. Barcodes can be retrieved with a .get_points() or a .to_multipers() or a .plot().
		"""
		out = PyMultiDiagram()
		out.set(self.cmod.get_barcode(Line[value_type](basepoint), degree, threshold))
		return out
	def barcodes(self, degree:int, basepoints = None, num=100, box = None,threshold = False):
		"""Computes barcodes of module along a set of lines.

		Parameters
		----------
		basepoints = None : list of vectors
			basepoints of the lines on which to compute the barcodes.
		degree = -1 : integer
			Homology degree on which to compute the bars. If negative, every dimension is computed
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
		if box is None:
			box = [self.get_bottom(), self.get_top()]
		if (len(box[0]) != 2) and (basepoints is None):
			raise ValueError("Basepoints has to be specified for filtration dimension >= 3 !")
		elif basepoints is None:
			h = box[1][1] - box[0][1]
			basepoints = np.linspace([box[0][0] - h,box[0][1]], [box[1][0],box[0][1]], num=num) 
		else :
			num=len(basepoints)
		cdef vector[cfiltration_type] cbasepoints
		for i in range(num):
			cbasepoints.push_back(Finitely_critical_multi_filtration(basepoints[i]))
		
		out.set(self.cmod.get_barcodes(cbasepoints, degree, threshold))
		return out
	def landscape(self, degree:int, k:int=0,box:list|np.ndarray|None=None, resolution:List=[100,100], bool plot=False):
		"""Computes the multiparameter landscape from a PyModule. Python interface only bifiltrations.

		Parameters
		----------
		degree : integer
			The homology degree of the landscape.
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
			box = self.get_box()
		cdef Box[value_type] c_box = Box[value_type](box)
		out = np.array(self.cmod.get_landscape(degree, k, c_box, resolution))
		if plot:
			plt.figure()
			aspect = (box[1][0]-box[0][0]) / (box[1][1]-box[0][1])
			extent = [box[0][0], box[1][0], box[0][1], box[1][1]]
			plt.imshow(out.T, origin="lower", extent=extent, aspect=aspect)
		return out
	def landscapes(self, degree:int, ks:list|np.ndarray=[0],box=None, resolution:list|np.ndarray=[100,100], bool plot=False):
		"""Computes the multiparameter landscape from a PyModule. Python interface only bifiltrations.

		Parameters
		----------
		degree : integer
			The homology degree of the landscape.
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
			box = self.get_box()
		out = np.array(self.cmod.get_landscapes(degree, ks, Box[value_type](box), resolution))
		if plot:
			to_plot = np.sum(out, axis=0)
			plt.figure()
			aspect = (box[1][0]-box[0][0]) / (box[1][1]-box[0][1])
			extent = [box[0][0], box[1][0], box[0][1], box[1][1]]
			plt.imshow(to_plot.T, origin="lower", extent=extent, aspect=aspect)
		return out


	def image(self, degrees:Iterable[int]=None, bandwidth:float=0.1, resolution:list|int=50, normalize:bool=False, plot:bool=False, save:bool=False, dpi:int=200,p:float=1., box=None, flatten=False, int n_jobs=0,  **kwargs)->np.ndarray:
		"""Computes a vectorization from a PyModule. Python interface only bifiltrations.

		Parameters
		----------
		degree = -1 : integer
			If positive returns only the image of homology degree `degree`.
		bandwidth = 0.1 : float
			Image parameter. TODO : different weights
		resolution = [100,100] : pair of integers
			Resolution of the image(s).
		normalize = True : Boolean
			Ensures that the image belongs to [0,1].
		plot = False : Boolean
			If true, plots the images;
		flatten=False :
			If False, reshapes the output to the expected shape.

		Returns
		-------
		List of Numpy arrays or numpy array
			The list of images, or the image of fixed dimension.
		"""
		# box = kwargs.get("box",[self.get_bottom(),self.get_top()])
		if box is None:
			box = self.get_box()
		num_parameters = self.num_parameters
		if degrees is None:
			degrees = np.arange(self.max_degree +1)
		num_degrees = len(degrees)
		try:
			int(resolution)
			resolution = [resolution]*num_parameters
		except:
			pass

		xx = [np.linspace(*np.asarray(box)[:,parameter], num=res) for parameter, res in zip(range(num_parameters), resolution)]
		mesh = np.meshgrid(*xx)
		coordinates = np.concatenate([stuff.flatten()[:,None] for stuff in mesh], axis=1)

		# if degree < 0:
		# 	image_vector = np.array(self.cmod.get_vectorization(bandwidth, p, normalize, Box[value_type](box), resolution[0], resolution[1]))
		# else:
		# 	image_vector = np.array([self.cmod.get_vectorization_in_dimension(degree, bandwidth, p,normalize,Box[value_type](box),  resolution[0], resolution[1])])
		
		concatenated_images = self._compute_pixels(coordinates, degrees=degrees, box=box, delta=bandwidth, p=p, normalize=normalize,n_jobs=n_jobs)
		if flatten:
			image_vector = concatenated_images.reshape((len(degrees),-1))
			if plot:
				from warnings import warn
				warn("Unflatten to plot.")
			return image_vector
		else:
			image_vector = concatenated_images.reshape((len(degrees),*resolution))
		if plot:
			assert num_parameters == 2 
			i=0
			n_plots = len(image_vector)
			scale:float = kwargs.get("size", 4.0)
			fig, axs = plt.subplots(1,n_plots, figsize=(n_plots*scale,scale))
			aspect = (box[1][0]-box[0][0]) / (box[1][1]-box[0][1])
			extent = [box[0][0], box[1][0], box[0][1], box[1][1]]
			for image, degree, i in zip(image_vector, degrees, range(num_degrees)):
				ax = axs if n_plots <= 1 else axs[i]
				temp = ax.imshow(image,origin="lower",extent=extent, aspect=aspect)
				if (kwargs.get('colorbar') or kwargs.get('cb')):
					plt.colorbar(temp, ax = ax)
				if degree < 0 :
					ax.set_title(rf"$H_{i}$ $2$-persistence image")
				if degree >= 0:
					ax.set_title(rf"$H_{degree}$ $2$-persistence image")
		return image_vector


	def euler_char(self, points:list|np.ndarray) -> np.ndarray:
		""" Computes the Euler Characteristic of the filtered complex at given (multiparameter) time

		Parameters
		----------
		points: list[float] | list[list[float]] | np.ndarray
			List of filtration values on which to compute the euler characteristic.
			WARNING FIXME : the points have to have the same dimension as the simplextree.	

		Returns
		-------
		The list of euler characteristic values
		"""
		if len(points) == 0:
			return []
		if type(points[0]) is float:
			points = [points]
		if type(points) is np.ndarray:
			assert len(points.shape) in [1,2]
			if len(points.shape) == 1:
				points = [points]
		cdef vector[Finitely_critical_multi_filtration] c_points = Finitely_critical_multi_filtration.from_python(points)
		cdef Module c_mod = self.cmod
		with nogil:
			c_euler = c_mod.euler_curve(c_points)
		euler = c_euler
		return np.array(euler, dtype=int)

def module_approximation(
	st:SimplexTreeMulti|None=None,
	max_error:float|None = None,
	box:list|np.ndarray|None = None,
	threshold:bool = False,
	complete:bool = True,
	multithread:bool = False, 
	verbose:bool = False,
	ignore_warning:bool = False,
	nlines:int = 500,
	max_dimension=np.inf,
	boundary = None,
	filtration = None,
	return_timings:bool = False,
	**kwargs
	):
	"""Computes an interval module approximation of a multiparameter filtration.

	Parameters
	----------
	st : n-filtered Simplextree, or None if boundary and filtration are provided.
		Defines the n-filtration on which to compute the homology.
	max_error: positive float
		Trade-off between approximation and computational complexity.
		Upper bound of the module approximation, in bottleneck distance, 
		for interval-decomposable modules.
	nlines: int
		Alternative to precision.
	box : pair of list of floats
		Defines a rectangle on which to compute the approximation.
		Format : [x,y], where x,y defines the rectangle {z : x ≤ z ≤ y}
	threshold: bool
		When true, intersects the module support with the box.
	verbose: bool
		Prints C++ infos.
	ignore_warning : bool
		Unless set to true, prevents computing on more than 10k lines. Useful to prevent a segmentation fault due to "infinite" recursion.
	return_timings : bool
		If true, will return the time to compute instead (computed in python, using perf_counter_ns).
	Returns
	-------
	PyModule
		An interval decomposable module approximation of the module defined by the
		homology of this multi-filtration.
	"""

	if boundary is None or filtration is None:
		boundary,filtration = simplex_tree2boundary_filtrations(st) # TODO : recomputed each time... maybe store this somewhere ?
	if max_dimension < np.inf: # TODO : make it more efficient
		nsplx = len(boundary)
		for i in range(nsplx-1,-1,-1):
			b = boundary[i]
			dim=len(b) -1
			if dim>max_dimension:
				boundary.pop(i)
				for f in filtration:
					f.pop(i)
	nfiltration = len(filtration)
	if nfiltration <= 0:
		return PyModule()
	if nfiltration == 1 and not(st is None):
		st = st.project_on_line(0)
		return st.persistence()

	if box is None and not(st is None):
		m,M = st.filtration_bounds()
	elif box is not None:
		m,M = box
	else:
		m, M = np.min(filtration, axis=0), np.max(filtration, axis=0)
	prod = 1
	h = M[-1] - m[-1]
	for i, [a,b] in enumerate(zip(m,M)):
		if i == len(M)-1:	continue
		prod *= (b-a + h)

	if max_error is None:
		max_error:float = (prod/nlines)**(1/(nfiltration-1))

	if box is None:
		M = [np.max(f)+2*max_error for f in filtration]
		m = [np.min(f)-2*max_error for f in filtration]
		box = [m,M]

	if ignore_warning and prod >= 20_000:
		from warnings import warn
		warn(f"Warning : the number of lines (around {np.round(prod)}) may be too high. Try to increase the precision parameter, or set `ignore_warning=True` to compute this module. Returning the trivial module.")
		return PyModule()
	
	approx_mod = PyModule()
	cdef vector[Finitely_critical_multi_filtration] c_filtration = Finitely_critical_multi_filtration.from_python(filtration)
	cdef boundary_matrix c_boundary = boundary
	cdef value_type c_max_error = max_error
	cdef bool c_threshold = threshold
	cdef bool c_complete = complete
	cdef bool c_multithread = multithread
	cdef bool c_verbose = verbose
	cdef Box[value_type] c_box = Box[value_type](box)
	if return_timings:
		from time import perf_counter_ns
		t = perf_counter_ns()
	with nogil:
		c_mod = compute_vineyard_barcode_approximation(c_boundary,c_filtration,c_max_error, c_box, c_threshold, c_complete, c_multithread,c_verbose)
	if return_timings:
		t = perf_counter_ns() -t 
		t /= 10**9
		return t
	approx_mod.set(c_mod)
	return approx_mod



	def dump(self, path:str|None=None):
		"""
		Dumps the module into a pickle-able format.
		
		Parameters
		----------
		path:str=None (optional) saves the pickled module in specified path
		
		Returns
		-------
		list of list, encoding the module, which can be retrieved with the function `from_dump`.
		"""
		## TODO : optimize, but not really used.
		out = [[] for _ in range(self.cmod.get_dimension()+1)]
		out += [self.get_box()]

		for summand in self.cmod:
			out[summand.get_dimension()].append((
				Finitely_critical_multi_filtration.to_python(summand.get_birth_list()), 
				Finitely_critical_multi_filtration.to_python(summand.get_death_list())
			))
		if path is None:
			return out
		pk.dump(out, open(path, "wb"))
		return out

cdef dump_summand(Summand& summand):
	return (
		np.asarray(Finitely_critical_multi_filtration.to_python(summand.get_birth_list())),
		np.asarray(Finitely_critical_multi_filtration.to_python(summand.get_death_list())), 
		summand.get_dimension()
	)

cdef Summand summand_from_dump(summand_dump):
	cdef vector[Finitely_critical_multi_filtration] births = Finitely_critical_multi_filtration.from_python(summand_dump[0])
	cdef vector[Finitely_critical_multi_filtration] deaths = Finitely_critical_multi_filtration.from_python(summand_dump[1])
	cdef int dim = summand_dump[2]
	return Summand(births,deaths,dim)

cdef dump_cmod(Module& mod):
	cbox = mod.get_box()
	cdef int dim = mod.get_dimension()
	bottom_corner = cbox.get_bottom_corner().get_vector()
	top_corner = cbox.get_upper_corner().get_vector()
	box = np.asarray([bottom_corner,top_corner])
	summands = tuple(dump_summand(summand) for summand in mod)
	return box, summands

cdef Module cmod_from_dump(module_dump):
	box = module_dump[0]
	summands = module_dump[1]
	cdef Module out_module = Module()
	out_module.set_box(Box[value_type](box))
	for i in range(len(summands)):
		out_module.add_summand(summand_from_dump(summands[i]))
	return out_module


def from_dump(dump)->PyModule:
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
	# TODO : optimize...
	mod = PyModule()
	if type(dump) is str:
		dump = pk.load(open(dump, "rb"))
	cdef Module cmod = cmod_from_dump(dump)
	mod.cmod = cmod
	return mod



################################################## MMA PLOTS




#################################################### DATASETS for test



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
	from multipers.synthetic_data import noisy_annulus
	points = noisy_annulus(**kwargs)
	points = np.unique(points, axis=0)
	from sklearn.neighbors import KernelDensity
	kde = KernelDensity(bandwidth = 1).fit(points)
	st = gd.RipsComplex(points = points).create_simplex_tree()
	st = gd.SimplexTreeMulti(st, parameters = 2)
	st.collapse_edges(full=True)
	f2 =  - kde.score_samples(points)
	st.fill_lowerstar(f2,parameter=1)
	mod = st.persistence(**kwargs)
	return mod

def nlines_precision_box(nlines, basepoint, scale, square = False):
	from random import choice, shuffle
	from sympy.ntheory import factorint
	h = scale
	dim = len(basepoint)
	basepoint = np.array(basepoint)
	if square:
		# here we want n^dim-1 lines (n = nlines)
		n=nlines
		basepoint = np.array(basepoint)
		deathpoint = basepoint.copy()
		deathpoint+=n*h + - h/2
		deathpoint[-1] = basepoint[-1]+h/2
		return [basepoint,deathpoint]
	factors = factorint(nlines)
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

##################################################### MMA CONVERSIONS





cdef extern from "multiparameter_module_approximation/format_python-cpp.h" namespace "Gudhi::multiparameter::mma":
	#list_simplicies_to_sparse_boundary_matrix
	#vector[vector[unsigned int]] build_sparse_boundary_matrix_from_simplex_list(vector[vector[unsigned int]] list_simplices)
	#list_simplices_ls_filtration_to_sparse_boundary_filtration
	#pair[vector[vector[unsigned int]], vector[vector[double]]] build_boundary_matrix_from_simplex_list(vector[vector[unsigned int]] list_simplices, vector[vector[double]] filtrations, vector[unsigned int] filters_to_permute)
	# pair[vector[vector[unsigned int]], vector[double]] simplextree_to_boundary_filtration(vector[boundary_type] &simplexList, filtration_type &filtration)
	pair[boundary_matrix, vector[Finitely_critical_multi_filtration]] simplextree_to_boundary_filtration(uintptr_t)

#	pair[vector[vector[unsigned int]], vector[double]] __old__simplextree_to_boundary_filtration(vector[boundary_type]&, filtration_type&)

#	string __old__simplextree2rivet(const uintptr_t, const vector[filtration_type]&)
	# void simplextree2rivet(const string&, const uintptr_t, const vector[filtration_type]&)

	

def simplex_tree2boundary_filtrations(simplextree:SimplexTreeMulti | SimplexTree):
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
	cdef intptr_t cptr
	if isinstance(simplextree, SimplexTreeMulti):
		cptr = simplextree.thisptr
	elif isinstance(simplextree, SimplexTree):
		temp_st = gd.SimplexTreeMulti(simplextree, parameters=1)
		cptr = temp_st.thisptr
	else:
		raise TypeError("Has to be a simplextree")
	cdef pair[boundary_matrix, vector[Finitely_critical_multi_filtration]] cboundary_filtration = simplextree_to_boundary_filtration(cptr)
	boundary = cboundary_filtration.first
	multi_filtrations = np.array(Finitely_critical_multi_filtration.to_python(cboundary_filtration.second))
	return boundary, multi_filtrations


# def simplextree_to_sparse_boundary(st:SimplexTree):
# 	return build_sparse_boundary_matrix_from_simplex_list([simplex[0] for simplex in st.get_simplices()])







############################################# MMA - Matching distances



