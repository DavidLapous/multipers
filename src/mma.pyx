from MMA_Classes cimport corner_type
from MMA_Classes cimport corner_list
from MMA_Classes cimport interval
from MMA_Classes cimport MultiDiagram_point
from MMA_Classes cimport Line
from MMA_Classes cimport Summand
from MMA_Classes cimport Box
from MMA_Classes cimport Module
from MMA_Classes cimport MultiDiagram
from MMA_Classes cimport MultiDiagrams

# from MMA_Classes cimport *

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

cdef extern from "mma_cpp/approximation.h" namespace "Vineyard":
	Module compute_vineyard_barcode_approximation(boundary_matrix, vector[vector[double]] , double precision, Box &box, bool threshold, bool complete, bool multithread, bool verbose) nogil

cdef class PySummand:
	"""
	Stores a Summand of a PyModule
	"""
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
		if len(self) == 0: return
		_cmap = get_cmap("Spectral")
		multibarcodes_, colors = self._get_plot_bars(degree, min_persistence)
		n_summands = np.max(colors)+1
		plt.rc('axes', prop_cycle = cycler('color', [_cmap(i/n_summands) for i in colors]))
		plt.plot(*multibarcodes_)
cdef class PyLine:
	cdef Line line

cdef class PyModule:
	"""
	Stores a representation of a n-persistence module.
	"""
	cdef Module cmod

	cdef set(self, Module m):
		self.cmod = m
	cdef set_box(self, Box box):
		self.cmod.set_box(box)
		return self
	def get_module_of_dimension(self, degree:int)->PyModule: # TODO : in c++ ?
		pmodule = PyModule()
		pmodule.set_box(self.cmod.get_box())
		for summand in self.cmod:
			if summand.get_dimension() == degree:
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
		out = [[] for _ in range(self.cmod.get_dimension()+1)]
		out += [self.get_box()]
		for summand in self.cmod:
			out[summand.get_dimension()].append([summand.get_birth_list(), summand.get_death_list()])
		if path is None:
			return out
		pk.dump(out, open(path, "wb"))
		return out
	def __getitem__(self, i:int) -> PySummand:
		summand = PySummand()
		if i>=0:
			summand.set(self.cmod.at(i))
		else:
			summand.set(self.cmod.at(self.cmod.size() - i))
		return summand
	
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
		plot2d(corners, box=box, dimension=degree, **kwargs)
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
		out.set(self.cmod.get_barcode(Line(basepoint), degree, threshold))
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
			print("Basepoints has to be specified for filtration dimension >= 3 !")
			return
		elif basepoints is None:
			h = box[1][1] - box[0][1]
			basepoints = np.linspace([box[0][0] - h,box[0][1]], [box[1][0],box[0][1]], num=num)
		out.set(self.cmod.get_barcodes(basepoints, degree, threshold))
		return out
	def landscape(self, degree:int, k:int=0,box:list|np.ndarray|None=None, resolution:List=[100,100], plot=True):
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
		out = np.array(self.cmod.get_landscape(degree, k, Box(box), resolution))
		if plot:
			plt.figure()
			aspect = (box[1][0]-box[0][0]) / (box[1][1]-box[0][1])
			extent = [box[0][0], box[1][0], box[0][1], box[1][1]]
			plt.imshow(out.T, origin="lower", extent=extent, aspect=aspect)
		return out
	def landscapes(self, degree:int, ks:list|np.ndarray=[0],box=None, resolution:list|np.ndarray=[100,100], plot=True):
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
		out = np.array(self.cmod.get_landscapes(degree, ks, Box(box), resolution))
		if plot:
			to_plot = np.sum(out, axis=0)
			plt.figure()
			aspect = (box[1][0]-box[0][0]) / (box[1][1]-box[0][1])
			extent = [box[0][0], box[1][0], box[0][1], box[1][1]]
			plt.imshow(to_plot.T, origin="lower", extent=extent, aspect=aspect)
		return out


	def image(self, degree:int = -1, bandwidth:float=0.1, resolution:list=[100,100], normalize:bool=False, plot:bool=True, save:bool=False, dpi:int=200,p:float=1., **kwargs)->np.ndarray:
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
		plot = True : Boolean
			If true, plots the images;

		Returns
		-------
		List of Numpy arrays or numpy array
			The list of images, or the image of fixed dimension.
		"""
		if (len(self.get_bottom()) != 2):
			print("Non 2 dimensional images not yet implemented in python !")
			return np.zeros(shape=resolution)
		box = kwargs.get("box",[self.get_bottom(),self.get_top()])
		if degree < 0:
			image_vector = np.array(self.cmod.get_vectorization(bandwidth, p, normalize, Box(box), resolution[0], resolution[1]))
		else:
			image_vector = np.array([self.cmod.get_vectorization_in_dimension(degree, bandwidth, p,normalize,Box(box),  resolution[0], resolution[1])])
		if plot:
			i=0
			n_plots = len(image_vector)
			scale:float = kwargs.get("size", 4.0)
			fig, axs = plt.subplots(1,n_plots, figsize=(n_plots*scale,scale))
			aspect = (box[1][0]-box[0][0]) / (box[1][1]-box[0][1])
			extent = [box[0][0], box[1][0], box[0][1], box[1][1]]
			for image in image_vector:
				ax = axs if n_plots <= 1 else axs[i]
				temp = ax.imshow(np.flip(np.array(image).transpose(),0),extent=extent, aspect=aspect)
				if (kwargs.get('colorbar') or kwargs.get('cb')):
					plt.colorbar(temp, ax = ax)
				if degree < 0 :
					ax.set_title(rf"$H_{i}$ $2$-persistence image")
				if degree >= 0:
					ax.set_title(rf"$H_{degree}$ $2$-persistence image")
				i+=1

		return image_vector[0] if degree >=0 else  image_vector


	def euler_char(self, points:list | list| np.ndarray) -> np.ndarray:
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
		cdef vector[vector[double]] c_points = points
		cdef Module c_mod = self.cmod
		with nogil:
			c_euler = c_mod.euler_curve(c_points)
		euler = c_euler
		return np.array(euler, dtype=int)
def approx(
	st = None,
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
	**kwargs)->PyModule:
	"""Computes an interval module approximation of a multiparameter filtration.

	Parameters
	----------
	st : n-filtered Simplextree.
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
	Returns
	-------
	PyModule
		An interval decomposable module approximation of the module defined by the
		homology of this multi-filtration.
	"""

	if boundary is None or filtration is None:
		boundary,filtration = splx2bf(st) # TODO : recomputed each time... maybe store this somewhere ?
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
		st = st.to_gudhi(0)
		return st.persistence()

	if box is None and not(st is None):
		m,M = st.filtration_bounds()
	else:
		m,M = box
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
	cdef boundary_matrix c_boundary = boundary
	cdef vector[vector[double]] c_filtration = filtration
	cdef double c_max_error = max_error
	cdef bool c_threshold = threshold
	cdef bool c_complete = complete
	cdef bool c_multithread = multithread
	cdef bool c_verbose = verbose
	cdef Box c_box = Box(box)
	with nogil:
		c_mod = compute_vineyard_barcode_approximation(c_boundary,c_filtration,c_max_error, c_box, c_threshold, c_complete, c_multithread,c_verbose)
	approx_mod.set(c_mod)
	return approx_mod


cdef set_from_dump(box, summands):
	mod = PyModule()
	mod.cmod.set_box(Box(box))
	for dim,summands_corners in enumerate(summands):
		for births, deaths in summands_corners:
			mod.cmod.add_summand(Summand(births, deaths, dim))
	return mod 
	

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
	if type(dump) is str:
		dump = pk.load(open(dump, "rb"))
	mod = set_from_dump(dump[-1], dump[:-1])
	# with nogil:
	# 	mod.cmod.set_box(Box(dump[-1]))
	# 	for dim,summands in enumerate(dump[:-1]):
	# 		for summand in summands:
	# 			mod.cmod.add_summand(Summand(summand[0], summand[1], dim))
	return mod



