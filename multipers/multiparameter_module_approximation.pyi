
#########################################################################
## Small hack for typing
from multipers.simplex_tree_multi import SimplexTreeMulti
import numpy as np


class PySummand:
	"""
	Stores a Summand of a PyModule
	"""


	def get_birth_list(self): # TODO: FIXME
		...
	def get_death_list(self):
		...
	@property
	def num_parameters(self)->int:
		...


class PyBox:
	@property
	def num_parameters(self)->int:
		...
	def contains(self, x)->bool:
		...


	def get(self):
		...
	def to_multipers(self):
		...

class PyMultiDiagramPoint:
	def get_degree(self):
		...
	def get_birth(self):
		...
	def get_death(self):
		...

class PyMultiDiagram:
	"""
	Stores the diagram of a PyModule on a line
	"""
	def get_points(self, degree:int=-1) -> np.ndarray:
		...
	def to_multipers(self, dimension:int):
		...
	def __len__(self) -> int:
		...
	def __getitem__(self,i:int) -> PyMultiDiagramPoint:
		...
class PyMultiDiagrams:
	"""
	Stores the barcodes of a PyModule on multiple lines
	"""

	def to_multipers(self):
		...
	def __getitem__(self,i:int):
		...
	def __len__(self):
		...
	def get_points(self, degree:int=-1):
		...

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
		...

class PyModule:
	"""
	Stores a representation of a n-persistence module.
	"""
	def get_module_of_dimension(self, degree:int)->PyModule: # TODO : in c++ ?
		...

	def __len__(self)->int:
		...
	def get_bottom(self):
		...
	def get_top(self):
		...
	def get_box(self):
		...
	@property
	def num_parameters(self)->int:
		...
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
		...
	def __getitem__(self, i:int) -> PySummand:
		...
	
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
		...

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
		...
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
		...
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
		...
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
		...


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
		...


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
		...

def module_approximation(
	st:SimplexTreeMulti,
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

	...

	

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
	...


#################################################### DATASETS for test

def noisy_annulus(r1:float=1, r2:float=2, n1:int=1000,n2:int=200, dim:int=2, center:np.ndarray|list|None=None, **kwargs)->np.ndarray:
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
	...

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
	...


##################################################### MMA CONVERSIONS






############################################# MMA - Matching distances





def estimate_matching(b1:PyMultiDiagrams, b2:PyMultiDiagrams)->float:
	...


#### Functions to estimate precision
def estimate_error(st:SimplexTreeMulti, module:PyModule, degree:int, nlines = 100, verbose:bool =False)->float:
	"""
	Given an MMA SimplexTree and PyModule, estimates the bottleneck distance using barcodes given by gudhi.
	
	Parameters
	----------
	st:SimplexTree
		The simplextree representing the n-filtered complex. Used to define the gudhi simplextrees on different lines.
	module:PyModule
		The module on which to estimate approximation error, w.r.t. the original simplextree st.
	degree: The homology degree to consider
	
	Returns
	-------
	The estimation of the matching distance, i.e., the maximum of the sampled bottleneck distances.
		
	"""
	...



