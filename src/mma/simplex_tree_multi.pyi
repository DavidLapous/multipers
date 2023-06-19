import numpy as np
filtration_type = np.ndarray
dimension_type = int
from mma.multiparameter_module_approximation import PyModule
from typing import Iterable
from gudhi.simplex_tree import SimplexTree


# SimplexTree python interface
class SimplexTreeMulti:
	"""The simplex tree is an efficient and flexible data structure for
	representing general (filtered) simplicial complexes. The data structure
	is described in Jean-Daniel Boissonnat and Clément Maria. The Simplex
	Tree: An Efficient Data Structure for General Simplicial Complexes.
	Algorithmica, pages 1–22, 2014.

	This class is a multi-filtered, with keys, and non contiguous vertices version
	of the simplex tree. 
	"""
	
	def __init__(self, other = None, num_parameters:int=2):
		"""SimplexTreeMulti constructor.
		
		:param other: If `other` is `None` (default value), an empty `SimplexTreeMulti` is created.
			If `other` is a `SimplexTree`, the `SimplexTreeMulti` is constructed from a deep copy of `other`.
			If `other` is a `SimplexTreeMulti`, the `SimplexTreeMulti` is constructed from a deep copy of `other`.
		:type other: SimplexTree or SimplexTreeMulti (Optional)
		:param num_parameters: The number of parameter of the multi-parameter filtration.
		:type num_parameters: int
		:returns: An empty or a copy simplex tree.
		:rtype: SimplexTreeMulti

		:raises TypeError: In case `other` is neither `None`, nor a `SimplexTree`, nor a `SimplexTreeMulti`.
		"""
		...

	def copy(self)->SimplexTreeMulti:
		"""
		:returns: A simplex tree that is a deep copy of itself.
		:rtype: SimplexTreeMulti

		:note: The persistence information is not copied. If you need it in the clone, you have to call
			:func:`compute_persistence` on it even if you had already computed it in the original.
		"""
		...

	def __deepcopy__(self)->SimplexTreeMulti:
		...

	def filtration(self, simplex:list|np.ndarray)->filtration_type:
		"""This function returns the filtration value for a given N-simplex in
		this simplicial complex, or +infinity if it is not in the complex.

		:param simplex: The N-simplex, represented by a list of vertex.
		:type simplex: list of int
		:returns:  The simplicial complex multi-critical filtration value.
		:rtype:  numpy array of shape (-1, num_parameters)
		"""
		...
	def assign_filtration(self, simplex:list|np.ndarray, filtration:list|np.ndarray)->None:
		"""This function assigns a new multi-critical filtration value to a
		given N-simplex.

		:param simplex: The N-simplex, represented by a list of vertex.
		:type simplex: list of int
		:param filtration:  The new filtration(s) value(s), concatenated.
		:type filtration:  list[float] or np.ndarray[float, ndim=1]

		.. note::
			Beware that after this operation, the structure may not be a valid
			filtration anymore, a simplex could have a lower filtration value
			than one of its faces. Callers are responsible for fixing this
			(with more :meth:`assign_filtration` or
			:meth:`make_filtration_non_decreasing` for instance) before calling
			any function that relies on the filtration property, like
			:meth:`persistence`.
		"""
		...

	
	def num_vertices(self)->int:
		"""This function returns the number of vertices of the simplicial
		complex.

		:returns:  The simplicial complex number of vertices.
		:rtype:  int
		"""
		...
	def num_simplices(self)->int:
		"""This function returns the number of simplices of the simplicial
		complex.

		:returns:  the simplicial complex number of simplices.
		:rtype:  int
		"""
		...

	def dimension(self)->dimension_type:
		"""This function returns the dimension of the simplicial complex.

		:returns:  the simplicial complex dimension.
		:rtype:  int

		.. note::

			This function is not constant time because it can recompute
			dimension if required (can be triggered by
			:func:`remove_maximal_simplex`
			or
			:func:`prune_above_filtration`
			methods).
		"""
		...
	def upper_bound_dimension(self)->dimension_type:
		"""This function returns a valid dimension upper bound of the
		simplicial complex.

		:returns:  an upper bound on the dimension of the simplicial complex.
		:rtype:  int
		"""
		...

	def set_dimension(self, dimension)->None:
		"""This function sets the dimension of the simplicial complex.

		:param dimension: The new dimension value.
		:type dimension: int

		.. note::

			This function must be used with caution because it disables
			dimension recomputation when required
			(this recomputation can be triggered by
			:func:`remove_maximal_simplex`
			or
			:func:`prune_above_filtration`
			).
		"""
		...

	def find(self, simplex)->bool:
		"""This function returns if the N-simplex was found in the simplicial
		complex or not.

		:param simplex: The N-simplex to find, represented by a list of vertex.
		:type simplex: list of int
		:returns:  true if the simplex was found, false otherwise.
		:rtype:  bool
		"""
		...

	def insert(self, simplex, filtration:list|np.ndarray|None=None)->bool:
		"""This function inserts the given N-simplex and its subfaces with the
		given filtration value (default value is '0.0'). If some of those
		simplices are already present with a higher filtration value, their
		filtration value is lowered.

		:param simplex: The N-simplex to insert, represented by a list of
			vertex.
		:type simplex: list of int
		:param filtration: The filtration value of the simplex.
		:type filtration: float
		:returns:  true if the simplex was not yet in the complex, false
			otherwise (whatever its original filtration value).
		:rtype:  bool
		"""
		...
		
	def insert_batch(self, vertex_array:np.ndarray, filtrations:np.array):
		"""Inserts k-simplices given by a sparse array in a format similar
		to `torch.sparse <https://pytorch.org/docs/stable/sparse.html>`_.
		The n-th simplex has vertices `vertex_array[0,n]`, ...,
		`vertex_array[k,n]` and filtration value `filtrations[n,num_parameters]`.
		/!\ Only compatible with 1-critical filtrations. If a simplex is repeated, 
		only one filtration value will be taken into account.

		:param vertex_array: the k-simplices to insert.
		:type vertex_array: numpy.array of shape (k+1,n)
		:param filtrations: the filtration values.
		:type filtrations: numpy.array of shape (n,num_parameters)
		"""
		...

	def get_simplices(self):
		"""This function returns a generator with simplices and their given
		filtration values.

		:returns:  The simplices.
		:rtype:  generator with tuples(simplex, filtration)
		"""
		...

	def get_filtration(self):
		"""This function returns a generator with simplices and their given
		filtration values sorted by increasing filtration values.

		:returns:  The simplices sorted by increasing filtration values.
		:rtype:  generator with tuples(simplex, filtration)
		"""
		...

	def get_skeleton(self, dimension):
		"""This function returns a generator with the (simplices of the) skeleton of a maximum given dimension.

		:param dimension: The skeleton dimension value.
		:type dimension: int
		:returns:  The (simplices of the) skeleton of a maximum dimension.
		:rtype:  generator with tuples(simplex, filtration)
		"""
		...

	def get_star(self, simplex):
		"""This function returns the star of a given N-simplex.

		:param simplex: The N-simplex, represented by a list of vertex.
		:type simplex: list of int
		:returns:  The (simplices of the) star of a simplex.
		:rtype:  list of tuples(simplex, filtration)
		"""
		...

	def get_cofaces(self, simplex, codimension):
		"""This function returns the cofaces of a given N-simplex with a
		given codimension.

		:param simplex: The N-simplex, represented by a list of vertex.
		:type simplex: list of int
		:param codimension: The codimension. If codimension = 0, all cofaces
			are returned (equivalent of get_star function)
		:type codimension: int
		:returns:  The (simplices of the) cofaces of a simplex
		:rtype:  list of tuples(simplex, filtration)
		"""
		...

	def get_boundaries(self, simplex):
		"""This function returns a generator with the boundaries of a given N-simplex.
		If you do not need the filtration values, the boundary can also be obtained as
		:code:`itertools.combinations(simplex,len(simplex)-1)`.

		:param simplex: The N-simplex, represented by a list of vertex.
		:type simplex: list of int.
		:returns:  The (simplices of the) boundary of a simplex
		:rtype:  generator with tuples(simplex, filtration)
		"""
		...

	def remove_maximal_simplex(self, simplex):
		"""This function removes a given maximal N-simplex from the simplicial
		complex.

		:param simplex: The N-simplex, represented by a list of vertex.
		:type simplex: list of int

		.. note::

			The dimension of the simplicial complex may be lower after calling
			remove_maximal_simplex than it was before. However,
			:func:`upper_bound_dimension`
			method will return the old value, which
			remains a valid upper bound. If you care, you can call
			:func:`dimension`
			to recompute the exact dimension.
		"""
		...


	def expansion(self, max_dim)->SimplexTreeMulti:
		"""Expands the simplex tree containing only its one skeleton
		until dimension max_dim.

		The expanded simplicial complex until dimension :math:`d`
		attached to a graph :math:`G` is the maximal simplicial complex of
		dimension at most :math:`d` admitting the graph :math:`G` as
		:math:`1`-skeleton.
		The filtration value assigned to a simplex is the maximal filtration
		value of one of its edges.

		The simplex tree must contain no simplex of dimension bigger than
		1 when calling the method.

		:param max_dim: The maximal dimension.
		:type max_dim: int
		"""
		...

	def make_filtration_non_decreasing(self)->bool: # FIXME TODO code in c++
		"""This function ensures that each simplex has a higher filtration
		value than its faces by increasing the filtration values.

		:returns: True if any filtration value was modified,
			False if the filtration was already non-decreasing.
		:rtype: bool
		"""
		...
		
	def reset_filtration(self, filtration, min_dim = 0):
		"""This function resets the filtration value of all the simplices of dimension at least min_dim. Resets all the
		simplex tree when `min_dim = 0`.
		`reset_filtration` may break the filtration property with `min_dim > 0`, and it is the user's responsibility to
		make it a valid filtration (using a large enough `filt_value`, or calling `make_filtration_non_decreasing`
		afterwards for instance).

		:param filtration: New threshold value.
		:type filtration: float.
		:param min_dim: The minimal dimension. Default value is 0.
		:type min_dim: int.
		"""
		...

	def expansion_with_blocker(self, max_dim, blocker_func):
		"""Expands the Simplex_tree containing only a graph. Simplices corresponding to cliques in the graph are added
		incrementally, faces before cofaces, unless the simplex has dimension larger than `max_dim` or `blocker_func`
		returns `True` for this simplex.

		The function identifies a candidate simplex whose faces are all already in the complex, inserts it with a
		filtration value corresponding to the maximum of the filtration values of the faces, then calls `blocker_func`
		with this new simplex (represented as a list of int). If `blocker_func` returns `True`, the simplex is removed,
		otherwise it is kept. The algorithm then proceeds with the next candidate.

		.. warning::
			Several candidates of the same dimension may be inserted simultaneously before calling `blocker_func`, so
			if you examine the complex in `blocker_func`, you may hit a few simplices of the same dimension that have
			not been vetted by `blocker_func` yet, or have already been rejected but not yet removed.

		:param max_dim: Expansion maximal dimension value.
		:type max_dim: int
		:param blocker_func: Blocker oracle.
		:type blocker_func: Callable[[List[int]], bool]
		"""
		...

		
	def persistence_approximation(self, **kwargs)->PyModule:
		"""Computes an interval module approximation of a multiparameter filtration.

		Parameters
		----------
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
			When true, computes the module restricted to the box.
		max_dimension:int
			Max simplextree dimension to consider. 
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
		
		
## This function is only meant for the edge collapse interface.
	def get_edge_list(self):
		...
	
	def collapse_edges(self, max_dimension:int=None, num:int=1, progress:bool=False, strong:bool=True, full:bool=False, ignore_warning:bool=False)->SimplexTreeMulti:
		"""Edge collapse for 1-critical 2-parameter clique complex (see https://arxiv.org/abs/2211.05574).
		It uses the code from the github repository https://github.com/aj-alonso/filtration_domination .

		Parameters
		----------
		max_dimension:int
			Max simplicial dimension of the complex. Unless specified, keeps the same dimension.
		num:int
			The number of collapses to do.
		strong:bool
			Whether to use strong collapses or standard collapses (slower, but may remove more edges)
		full:bool
			Collapses the maximum number of edges if true, i.e., will do (at most) 100 strong collapses and (at most) 100 non-strong collapses afterward.
		progress:bool
			If true, shows the progress of the number of collapses.

		WARNING
		-------
			- This will destroy all of the k-simplices, with k>=2. Be sure to use this with a clique complex, if you want to preserve the homology >= dimension 1.
			- This is for 1 critical simplices, with 2 parameter persistence.
		Returns
		-------
		self:SimplexTreeMulti
			A (smaller) simplex tree that has the same homology over this bifiltration.

		"""
		...
	def _reconstruct_from_edge_list(self, edges, swap:bool=True, expand_dimension:int=None):
		"""
		Generates a 1-dimensional copy of self, with the edges given as input. Useful for edge collapses

		Input
		-----
		 - edges : Iterable[(int,int),(float,float)] ## This is the format of the rust library filtration-domination
		 - swap : bool
		 	If true, will swap self and the collapsed simplextrees.
		 - expand_dim : int
		 	expands back the simplextree to this dimension
		Ouput
		-----
		The reduced SimplexTreeMulti having only these edges.
		"""
		...
	@property
	def num_parameters(self)->int:
		...
	def get_simplices_of_dimension(self, dim:int):
		...
	def key(self, simplex:list|np.ndarray):
		...
	def set_keys_to_enumerate(self)->None:
		...
	def set_key(self,simplex:list|np.ndarray, key:int)->None:
		...
	
	
	def to_scc(self, path="scc_dataset.txt", progress:bool=True, overwrite:bool=False, ignore_last_generators:bool=True, strip_comments:bool=False, reverse_block:bool=True, rivet_compatible=False)->None:
		""" Create a file with the scc2020 standard, representing the n-filtration of the simplextree.
		Link : https://bitbucket.org/mkerber/chain_complex_format/src/master/

		Parameters
		----------
		path:str
			path of the file.
		ignore_last_generators:bool = True
			If false, will include the filtration values of the last free persistence module.
		progress:bool = True
			Shows the progress bar.
		overwrite:bool = False
			If true, will overwrite the previous file if it already exists.
		ignore_last_generators:bool=True
			If true, does not write the final generators to the file. Rivet ignores them.
		reverse_block:bool=True
			Some obscure programs reverse the inside-block order.
		rivet_compatible:bool=False
			Returns a firep (old scc2020) format instead. Only Rivet uses this.

		Returns
		-------
		Nothing
		"""
		...
	def to_rivet(self, path="rivet_dataset.txt", degree:int|None = None, progress:bool=False, overwrite:bool=False, xbins:int|None=None, ybins:int|None=None)->None:
		""" Create a file that can be imported by rivet, representing the filtration of the simplextree.

		Parameters
		----------
		path:str
			path of the file.
		degree:int
			The homological degree to ask rivet to compute.
		progress:bool = True
			Shows the progress bar.
		overwrite:bool = False
			If true, will overwrite the previous file if it already exists.
		Returns
		-------
		Nothing
		"""
		...



	def _get_filtration_values(self, degrees:Iterable[int], inf_to_nan:bool=False):
		...
	
	def get_filtration_grid(self, resolution:Iterable[int]|int|None=None, degrees:Iterable[int]|None=None, q:float=0., grid_strategy:str="regular")->Iterable[np.ndarray]:
		"""
		Returns a grid over the n-filtration, from the simplextree. Usefull for grid_squeeze. TODO : multicritical

		Parameters
		----------
			resolution: list[int]
				resolution of the grid, for each parameter
			box=None : pair[list[float]]
				Grid bounds. format : [low bound, high bound]
				If None is given, will use the filtration bounds of the simplextree.
			grid_strategy="regular" : string
				Either "regular", "quantile", or "exact".
		Returns
		-------
			List of filtration values, for each parameter, defining the grid.
		"""
		...
	

	def grid_squeeze(self, filtration_grid:np.ndarray|list|None=None, coordinate_values:bool=False)->SimplexTreeMulti:
		"""
		Fit the filtration of the simplextree to a grid.
		
		:param filtration_grid: The grid on which to squeeze. An example of grid can be given by the `get_filtration_grid` method.
		:type filtration_grid: list[list[float]]
		:param coordinate_values: If true, the filtrations values of the simplices will be set to the coordinate of the filtration grid.
		:type coordinate_values: bool
		"""
		...

	def filtration_bounds(self, degrees:Iterable[int]|None=None, q:float=0, split_dimension:bool=False)->np.ndarray:
		"""
		Returns the filtrations bounds of the finite filtration values.
		"""
		...


	

	def fill_lowerstar(self, F, parameter:int)->SimplexTreeMulti:
		""" Fills the `dimension`th filtration by the lower-star filtration defined by F.

		Parameters
		----------
		F:1d array
			The density over the vertices, that induces a lowerstar filtration.
		parameter:int
			Which filtration parameter to fill. /!\ python starts at 0.

		Returns
		-------
		self:SimplexTreeMulti
		"""
		...


	def project_on_line(self, parameter:int=0, basepoint:None|list|np.ndarray= None)-> SimplexTree:
		"""Converts an multi simplextree to a gudhi simplextree.
		Parameters
		----------
			parameter:int = 0
				The parameter to keep. WARNING will crash if the multi simplextree is not well filled.
			basepoint:None
				Instead of keeping a single parameter, will consider the filtration defined by the diagonal line crossing the basepoint.
		WARNING 
		-------
			There are no safeguard yet, it WILL crash if asking for a parameter that is not filled.
		Returns
		-------
			A SimplexTree with chosen 1D filtration.
		"""
		...

	def set_num_parameter(self, num:int)->None:
		"""
		Sets the numbers of parameters. 
		WARNING : it will resize all the filtrations to this size. 
		"""
		...

	def __eq__(self, other:SimplexTreeMulti)->bool:
		"""Test for structural equality
		:returns: True if the 2 simplex trees are equal, False otherwise.
		:rtype: bool
		"""
		...
	


def _collapse_edge_list(edges, num:int=0, full:bool=False, strong:bool=False):
	"""
	Given an edge list defining a 1 critical 2 parameter 1 dimensional simplicial complex, simplificates this filtered simplicial complex, using filtration-domination's edge collapser.
	"""
	...



def _simplextree_multify(simplextree:SimplexTree, num_parameters:int=2)->SimplexTreeMulti:
	"""Converts a gudhi simplextree to a multi simplextree.
	Parameters
	----------
		parameters:int = 2
			The number of filtrations
	Returns
	-------
		A multi simplextree, with first filtration value being the one from the original simplextree.
	"""
	...