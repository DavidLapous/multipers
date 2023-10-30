
from typing import Iterable, Optional

from itertools import product
import matplotlib.pyplot as plt
from multipers.ml.convolutions import convolution_signed_measures
import numpy as np
from joblib import Parallel, delayed 
from sklearn.base import BaseEstimator, TransformerMixin
from tqdm import tqdm

import multipers as mp
from multipers.simplex_tree_multi import SimplexTreeMulti

reduce_grid = SimplexTreeMulti._reduce_grid


class SimplexTree2SignedMeasure(BaseEstimator,TransformerMixin):
    """
    Input
    -----
    Iterable[SimplexTreeMulti]

    Output
    ------
    Iterable[ list[signed_measure for degree] ]

    signed measure is either 
        - (points : (n x num_parameters) array, weights : (n) int array ) if sparse, 
        - else an integer matrix.

    Parameters
    ----------
        - degrees : list of degrees to compute. None correspond to the euler characteristic
        - filtration grid : the grid on which to compute. 
        If None, the fit will infer it from
        - fit_fraction : the fraction of data to consider for the fit, seed is controlled by the seed parameter
        - resolution : the resolution of this grid
        - filtration_quantile : filtrations values quantile to ignore
        - grid_strategy:str : 'regular' or 'quantile' or 'exact'
        - normalize filtration : if sparse, will normalize all filtrations.
        - expand : expands the simplextree to compute correctly the degree, for
        flag complexes
        - invariant : the topological invariant to produce the signed measure.
        Choices are "hilbert" or "euler". Will add rank invariant later.
        - num_collapse : Either an int or "full". Collapse the complex before
        doing computation.
        - _möbius_inversion : if False, will not do the mobius inversion. output
        has to be a matrix then.
        - enforce_null_mass : Returns a zero mass measure, by thresholding the
        module if True.
    """
    def __init__(self, 
            degrees:list[int|None]|None=[], # homological degrees + None for euler
            rank_degrees:list[int]=[], # same for rank invariant
            filtration_grid:Iterable[np.ndarray]|None=None, # filtration values to consider. Format : [ filtration values of Fi for Fi:filtration values of parameter i] 
            progress=False, # tqdm
            num_collapses:int|str=0, # edge collapses before computing 
            n_jobs=None, 
            resolution:Iterable[int]|int|None=None, # when filtration grid is not given, the resolution of the filtration grid to infer
            # sparse=True, # sparse output # DEPRECATED TO Ssigned measure formatter
            plot:bool=False, 
            filtration_quantile:float=0., # quantile for inferring filtration grid
            _möbius_inversion:bool=True, # wether or not to do the möbius inversion (not recommended to touch)
            expand=True, # expand the simplextree befoe computing the homology
            normalize_filtrations:bool=False,
            # exact_computation:bool=False, # compute the exact signed measure.
            grid_strategy:str='exact',
            seed:int=0, # if fit_fraction is not 1, the seed sampling
            fit_fraction = 1,  # the fraction of the data on which to fit
            out_resolution:Iterable[int]|int|None=None,
            individual_grid:Optional[bool]=None, # Can be significantly faster for some grid strategies, but can drop statistical performance
            enforce_null_mass:bool=False,
            flatten=True,
		):
        super().__init__()
        self.degrees = degrees
        self.rank_degrees = rank_degrees
        self.filtration_grid = filtration_grid
        self.progress = progress
        self.num_collapses=num_collapses
        self.n_jobs = n_jobs
        self.resolution = np.inf if grid_strategy == 'exact' and resolution is None else 100
        self.plot=plot
        # self.sparse=sparse # TODO : deprecate
        self.filtration_quantile=filtration_quantile
        self.normalize_filtrations = normalize_filtrations # Will only work for non sparse output. (discrete matrices cannot be "rescaled")
        self.grid_strategy = grid_strategy
        self.num_parameter = None
        self._is_input_delayed = None
        self._möbius_inversion = _möbius_inversion
        self._reconversion_grid = None
        self.expand = expand
        # will only refit the grid if filtration_grid has never been given.
        self._refit_grid = None
        self.seed=seed
        self.fit_fraction = fit_fraction
        self._transform_st = None
        self._to_simplex_tree = None
        self.out_resolution = out_resolution
        self.individual_grid = individual_grid
        self.enforce_null_mass = enforce_null_mass
        self._default_mass_location=None
        self.flatten=flatten
        return
    def _infer_filtration(self,X):
        indices = np.random.choice(
            len(X), min(int(self.fit_fraction* len(X)) +1, len(X)), 
            replace=False
        )
        get_st_filtration = lambda x : self._to_simplex_tree(x).get_filtration_grid(grid_strategy="exact")
        filtrations =  Parallel(n_jobs=self.n_jobs, backend="threading")(
            delayed(get_st_filtration)(x) for x in (X[idx] for idx in indices)
        )
        num_parameters = len(filtrations[0])
        filtrations_values = [np.unique(np.concatenate([x[i] for x in filtrations])) for i in range(num_parameters)]
        filtration_grid = reduce_grid(filtrations_values, resolutions = self.resolution, strategy = self.grid_strategy) # TODO :use more parameters
        self.filtration_grid=filtration_grid
        return filtration_grid

    def fit(self, X, y=None): # Todo : infer filtration grid ? quantiles ?
        # assert not self.normalize_filtrations or self.sparse, "Not able to normalize a matrix without losing information."
        assert self.resolution is not None or self.filtration_grid is not None or self.grid_strategy == "exact" or self.individual_grid, 'For non exact filtrations, a resolution has to be specified.'
        assert self._möbius_inversion or not self.individual_grid, "The grid has to be aligned when not using mobius inversion; disable individual_grid or enable mobius inversion."
        # assert self.invariant != "_" or self._möbius_inversion
        self._is_input_delayed = not isinstance(X[0], mp.SimplexTreeMulti)
        if self._is_input_delayed:
            from multipers.ml.tools import get_simplex_tree_from_delayed
            self._to_simplex_tree = get_simplex_tree_from_delayed
        else:
            self._to_simplex_tree = lambda x : x
        if isinstance(self.resolution, int) or self.resolution == np.inf:
            self.resolution = [self.resolution]*self._to_simplex_tree(X[0]).num_parameters
        self.num_parameter = len(self.filtration_grid) if self.resolution is None else len(self.resolution)

        self.individual_grid = self.individual_grid if self.individual_grid is not None else self.grid_strategy in ["regular_closest", "exact", "quantile", "partition"]
        
        if not self.enforce_null_mass and self.individual_grid or self.filtration_grid is not None:	
            self._refit_grid = False
        else: self._refit_grid = True

        if self._refit_grid:
            self._infer_filtration(X=X)
        if self.out_resolution is None:
            self.out_resolution = self.resolution
        elif isinstance(self.out_resolution, int):
            self.out_resolution = [self.out_resolution]*len(self.resolution)
        
        if self.normalize_filtrations and not self.individual_grid:
            # self._reconversion_grid = [np.linspace(0,1, num=len(f), dtype=float) for f in self.filtration_grid] ## This will not work for non-regular grids...
            self._reconversion_grid = [f/np.std(f) for f in self.filtration_grid] # not the best, but better than some weird magic
        # elif not self.sparse: # It actually renormalizes the filtration !!  
        #		self._reconversion_grid = [np.linspace(0,r, num=r, dtype=int) for r in self.out_resolution] 
        else:
            self._reconversion_grid = self.filtration_grid
        self._default_mass_location = [g[-1] for g in self._reconversion_grid] if self.enforce_null_mass else None
        return self

    def transform1(self, simplextree, filtration_grid=None, _reconversion_grid=None):
        if filtration_grid is None: filtration_grid = self.filtration_grid
        if _reconversion_grid is None: _reconversion_grid = self._reconversion_grid
        st = self._to_simplex_tree(simplextree)
        st = mp.SimplexTreeMulti(st, num_parameters = st.num_parameters) ## COPY
        if self.individual_grid:
            filtration_grid = st.get_filtration_grid(grid_strategy=self.grid_strategy, resolution=self.resolution)
            if self.enforce_null_mass:
                filtration_grid = [np.concatenate([f,[d]], axis=0) for f,d in zip(filtration_grid, self._default_mass_location)]
        st.grid_squeeze(filtration_grid = filtration_grid, coordinate_values = True)
        if st.num_parameters == 2:
            if self.num_collapses == "full":
                st.collapse_edges(full=True,max_dimension=1)
            elif isinstance(self.num_collapses, int):
                st.collapse_edges(num=self.num_collapses,max_dimension=1)
            else:
                raise Exception("Bad edge collapse type. either 'full' or an int.")
        int_degrees = np.asarray([d for d in self.degrees if d is not None])
        if self._möbius_inversion:
            ## EULER. First as there is prune above dimension below
            if self.expand and None in self.degrees:
                st.expansion(st.num_vertices)
            signed_measures_euler = mp.signed_measure(
                    simplextree=st,degrees=[None],
                    plot=self.plot,
                    mass_default = self._default_mass_location,
                    invariant="euler",
            )[0] if None in self.degrees else []

            if self.expand and len(int_degrees) > 0:	
                st.expansion(np.max(int_degrees)+1)
            if len(int_degrees) > 0:
                st.prune_above_dimension(np.max(np.concatenate([int_degrees, self.rank_degrees]))+1) ## no need to compute homology beyond this
            signed_measures_pers = mp.signed_measure(
                    simplextree=st,degrees=int_degrees,
                    mass_default = self._default_mass_location,
                    plot=self.plot,
                    invariant="hilbert",
            ) if len(int_degrees) >0 else []
            if self.plot:
                plt.show()
            if self.expand and len(self.rank_degrees) > 0 :	
                st.expansion(np.max(self.rank_degrees)+1)
            if len(self.rank_degrees) > 0:
                st.prune_above_dimension(np.max(self.rank_degrees)+1) ## no need to compute homology beyond this
            signed_measures_rank = mp.signed_measure(
                    simplextree=st,degrees=self.rank_degrees,
                    mass_default = self._default_mass_location,
                    plot=self.plot,
                    invariant="rank",
            ) if len(self.rank_degrees) >0 else []
            if self.plot:
                plt.show()
            
        else:
            from multipers.euler_characteristic import euler_surface
            from multipers.hilbert_function import hilbert_surface
            from multipers.rank_invariant import rank_invariant

            if self.expand and None in self.degrees:	st.expansion(st.num_vertices)
            signed_measures_euler = euler_surface(
                    simplextree=st,
                    plot=self.plot,
            )[1][None] if None in self.degrees else []
            
            if self.expand and len(int_degrees) > 0:	
                st.expansion(np.max(int_degrees)+1)
            if len(int_degrees) > 0:
                st.prune_above_dimension(np.max(np.concatenate([int_degrees, self.rank_degrees]))+1) 
                ## no need to compute homology beyond this
            signed_measures_pers = hilbert_surface(
                    simplextree=st,degrees=int_degrees,
                    plot=self.plot,
            )[1] if len(int_degrees) >0 else []
            if self.plot:	plt.show()

            if self.expand and len(self.rank_degrees) > 0 :	
                st.expansion(np.max(self.rank_degrees)+1)
            if len(self.rank_degrees) > 0:
                st.prune_above_dimension(np.max(self.rank_degrees)+1) ## no need to compute homology beyond this
            signed_measures_rank = rank_invariant(
                    simplextree=st,degrees=self.rank_degrees,
                    plot=self.plot,
            ) if len(self.rank_degrees) >0 else []

        
        count = 0
        signed_measures = []
        for d in self.degrees:
            if d is None:
                signed_measures.append(signed_measures_euler)
            else:
                signed_measures.append(signed_measures_pers[count])
                count += 1
        signed_measures += signed_measures_rank
        if not self._möbius_inversion and self.flatten:
            signed_measures = np.asarray(signed_measures).flatten()
        return signed_measures
    def transform(self,X):
        assert self.filtration_grid is not None or self.individual_grid, "Fit first"
        prefer = "loky" if self._is_input_delayed else "threading"
        out = Parallel(n_jobs=self.n_jobs, backend=prefer)(
            delayed(self.transform1)(to_st) for to_st in tqdm(X, disable = not self.progress, desc=f"Computing signed measure decompositions")
        )
        return out
        # return [self.transform1(x) for x in tqdm(X, disable = not self.progress, desc="Computing Hilbert function")]





class SimplexTrees2SignedMeasures(SimplexTree2SignedMeasure):
	"""
	Input
	-----
	
	(data) x (axis, e.g. different bandwidths for simplextrees) x (simplextree)
	
	Output
	------ 
	(data) x (axis) x (degree) x (signed measure)
	"""
	def __init__(self,**kwargs):
		super().__init__(**kwargs)
		self._num_st_per_data=None
		# self._super_model=SimplexTree2SignedMeasure(**kwargs)
		self._filtration_grids = None
		return
	def fit(self, X, y=None):
		if len(X) == 0: return self
		try:
			self._num_st_per_data = len(X[0])
		except:
			raise Exception("Shape has to be (num_data, num_axis), dtype=SimplexTreeMulti")
		self._filtration_grids=[]
		for axis in range(self._num_st_per_data):
			self._filtration_grids.append(super().fit([x[axis] for x in X]).filtration_grid)
			# self._super_fits.append(truc)
		# self._super_fits_params = [super().fit([x[axis] for x in X]).get_params() for axis in range(self._num_st_per_data)]
		return self
	def transform(self, X):
		if self.normalize_filtrations:
			_reconversion_grids = [[np.linspace(0,1, num=len(f), dtype=float) for f in F] for F in self._filtration_grids]
		else:
			_reconversion_grids = self._filtration_grids
		def todo(x):
			# return [SimplexTree2SignedMeasure().set_params(**transformer_params).transform1(x[axis]) for axis,transformer_params in enumerate(self._super_fits_params)]
			out = [
				self.transform1(
					x[axis],filtration_grid=filtration_grid, 
					_reconversion_grid=_reconversion_grid) 
				for axis, filtration_grid, _reconversion_grid in 
				zip(range(self._num_st_per_data), self._filtration_grids, _reconversion_grids)
			]
			return out

		return Parallel(n_jobs=self.n_jobs, backend="threading")(delayed(todo)(x) 
			for x in tqdm(X, disable=not self.progress,
				desc="Computing Signed Measures from simplextrees."))


def rescale_sparse_signed_measure(signed_measure, filtration_weights, normalize_scales=None):
	from copy import deepcopy
	out = deepcopy(signed_measure)
	if normalize_scales is None:
		for degree in range(len(out)): # degree
			for parameter in range(len(filtration_weights)):
				out[degree][0][:,parameter] *= filtration_weights[parameter] 
				## TODO Broadcast w.r.t. the parameter
	else:
		for degree in range(len(out)):
			for parameter in range(len(filtration_weights)):
				out[degree][0][:,parameter] *= filtration_weights[parameter] / normalize_scales[degree][parameter]
	return out



class SignedMeasureFormatter(BaseEstimator,TransformerMixin):
	"""
	Input
	-----
	
	(data) x (degree) x (signed measure) or (data) x (axis) x (degree) x (signed measure)
	
	Iterable[list[signed_measure_matrix of degree]] or Iterable[previous].
	
	The second is meant to use multiple choices for signed measure input. An example of usage : they come from a Rips + Density with different bandwidth. 
	It is controlled by the axis parameter.

	Output
	------
	
	Iterable[list[(reweighted)_sparse_signed_measure of degree]]
	"""
	def __init__(self, 
			filtrations_weights:Iterable[float]=None,
			normalize=False,
			num_parameters:int|None=None,
			plot:bool=False,
			unsparse:bool=False,
			axis:int=None,
			resolution:int|Iterable[int]=50,
			flatten:bool=False,
			deep_format:bool=False,
			unrag:bool=True,
			n_jobs:int=1,
			verbose:bool=False,
			integrate:bool=False,
			grid_strategy='regular',
		):
		super().__init__()
		self.filtrations_weights = filtrations_weights
		self.num_parameters = num_parameters
		self.plot=plot
		self.unsparse = unsparse
		self.n_jobs=n_jobs
		self.axis=axis
		self._num_axis=None
		self._is_input_sparse=None
		self.resolution=resolution
		self._filtrations_bounds=None
		self.flatten=flatten
		self.normalize=normalize
		self._normalization_factors=None
		self.deep_format=deep_format
		self.unrag = unrag
		assert not self.deep_format or not self.unsparse
		assert not normalize or (not unsparse and not deep_format and not integrate)
		self.verbose=verbose
		self._num_degrees=None
		self.integrate=integrate
		self.grid_strategy=grid_strategy
		self._infered_grids=None
		return
	
	def _get_filtration_bounds(self, X, axis):
		num_degrees = len(X[0][axis])
		stuff = [
			np.concatenate([sm[axis][degree][0] for sm in X], axis=0) 
				for degree in range(num_degrees)
		]
		sizes_ = np.array([len(x) == 0 for x in stuff])
		assert np.all(1- sizes_), f"Degree axis {np.where(sizes_)} is/are trivial !"
		filtrations_bounds = np.asarray([[f.min(axis=0), f.max(axis=0)] for f in stuff])
		normalization_factors = filtrations_bounds[:,1] - filtrations_bounds[:,0] if self.normalize else None
		# print("Normalization factors : ",self._normalization_factors)
		if np.any(normalization_factors == 0 ):
			indices = np.where(normalization_factors == 0)
			# warn(f"Constant filtration encountered, at degree, parameter {indices} and axis {self.axis}.")
			normalization_factors[indices] = 1
		return filtrations_bounds,normalization_factors
	
	def _plot_signed_measures(self, sms:Iterable[np.ndarray], size=4):
		from multipers.plots import plot_signed_measure
		num_degrees = len(sms[0])
		num_imgs = len(sms)
		fig, axes = plt.subplots(ncols=num_degrees, nrows=num_imgs, figsize=(size*num_degrees,size*num_imgs))
		axes = np.asarray(axes).reshape(num_imgs,num_degrees)
		# assert axes.ndim==2, "Internal error"
		for i, sm in enumerate(sms):
			for j, sm_of_degree in enumerate(sm):
				plot_signed_measure(sm_of_degree, ax=axes[i,j])

	def fit(self, X, y=None):
		assert not self.normalize or (not self.unsparse and not self.deep_format and not self.integrate)
		## Gets a grid. This will be the max in each coord+1
		if len(X) == 0 or len(X[0]) == 0 or (self.axis is not None and len(X[0][0][0]) == 0):	return self
		
		self._is_input_sparse = (isinstance(X[0][0], tuple) and self.axis is None) or (isinstance(X[0][0][0], tuple) and self.axis is not None)
		# print("Sparse input : ", self._is_input_sparse)
		if self.axis is None:
			# try:
			## DATA,NOAXIS,DEGREE,(sm,weights)
			self.num_parameters = X[0][0][0].shape[1] if self._is_input_sparse else X[0][0].ndim
			# except:
			#		print(X)
			#		raise Exception("")
			self._num_degrees = len(X[0])
		else:
			#  (data) x (axis) x (degree) x (signed measure)
			self.num_parameters = X[0][0][0][0].shape[1] if self._is_input_sparse else X[0][0][0].ndim
			self._num_degrees = len(X[0][0])
		# Sets weights to 1 if None
		if self.filtrations_weights is None:
			self.filtrations_weights = np.array([1]*self.num_parameters)
		## Checks compatibilities
		assert self._is_input_sparse or (not self.deep_format)

		# resolution is iterable over the parameters
		try:
			float(self.resolution)
			self.resolution = [self.resolution]*self.num_parameters
		except:
			None
		assert len(self.filtrations_weights) == self.num_parameters == len(self.resolution), f"Number of parameter is not consistent. Inferred : {self.num_parameters}, Filtration weigths : {len(self.filtrations_weights)}, Resolutions : {len(self.resolution)}."
		# if not sparse : not recommended. 
		assert np.all(1 == np.asarray(self.filtrations_weights)) or self._is_input_sparse, f"Use sparse signed measure to rescale. Recieved weights {self.filtrations_weights}"
		self._num_axis = None if self.axis is None else len(X[0]) 
		
		## Computes normalization factors
		if self._is_input_sparse and self.normalize:
			axis = slice(None) if self.axis is None else self.axis
			if axis == -1:
				self._filtrations_bounds = []
				self._normalization_factors = []
				for ax in range(self._num_axis):
					filtration_bounds, normalization_factors = self._get_filtration_bounds(X, axis=ax)
					self._filtrations_bounds.append(filtration_bounds)
					self._normalization_factors.append(normalization_factors)
			else:
				self._filtrations_bounds, self._normalization_factors = self._get_filtration_bounds(X, axis=axis)
		elif self._is_input_sparse and (self.integrate or self.unsparse):
			axis = [slice(None)] if self.axis is None else range(self._num_axis) if self.axis == -1 else [self.axis]
			filtration_values = [np.concatenate([x[ax][degree][0] for x in X for degree in range(self._num_degrees)]) for ax in axis]
			## axis, filtration_values
			filtration_values = [reduce_grid(f_ax.T, resolutions=self.resolution, strategy=self.grid_strategy) for f_ax in filtration_values]
			self._infered_grids = filtration_values

		if self.verbose:
			print("------------SignedMeasureFormatter------------")
			print("---- Parameters")
			print(f"Sparse input : {self._is_input_sparse}")
			print(f"Number of axis : {self._num_axis}")
			print(f"Number of degrees : {self._num_degrees}")
			print(f"Filtration bounds : \n{self._filtrations_bounds}")
			print(f"Normalization factor : \n{self._normalization_factors}")
			if self._infered_grids is not None:
				print(f"Filtration grid shape : {tuple(tuple(len(f) for f in F) for F in self._infered_grids)}")
			print("---- SM stats")
			print(f"In axis : {1 if self.axis is None else len(X[0])}")
			if self.axis == -1:
				axis = range(len(X[0]))
			else:
				axis = [slice(None)] if self.axis is None else [self.axis]
			sizes = [[[len(xd[1]) for xd in x[ax]] for x in X] for ax in axis]
			print(f"Size means (axis) x (degree): {np.mean(sizes, axis=(1))}")
			print(f"Size std : {np.std(sizes, axis=(1))}")
			print("----------------------------------------------")
		return self

	def unsparse_signed_measure(self, sparse_signed_measure):
		filtrations = self._infered_grids # ax, filtration
		out = []
		axis = range(self._num_axis) if self.axis == -1 else [slice(None)]
		for filtrations_of_ax, ax in zip(filtrations, axis):
			sparse_signed_measure_of_ax = sparse_signed_measure[ax]
			measure_of_ax = []
			for pts, weights in sparse_signed_measure_of_ax: # over degree
				signed_measure,_ = np.histogramdd(
					pts,bins=filtrations_of_ax, 
					weights=weights
					)
				if self.flatten:	signed_measure = signed_measure.flatten()
				measure_of_ax.append(signed_measure)
			out.append(np.asarray(measure_of_ax))
			
		if self.flatten:	out = np.concatenate(out).flatten()
		if self.axis == -1:
			return np.asarray(out)
		else:
			return np.asarray(out)[0]
	
	@staticmethod
	def deep_format_measure(signed_measure):
		from numpy import empty, float32
		dirac_positions,dirac_signs = signed_measure
		new_shape = list(dirac_positions.shape)
		new_shape[1]+=1
		c=empty(new_shape, dtype=float32)
		c[:,:-1] =dirac_positions
		c[:,-1] = dirac_signs
		return c
	
	@staticmethod
	def _integrate_measure(sm, filtrations):
		from multipers.point_measure_integration import integrate_measure
		return integrate_measure(sm[0], sm[1],filtrations)
	


	def transform(self,X):
		def rescale_from_not_sparse(signed_measure:Iterable[np.ndarray]):
			if not self.flatten:
				return signed_measure
			return np.asarray([sm.flatten() for sm in signed_measure]).flatten()

		def rescale_from_sparse(sparse_signed_measure):
			if self.axis == -1:
				return [
					rescale_sparse_signed_measure(sparse_signed_measure[ax], filtration_weights=self.filtrations_weights, normalize_scales = n) 
					for ax,n in enumerate(self._normalization_factors)
				]
			return rescale_sparse_signed_measure(sparse_signed_measure, filtration_weights=self.filtrations_weights, normalize_scales = self._normalization_factors)
			
			
		if self._is_input_sparse:
			todo_rescale = rescale_from_sparse
		else:
			todo_rescale = rescale_from_not_sparse
		
		if self.axis is None or self.axis == -1:
			out = X
		else:
			out = tuple(x[self.axis] for x in X)
		if self._normalization_factors is not None:
			if self.n_jobs >1:
				out = Parallel(n_jobs=self.n_jobs, backend="threading")(delayed(todo_rescale)(x) for x in out)
			else:
				out = tuple(todo_rescale(x) for x in out)
		
		if self._is_input_sparse:
			if self.plot:
				# assert ax != -1, "Not implemented"
				self._plot_signed_measures(out)
			if self.integrate:
				filtrations = self._infered_grids
				# if self.axis != -1:
				ax = 0 #if self.axis is None else self.axis # TODO deal with axis -1
				
				assert ax != -1, "Not implemented"
				# try:
				out = np.asarray([[self._integrate_measure(x[degree],filtrations=filtrations[ax]) for degree in range(self._num_degrees)] for x in out])
				# except:
				#		print(self.axis, ax, filtrations)
				if self.flatten:
					out = out.reshape((len(X), -1))
				# else:
				#		out = [[[self._integrate_measure(x[axis][degree],filtrations=filtrations[degree].T) for degree in range(self._num_degrees)] for axis in range(self._num_axis)] for x in out]
			elif self.unsparse:
				out = [self.unsparse_signed_measure(x) for x in out]
			elif self.deep_format:
				if self.axis is None:
					num_degrees = len(out[0])
					axes = [slice(None)]
				else:
					num_degrees = len(out[0][0])
					axes = range(self._num_axis) if self.axis == -1 else (self.axis,)
				out = [[self.deep_format_measure(sm[axis][degree]) for sm in out] for degree in range(num_degrees) for axis in axes]
				if self.unrag:
					max_num_pts = np.max([sm.shape[0] for sm_of_axis in out for sm in sm_of_axis])
					num_axis = len(out)
					num_data = len(out[0])
					num_parameters = out[0][0].shape[1]
					unragged_tensor = np.zeros(shape=(num_axis,num_data, max_num_pts,num_parameters),dtype=np.float32)
					for ax in range(num_axis):
						for data in range(num_data):
							sm = out[ax][data]
							a,b = sm.shape
							unragged_tensor[ax,data,:a,:b] = sm
					out = unragged_tensor
		return out
	






class SignedMeasure2Convolution(BaseEstimator,TransformerMixin):
	"""
	Discrete convolution of a signed measure

	Input
	-----
	
	(data) x (degree) x (signed measure)

	Parameters
	----------
	 - filtration_grid : Iterable[array] For each filtration, the filtration values on which to evaluate the grid
	 - resolution : int or (num_parameter) : If filtration grid is not given, will infer a grid, with this resolution
	 - grid_strategy : the strategy to generate the grid. Available ones are regular, quantile, exact
	 - flatten : if true, the output will be flattened
	 - kernel : kernel to used to convolve the images.
	 - flatten : flatten the images if True
	 - progress : progress bar if True
	 - backend : sklearn, pykeops or numba.
	 - plot : Creates a plot Figure.

	Output
	------
	
	(data) x (concatenation of imgs of degree)
	"""
	def __init__(self,
            filtration_grid:Iterable[np.ndarray]=None, 
            kernel="gaussian", 
            bandwidth:float|Iterable[float]=1., 
            flatten:bool=False, n_jobs:int=1,
            resolution:int|None=None, 
            grid_strategy:str="regular",
            progress:bool=False, 
            backend:str='pykeops',
            plot:bool=False,
        #   **kwargs ## DANGEROUS
        ):
		super().__init__()
		self.kernel=kernel
		self.bandwidth=bandwidth
		# self.more_kde_kwargs=kwargs
		self.filtration_grid=filtration_grid
		self.flatten=flatten
		self.progress=progress
		self.n_jobs = n_jobs
		self.resolution = resolution
		self.grid_strategy = grid_strategy
		self._is_input_sparse = None
		self._refit = filtration_grid is None
		self._input_resolution=None
		self._bandwidths=None
		self.diameter=None
		self.backend=backend
		self.plot=plot
		return
	def fit(self, X, y=None):
		## Infers if the input is sparse given X 
		if len(X) == 0: return self
		if isinstance(X[0][0], tuple):	self._is_input_sparse = True 
		else: self._is_input_sparse = False
		# print(f"IMG output is set to {'sparse' if self.sparse else 'matrix'}")
		if not self._is_input_sparse:
			self._input_resolution = X[0][0].shape
			try:
				float(self.bandwidth)
				b = float(self.bandwidth)
				self._bandwidths = [b if b > 0 else -b * s for s in self._input_resolution]
			except:
				self._bandwidths = [b if b > 0 else -b * s for s,b in zip(self._input_resolution, self.bandwidth)]
			return self # in that case, singed measures are matrices, and the grid is already given
		
		if self.filtration_grid is None and self.resolution is None:
			raise Exception("Cannot infer filtration grid. Provide either a filtration grid or a resolution.")
		## If not sparse : a grid has to be defined
		if self._refit:
			# print("Fitting a grid...", end="")
			pts = np.concatenate([
				sm[0] for signed_measures in X for sm in signed_measures
			]).T
			self.filtration_grid = reduce_grid(filtrations_values=pts, strategy=self.grid_strategy, resolutions=self.resolution)
			# print('Done.')
		if self.filtration_grid is not None: 
			self.diameter=np.linalg.norm([f.max() - f.min() for f in self.filtration_grid])
			if self.progress: print(f"Computed a diameter of {self.diameter}")
		return self
	
	def _sparsify(self,sm):
		return tensor_möbius_inversion(input=sm,grid_conversion=self.filtration_grid)

	def _sm2smi(self, signed_measures:Iterable[np.ndarray]):
		# print(self._input_resolution, self.bandwidths, _bandwidths)
		from scipy.ndimage import gaussian_filter
		return np.concatenate([
				gaussian_filter(input=signed_measure, sigma=self._bandwidths,mode="constant", cval=0)
			for signed_measure in signed_measures], axis=0)
	# def _sm2smi_sparse(self, signed_measures:Iterable[tuple[np.ndarray]]):
	#		return np.concatenate([
	#				_pts_convolution_sparse(
	#					pts = signed_measure_pts, pts_weights = signed_measure_weights,
	#					filtration_grid = self.filtration_grid, 
	#					kernel=self.kernel,
	#					bandwidth=self.bandwidths,
	#					**self.more_kde_kwargs
	#				)
	#			for signed_measure_pts, signed_measure_weights  in signed_measures], axis=0)
	def _transform_from_sparse(self,X):
		bandwidth = self.bandwidth if self.bandwidth > 0 else -self.bandwidth * self.diameter
## COMPILE KEOPS FIRST
		dummyx = [X[0]]
		dummyf = [f[:2] for f in self.filtration_grid]
		convolution_signed_measures(dummyx, filtrations=dummyf, bandwidth=bandwidth, flatten=self.flatten, n_jobs=1, kernel=self.kernel, backend=self.backend)

		return convolution_signed_measures(X, filtrations=self.filtration_grid, bandwidth=bandwidth, flatten=self.flatten, n_jobs=self.n_jobs, kernel=self.kernel, backend=self.backend)

	def _plot_imgs(self, imgs:Iterable[np.ndarray], size=4):
		from multipers.plots import plot_surface
		num_degrees = imgs[0].shape[0]
		num_imgs = len(imgs)
		fig, axes = plt.subplots(ncols=num_degrees, nrows=num_imgs, figsize=(size*num_degrees,size*num_imgs))
		axes = np.asarray(axes).reshape(num_imgs,num_degrees)
		# assert axes.ndim==2, "Internal error"
		for i, img in enumerate(imgs):
			for j, img_of_degree in enumerate(img):
				plot_surface(self.filtration_grid, img_of_degree, ax=axes[i,j], cmap='Spectral')
	def transform(self,X):
		if self._is_input_sparse is None:	raise Exception("Fit first")
		if self._is_input_sparse:
				out = self._transform_from_sparse(X)
		else:
				todo = SignedMeasure2Convolution._sm2smi
				out =  Parallel(n_jobs=self.n_jobs, backend="threading")(delayed(todo)(self, signed_measures) for signed_measures in tqdm(X, desc="Computing images", disable = not self.progress))
		if self.plot and not self.flatten:
				if self.progress:	print("Plotting convolutions...", end="")
				self._plot_imgs(out)
				if self.progress:	print("Done !")
		if self.flatten and not self._is_input_sparse:	out = [x.flatten() for x in out]
		return np.asarray(out)



class SignedMeasure2SlicedWassersteinDistance(BaseEstimator,TransformerMixin):
	"""
	Transformer from signed measure to distance matrix.
	
	Input
	-----
	
	(data) x (degree) x (signed measure)

	Format
	------
	- a signed measure : tuple of array. (point position) : npts x (num_paramters) and weigths : npts
	- each data is a list of signed measure (for e.g. multiple degrees)

	Output
	------
	- (degree) x (distance matrix)
	"""
	def __init__(self, n_jobs=None, num_directions:int=10, _sliced:bool=True, epsilon=-1, ground_norm=1, progress = False, grid_reconversion=None, scales=None):
		super().__init__()
		self.n_jobs=n_jobs
		self._SWD_list = None
		self._sliced=_sliced
		self.epsilon = epsilon
		self.ground_norm = ground_norm
		self.num_directions = num_directions
		self.progress = progress
		self.grid_reconversion=grid_reconversion
		self.scales=scales
		return
		
	def fit(self, X, y=None):
		from multipers.ml.sliced_wasserstein import (SlicedWassersteinDistance,
																								 WassersteinDistance)

		# _DISTANCE = lambda : SlicedWassersteinDistance(num_directions=self.num_directions) if self._sliced else WassersteinDistance(epsilon=self.epsilon, ground_norm=self.ground_norm) # WARNING if _sliced is false, this distance is not CNSD
		if len(X) == 0:	return self
		num_degrees = len(X[0])
		self._SWD_list = [
			SlicedWassersteinDistance(num_directions=self.num_directions, n_jobs=self.n_jobs, scales=self.scales) 
			if self._sliced else 
			WassersteinDistance(epsilon=self.epsilon, ground_norm=self.ground_norm, n_jobs=self.n_jobs) 
			for _ in range(num_degrees)
		]
		for degree, swd in enumerate(self._SWD_list):
			signed_measures_of_degree = [x[degree] for x in X]
			swd.fit(signed_measures_of_degree)
		return self
	def transform(self,X):
		assert self._SWD_list is not None, "Fit first"
		# out = []
		# for degree, swd in tqdm(enumerate(self._SWD_list), desc="Computing distance matrices", total=len(self._SWD_list), disable= not self.progress):
		with tqdm(enumerate(self._SWD_list), desc="Computing distance matrices", total=len(self._SWD_list), disable= not self.progress) as SWD_it:
			# signed_measures_of_degree = [x[degree] for x in X]
			# out.append(swd.transform(signed_measures_of_degree))
			todo = lambda swd, X_of_degree : swd.transform(X_of_degree)
			out = Parallel(n_jobs=self.n_jobs, prefer="threads")(delayed(todo)(swd,[x[degree] for x in X]) for degree, swd in SWD_it)
			return np.asarray(out)
	def predict(self, X): 
		return self.transform(X)


class SignedMeasures2SlicedWassersteinDistances(BaseEstimator,TransformerMixin):
	"""
	Transformer from signed measure to distance matrix.
	Input
	-----
	(data) x opt (axis) x (degree) x (signed measure)
	
	Format
	------
	- a signed measure : tuple of array. (point position) : npts x (num_paramters) and weigths : npts
	- each data is a list of signed measure (for e.g. multiple degrees)

	Output
	------
	- (axis) x (degree) x (distance matrix)
	"""
	def __init__(self, progress=False, n_jobs:int=1, scales:Iterable[Iterable[float]]|None = None, **kwargs): # same init
		self._init_child = SignedMeasure2SlicedWassersteinDistance(progress=False, scales=None,n_jobs=-1, **kwargs)
		self._axe_iterator=None
		self._childs_to_fit=None
		self.scales = scales
		self.progress = progress
		self.n_jobs=n_jobs
		return
		
	def fit(self, X, y=None):
		from sklearn.base import clone
		if len(X) == 0:	 return self
		if isinstance(X[0][0],tuple): # Meaning that there are no axes
			self._axe_iterator = [slice(None)]
		else:
			self._axe_iterator = range(len(X[0]))
		if self.scales is None: 
			self.scales = [None]
		else:
			self.scales = np.asarray(self.scales)
			if self.scales.ndim == 1:	
				self.scales = np.asarray([self.scales])
		assert self.scales[0] is None or self.scales.ndim==2, "Scales have to be either None or a list of scales !"
		self._childs_to_fit = [
			clone(self._init_child).set_params(scales=scales).fit(
				[x[axis] for x in X]) 
				for axis, scales in product(self._axe_iterator, self.scales)
			]
		print("New axes : ", list(product(self._axe_iterator, self.scales)))
		return self
	def transform(self,X):
		return Parallel(n_jobs=self.n_jobs, prefer="processes")(
			delayed(self._childs_to_fit[child_id].transform)([x[axis] for x in X])
				for child_id, (axis, _) in tqdm(enumerate(product(self._axe_iterator, self.scales)), 
					desc=f"Computing distances matrices of axis, and scales", disable=not self.progress, total=len(self._childs_to_fit)
				) 
		)
		# [
		#			child.transform([x[axis // len(self.scales)] for x in X]) 
		#			for axis, child in tqdm(enumerate(self._childs_to_fit), 
		#				desc=f"Computing distances of axis", disable=not self.progress, total=len(self._childs_to_fit)
		#			)
		#		]


class SimplexTree2RectangleDecomposition(BaseEstimator,TransformerMixin):
	"""
	Transformer. 2 parameter SimplexTrees to their respective rectangle decomposition. 
	"""
	def __init__(self, filtration_grid:np.ndarray, degrees:Iterable[int], plot=False, reconvert_grid=True, num_collapses:int=0):
		super().__init__()
		self.filtration_grid = filtration_grid
		self.degrees = degrees
		self.plot=plot
		self.reconvert_grid = reconvert_grid
		self.num_collapses=num_collapses
		return
	def fit(self, X, y=None):
		"""
		TODO : infer grid from multiple simplextrees
		"""
		return self
	def transform(self,X:Iterable[mp.SimplexTreeMulti]):
		rectangle_decompositions = [
			[_st2ranktensor(
				simplextree, filtration_grid=self.filtration_grid,
				degree=degree,
				plot=self.plot,
				reconvert_grid = self.reconvert_grid,
				num_collapse=self.num_collapses
			) for degree in self.degrees]
			for simplextree in X
		]
		## TODO : return iterator ?
		return rectangle_decompositions



def _st2ranktensor(st:mp.SimplexTreeMulti, filtration_grid:np.ndarray, degree:int, plot:bool, reconvert_grid:bool, num_collapse:int|str=0):
	"""
	TODO
	"""
	## Copy (the squeeze change the filtration values)
	stcpy = mp.SimplexTreeMulti(st)
	# turns the simplextree into a coordinate simplex tree
	stcpy.grid_squeeze(
		filtration_grid = filtration_grid, 
		coordinate_values = True)
	# stcpy.collapse_edges(num=100, strong = True, ignore_warning=True)
	if num_collapse == "full":
		stcpy.collapse_edges(full=True, ignore_warning=True, max_dimension=degree+1)
	elif isinstance(num_collapse, int):
		stcpy.collapse_edges(num=num_collapse,ignore_warning=True, max_dimension=degree+1)
	else:
		raise TypeError(f"Invalid num_collapse={num_collapse} type. Either full, or an integer.")
	# computes the rank invariant tensor
	rank_tensor = mp.rank_invariant2d(stcpy, degree=degree, grid_shape=[len(f) for f in filtration_grid])
	# refactor this tensor into the rectangle decomposition of the signed betti
	grid_conversion = filtration_grid if reconvert_grid else None 
	rank_decomposition = rank_decomposition_by_rectangles(
		rank_tensor, threshold=True,
		)
	rectangle_decomposition = tensor_möbius_inversion(tensor = rank_decomposition, grid_conversion = grid_conversion, plot=plot, num_parameters=st.num_parameters)
	return rectangle_decomposition

class DegreeRips2SignedMeasure(BaseEstimator, TransformerMixin):
	def __init__(self, degrees:Iterable[int], min_rips_value:float, 
				max_rips_value,max_normalized_degree:float, min_normalized_degree:float, 
			grid_granularity:int, progress:bool=False, n_jobs=1, sparse:bool=False, 
			_möbius_inversion=True,
			fit_fraction=1,
			) -> None:
		super().__init__()
		self.min_rips_value = min_rips_value
		self.max_rips_value = max_rips_value
		self.min_normalized_degree = min_normalized_degree
		self.max_normalized_degree = max_normalized_degree
		self._max_rips_value = None
		self.grid_granularity = grid_granularity
		self.progress=progress
		self.n_jobs = n_jobs
		self.degrees = degrees
		self.sparse=sparse
		self._möbius_inversion = _möbius_inversion
		self.fit_fraction=fit_fraction
		return
	def fit(self, X:np.ndarray|list, y=None):
		if self.max_rips_value < 0:
			print("Estimating scale...", flush=True, end="")
			indices = np.random.choice(len(X),min(len(X), int(self.fit_fraction*len(X))+1) ,replace=False)
			diameters =np.max([distance_matrix(x,x).max() for x in (X[i] for i in indices)])
			print(f"Done. {diameters}", flush=True)
		self._max_rips_value = - self.max_rips_value * diameters if self.max_rips_value < 0 else self.max_rips_value
		return self
	
	def _transform1(self, data:np.ndarray):
		_distance_matrix = distance_matrix(data, data)
		signed_measures = []
		rips_values, normalized_degree_values, hilbert_functions, minimal_presentations = hf_degree_rips(
			_distance_matrix,
			min_rips_value = self.min_rips_value,
			max_rips_value = self._max_rips_value,
			min_normalized_degree = self.min_normalized_degree,
			max_normalized_degree = self.max_normalized_degree,
			grid_granularity = self.grid_granularity,
			max_homological_dimension = np.max(self.degrees),
		)
		for degree in self.degrees:
			hilbert_function = hilbert_functions[degree]
			signed_measure = signed_betti(hilbert_function, threshold=True) if self._möbius_inversion else hilbert_function
			if self.sparse:
				signed_measure = tensor_möbius_inversion(
					tensor=signed_measure,num_parameters=2,
					grid_conversion=[rips_values, normalized_degree_values]
				)
			if not self._möbius_inversion: signed_measure = signed_measure.flatten()
			signed_measures.append(signed_measure)
		return signed_measures
	def transform(self,X):
		return Parallel(n_jobs=self.n_jobs)(delayed(self._transform1)(data) 
		for data in tqdm(X, desc=f"Computing DegreeRips, of degrees {self.degrees}"))




def tensor_möbius_inversion(tensor, grid_conversion:Iterable[np.ndarray]|None = None, plot:bool=False, raw:bool=False, num_parameters:int|None=None):
	from torch import Tensor
	betti_sparse = Tensor(tensor.copy()).to_sparse() # Copy necessary in some cases :(
	num_indices, num_pts = betti_sparse.indices().shape
	num_parameters = num_indices if num_parameters is None else num_parameters
	if num_indices == num_parameters: # either hilbert or rank invariant
		rank_invariant = False
	elif 2*num_parameters == num_indices:
		rank_invariant = True
	else:
		raise TypeError(f"Unsupported betti shape. {num_indices} has to be either {num_parameters} or {2*num_parameters}.")
	points_filtration = np.asarray(betti_sparse.indices().T, dtype=int)
	weights = np.asarray(betti_sparse.values(), dtype=int)

	if grid_conversion is not None:
		coords = np.empty(shape=(num_pts,num_indices), dtype=float)
		for i in range(num_indices):
			coords[:,i] = grid_conversion[i%num_parameters][points_filtration[:,i]]
	else:
		coords = points_filtration
	if (not rank_invariant) and plot:
		plt.figure()
		color_weights = np.empty(weights.shape)
		color_weights[weights>0] = np.log10(weights[weights>0])+2
		color_weights[weights<0] = -np.log10(-weights[weights<0])-2
		plt.scatter(points_filtration[:,0],points_filtration[:,1], c=color_weights, cmap="coolwarm")
	if (not rank_invariant) or raw: return coords, weights
	def _is_trivial(rectangle:np.ndarray):
		birth=rectangle[:num_parameters]
		death=rectangle[num_parameters:]
		return np.all(birth<=death) # and not np.array_equal(birth,death)
	correct_indices = np.array([_is_trivial(rectangle) for rectangle in coords])
	if len(correct_indices) == 0:	return np.empty((0, num_indices)), np.empty((0))
	signed_measure = np.asarray(coords[correct_indices])
	weights = weights[correct_indices]
	if plot:
		assert signed_measure.shape[1] == 4 # plot only the rank decompo for the moment
		def _plot_rectangle(rectangle:np.ndarray, weight:float):
			x_axis=rectangle[[0,2]]
			y_axis=rectangle[[1,3]]
			color = "blue" if weight > 0 else "red"
			plt.plot(x_axis, y_axis, c=color)
		for rectangle, weight in zip(signed_measure, weights):
			_plot_rectangle(rectangle=rectangle, weight=weight)
	return signed_measure, weights

