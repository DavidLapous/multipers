from sklearn.base import BaseEstimator, TransformerMixin
import gudhi as gd
from os.path import exists
import networkx as nx
from joblib import Parallel, delayed
import numpy as np
from tqdm import tqdm
from warnings import warn
from sklearn.neighbors import KernelDensity
from typing import Iterable
from gudhi.representations import Landscape
from gudhi.representations.vector_methods import PersistenceImage
from gudhi.representations.kernel_methods import SlicedWassersteinDistance


from types import FunctionType
def get_simplextree(x)->gd.SimplexTree:
	if isinstance(x, gd.SimplexTree):
		return x
	if isinstance(x, FunctionType):
		return x()
	if len(x) == 3 and isinstance(x[0],FunctionType):
		f,args, kwargs = x
		return f(*args,**kwargs)
	raise TypeError("Not a valid SimplexTree")
def get_simplextrees(X)->Iterable[gd.SimplexTree]:
	if len(X) == 2 and isinstance(X[0], FunctionType):
		f,data = X
		return (f(x) for x in data)
	if len(X) == 0: return []
	if not isinstance(X[0], gd.SimplexTree):
		raise TypeError
	return X
	



############## INTERVALS (for sliced wasserstein)
class Graph2SimplexTree(BaseEstimator,TransformerMixin):
	def __init__(self, f:str="ricciCurvature",dtype=gd.SimplexTree, reverse_filtration:bool=False):
		super().__init__()
		self.f=f # filtration to search in graph
		self.dtype = dtype # If None, will delay the computation in the pipe (for parallelism)
		self.reverse_filtration = reverse_filtration # reverses the filtration #TODO
	def fit(self, X, y=None):
		return self
	def transform(self,X:list[nx.Graph]):
		def todo(graph, f=self.f) -> gd.SimplexTree: # TODO : use batch insert
			st = gd.SimplexTree()
			for i in graph.nodes:	st.insert([i], graph.nodes[i][f])
			for u,v in graph.edges:	st.insert([u,v], graph[u][v][f])
			return st
		return [todo, X] if self.dtype is None else Parallel(n_jobs=-1, prefer="threads")(delayed(todo)(graph) for graph in X)


class PointCloud2SimplexTree(BaseEstimator,TransformerMixin):
	def __init__(self, delayed:bool = False, threshold = np.inf):
		super().__init__()
		self.delayed = delayed
		self.threshold=threshold
	@staticmethod
	def _get_point_cloud_diameter(x):
		from scipy.spatial import distance_matrix
		return np.max(distance_matrix(x,x))
	def fit(self, X, y=None):
		if self.threshold < 0:
			self.threshold = max(self._get_point_cloud_diameter(x) for x in X)
		return self
	def transform(self,X:list[nx.Graph]):
		def todo(point_cloud) -> gd.SimplexTree: # TODO : use batch insert
			st = gd.AlphaComplex(points=point_cloud).create_simplex_tree(max_alpha_square = self.threshold**2)
			return st
		return [todo, X] if self.delayed is None else Parallel(n_jobs=-1, prefer="threads")(delayed(todo)(point_cloud) for point_cloud in X)



#################### FILVEC
def get_filtration_values(g:nx.Graph, f:str)->np.ndarray:
	filtrations_values = [
		g.nodes[node][f] for node in g.nodes
	]+[
		g[u][v][f] for u,v in g.edges
	]
	return np.array(filtrations_values)
def graph2filvec(g:nx.Graph, f:str, range:tuple, bins:int)->np.ndarray:
    fs = get_filtration_values(g, f)
    return np.histogram(fs, bins=bins,range=range)[0]
class FilvecGetter(BaseEstimator, TransformerMixin):
	def __init__(self, f:str="ricciCurvature",quantile:float=0., bins:int=100, n_jobs:int=1):
		super().__init__()
		self.f=f
		self.quantile=quantile
		self.bins=bins
		self.range:tuple[float]|None=None
		self.n_jobs=n_jobs
	def fit(self, X, y=None):
		filtration_values = np.concatenate(Parallel(n_jobs=self.n_jobs)(delayed(get_filtration_values)(g,f=self.f) for g in X))
		self.range= tuple(np.quantile(filtration_values, [self.quantile, 1-self.quantile]))
		return self
	def transform(self,X):
		if self.range == None:
			print("Fit first")
			return
		return Parallel(n_jobs=self.n_jobs)(delayed(graph2filvec)(g,f=self.f, range=self.range, bins=self.bins) for g in X)




############# Filvec from SimplexTree
# Input list of [list of diagrams], outputs histogram of persitence values (x and y coord mixed) 
def simplextree2hist(simplextree, range:tuple[float, float], bins:int, density:bool)->np.ndarray: #TODO : Anything to histogram
	filtration_values = np.array([f for s,f in simplextree.get_simplices()])
	return np.histogram(filtration_values, bins=bins,range=range, density=density)[0]
class SimplexTree2Histogram(BaseEstimator, TransformerMixin):
	def __init__(self, quantile:float=0., bins:int=100, n_jobs:int=1, progress:bool=False, density:bool=True):
		super().__init__()
		self.range:np.ndarray | None=None
		self.quantile:float=quantile
		self.bins:int=bins
		self.n_jobs=n_jobs
		self.density=density
		self.progress = progress
		# self.max_dimension=None # TODO: maybe use it
	def fit(self, X, y=None): # X:list[diagrams]
		if len(X) == 0:	return self
		if type(X[0]) is gd.SimplexTree: # If X contains simplextree : nothing to do
			data = X
			to_st = lambda x : x
		else: # otherwise we assume that we retrieve simplextrees using f,data = X; simplextrees = (f(x) for x in data)
			# assert len(X) == 2
			to_st, data = X
		persistence_values = np.array([f for st in data for s,f in to_st(st).get_simplices()])
		persistence_values = persistence_values[persistence_values<np.inf]
		self.range = np.quantile(persistence_values, [self.quantile, 1-self.quantile])
		return self
	def transform(self,X):
		if len(X) == 0:	return self
		if type(X[0]) is gd.SimplexTree: # If X contains simplextree : nothing to do
			if self.n_jobs > 1:
				warn("Cannot pickle simplextrees, reducing to 1 thread to compute the simplextrees")
			return [simplextree2hist(g,range=self.range, bins=self.bins, density=self.density) for g in tqdm(X, desc="Computing diagrams", disable=not self.progress)]
		else: # otherwise we assume that we retrieve simplextrees using f,data = X; simplextrees = (f(x) for x in data)
			to_st, data = X # asserts len(X) == 2
			def pickle_able_todo(x, **kwargs):
				simplextree = to_st(x)
				return simplextree2hist(simplextree=simplextree, **kwargs)
		return Parallel(n_jobs=self.n_jobs)(delayed(pickle_able_todo)(g,range=self.range, bins=self.bins, density=self.density) for g in tqdm(data, desc="Computing simplextrees and their diagrams", disable=not self.progress))




############# PERVEC
# Input list of [list of diagrams], outputs histogram of persitence values (x and y coord mixed) 
def dgm2pervec(dgms, range:tuple[float, float], bins:int)->np.ndarray: #TODO : Anything to histogram
	dgm_union = np.concatenate([dgm.flatten() for dgm in dgms]).flatten()
	return np.histogram(dgm_union, bins=bins,range=range)[0]
class Dgm2Histogram(BaseEstimator, TransformerMixin):
	def __init__(self, quantile:float=0., bins:int=100, n_jobs:int=1):
		super().__init__()
		self.range:np.ndarray | None=None
		self.quantile:float=quantile
		self.bins:int=bins
		self.n_jobs=n_jobs
	def fit(self, X, y=None): # X:list[diagrams]
		persistence_values = np.concatenate([dgm.flatten() for dgms in X for dgm in dgms], axis=0).flatten()
		persistence_values = persistence_values[persistence_values<np.inf]
		self.range = np.quantile(persistence_values, [self.quantile, 1-self.quantile])
		return self
	def transform(self,X):
		return Parallel(n_jobs=self.n_jobs)(delayed(dgm2pervec)(g,range=self.range, bins=self.bins) for g in X)







################# SignedMeasureImage
class Dgms2SignedMeasureImage(BaseEstimator, TransformerMixin):
	def __init__(self, ranges:None|Iterable[Iterable[float]]=None, resolution:int=100, quantile:float=0, bandwidth:float=1, kernel:str="gaussian") -> None:
		super().__init__()
		self.ranges=ranges
		self.resolution=resolution
		self.quantile = quantile
		self.bandwidth = bandwidth
		self.kernel = kernel
	def fit(self, X, y=None): # X:list[diagrams]
		num_degrees = len(X[0])
		persistence_values = [np.concatenate([dgms[i].flatten() for dgms in X], axis=0) for i in range(num_degrees)] # values per degree
		persistence_values = [degrees_values[(-np.inf<degrees_values) * (degrees_values<np.inf)] for degrees_values in persistence_values] # non-trivial values
		quantiles = [np.quantile(degree_values, [self.quantile, 1-self.quantile]) for degree_values in persistence_values] # quantiles 
		self.ranges = np.array([np.linspace(start=[a], stop=[b], num=self.resolution) for a,b in quantiles])
		return self

	def _dgm2smi(self, dgms:Iterable[np.ndarray]):
		smi = np.concatenate(
				[
					KernelDensity(bandwidth=self.bandwidth, kernel=self.kernel).fit(dgm[:,[0]]).score_samples(range)
					- KernelDensity(bandwidth=self.bandwidth).fit(dgm[:,[1]]).score_samples(range)
					for dgm, range in zip(dgms, self.ranges)
				],
			axis=0)
		return smi
		
	def transform(self,X): # X is a list (data) of list of diagrams
		assert self.ranges is not None
		out = Parallel(n_jobs=1, prefer="threads")(
			delayed(Dgms2SignedMeasureImage._dgm2smi)(self=self, dgms=dgms)
			for dgms in X
			)

		return out



################# SignedMeasureHistogram
class Dgms2SignedMeasureHistogram(BaseEstimator, TransformerMixin):
	def __init__(self, ranges:None|list[tuple[float,float]]=None, bins:int=100, quantile:float=0) -> None:
		super().__init__()
		self.ranges=ranges
		self.bins=bins
		self.quantile = quantile
	def fit(self, X, y=None): # X:list[diagrams]
		num_degrees = len(X[0])
		persistence_values = [np.concatenate([dgms[i].flatten() for dgms in X], axis=0) for i in range(num_degrees)] # values per degree
		persistence_values = [degrees_values[(-np.inf<degrees_values) * (degrees_values<np.inf)] for degrees_values in persistence_values] # non-trivial values
		self.ranges = [np.quantile(degree_values, [self.quantile, 1-self.quantile]) for degree_values in persistence_values] # quantiles 
		return self
	def transform(self,X): # X is a list (data) of list of diagrams
		assert self.ranges is not None
		out = [
			np.concatenate(
				[np.histogram(dgm[:,0], bins=self.bins,range=range)[0] - np.histogram(dgm[:,1], bins=self.bins,range=range)[0]
				for dgm, range in zip(dgms, self.ranges)]
			)
		for dgms in X]
		return out








################## Signed Measure Kernel 1D
# input : list of [list of diagrams], outputs: the kernel to feed to an svm

# TODO : optimize ?
## TODO : np.triu
class Dgms2SignedMeasureDistance(BaseEstimator, TransformerMixin):
	def __init__(self, n_jobs:int=1, distance_matrix_path:str|None=None, progress:bool = False) -> None:
		super().__init__()
		self.degrees:list[int]|None=None
		self.X:None|list[np.ndarray] = None
		self.n_jobs=n_jobs
		self.distance_matrix_path = distance_matrix_path
		self.progress=progress
	def fit(self, X:list[np.ndarray], y=None):
		if len(X) <= 0:
			warn("Fit a nontrivial vector")
			return
		self.X = X
		self.degrees = list(range(len(X[0]))) # Assumes that all x \in X have the same number of diagrams
		return self
	
	@staticmethod
	def wasserstein_1(a:np.ndarray,b:np.ndarray)->float:
		return np.abs(np.sort(a) - np.sort(b)).mean() # norm 1
	@staticmethod
	def OSWdistance(mu:list[np.ndarray], nu:list[np.ndarray], dim:int)->float:
		return Dgms2SignedMeasureDistance.wasserstein_1(np.hstack([mu[dim][:,0], nu[dim][:,1]]), np.hstack([nu[dim][:,0], mu[dim][:,1]])) # TODO : check: do we want to sum the kernels or the distances ? add weights ?
	@staticmethod
	def _ds(mu:list[np.ndarray], nus:list[list[np.ndarray]], dim:int): # mu and nu are lists of diagrams seen as signed measures (birth = +, death = -)
		return [Dgms2SignedMeasureDistance.OSWdistance(mu,nu, dim) for nu in nus]
	
	def transform(self,X): # X is a list (data) of list of diagrams
		if self.X is None or self.degrees is None:
			warn("Fit first !")
			return np.array([[]])
		# Cannot use sklearn / scipy, measures don't have the same size, -> no numpy array
		# from sklearn.metrics import pairwise_distances
		# distances = pairwise_distances(X, self.X, metric = OSWdistance, n_jobs=self.n_jobs)
		# from scipy.spatial.distance import cdist
		# distances = cdist(X, self.X, metric=self.OSWdistance)
		distances_matrices = []
		if not self.distance_matrix_path is None:
			for degree in self.degrees:
				with tqdm(X, desc=f"Computing distance matrix of degree {degree}") as diagrams_iterator:
					matrix_path = f"{self.distance_matrix_path}_{degree}"
					if exists(matrix_path):
						distance_matrix = np.load(open(matrix_path, "rb"))
					else:
						distance_matrix = np.array(Parallel(n_jobs=self.n_jobs)(delayed(self._ds)(mu, self.X, degree) for mu in diagrams_iterator))
						np.save(open(matrix_path, "wb"), distance_matrix)
					distances_matrices.append(distance_matrix)
		else:
			for degree in self.degrees:
				with tqdm(X, desc=f"Computing distance matrix of degree {degree}") as diagrams_iterator:
					distances_matrices.append(np.array(Parallel(n_jobs=self.n_jobs, prefer="threads")(delayed(self._ds)(mu, self.X, degree) for mu in diagrams_iterator)))
		return np.asarray(distances_matrices)
		# kernels = [np.exp(-distance_matrix / (2*self.sigma**2)) for distance_matrix in distances_matrices]
		# return np.sum(kernels, axis=0)





## Wrapper for SW, in order to take as an input a list of (list of diagrams)
class Dgms2SWK(BaseEstimator, TransformerMixin):
	def __init__(self, num_directions:int=10, bandwidth:float=1.0, n_jobs:int=1, distance_matrix_path:str|None = None, progress:bool = False) -> None:
		super().__init__()
		self.num_directions:int=num_directions
		self.bandwidth:float = bandwidth
		self.n_jobs=n_jobs
		self.SW_:list = []
		self.distance_matrix_path = distance_matrix_path
		self.progress = progress
	def fit(self, X:list[list[np.ndarray]], y=None):
		# Assumes that all x \in X have the same size
		self.SW_ = [
			SlicedWassersteinDistance(num_directions=self.num_directions, n_jobs = self.n_jobs) for _ in range(len(X[0]))
		]
		for i, sw in enumerate(self.SW_):
			self.SW_[i]=sw.fit([dgms[i] for dgms in X]) # TODO : check : Not sure copy is necessary here
		return self
	def transform(self,X)->np.ndarray:
		if not self.distance_matrix_path is None:
			distance_matrices = []
			for i in range(len(self.SW_)):
				SW_i_path = f"{self.distance_matrix_path}_{i}"
				if exists(SW_i_path):
					distance_matrices.append(np.load(open(SW_i_path, "rb"))) 
				else:
					distance_matrix = self.SW_[i].transform([dgms[i] for dgms in X])
					np.save(open(SW_i_path, "wb"), distance_matrix)
		else:
			distance_matrices = [sw.transform([dgms[i] for dgms in X]) for i, sw in enumerate(self.SW_)]
		kernels = [np.exp(-distance_matrix / (2*self.bandwidth**2)) for distance_matrix in distance_matrices]
		return np.sum(kernels, axis=0) # TODO fix this, we may want to sum the distances instead of the kernels. 


class Dgms2SlicedWassersteinDistanceMatrices(BaseEstimator, TransformerMixin):
	def __init__(self, num_directions:int=10, n_jobs:int=1) -> None:
		super().__init__()
		self.num_directions:int=num_directions
		self.n_jobs=n_jobs
		self.SW_:list = []
	def fit(self, X:list[list[np.ndarray]], y=None):
		# Assumes that all x \in X have the same size
		self.SW_ = [
			SlicedWassersteinDistance(num_directions=self.num_directions, n_jobs = self.n_jobs) for _ in range(len(X[0]))
		]
		for i, sw in enumerate(self.SW_):
			self.SW_[i]=sw.fit([dgms[i] for dgms in X]) # TODO : check : Not sure copy is necessary here
		return self
	
	@staticmethod
	def _get_distance(diagrams, SWD):
		return SWD.transform(diagrams)
	def transform(self,X):
		distance_matrices = Parallel(n_jobs = self.n_jobs)(delayed(self._get_distance)([dgms[degree] for dgms in X], swd) for degree, swd in enumerate(self.SW_))		
		return np.asarray(distance_matrices)



# Gudhi simplexTree to list of diagrams
class SimplexTree2Dgm(BaseEstimator, TransformerMixin):
	def __init__(self, degrees:list[int]|None = None, extended:list[int]|bool=[], n_jobs=1, progress:bool=False, threshold:float=np.inf) -> None:
		super().__init__()
		self.extended:list[int]|bool = False if not extended else extended if type(extended) is list else [0,2,5,7] # extended persistence.
		# There are 4 diagrams per dimension then, the list of ints acts as a filter, on which to consider,
		#  eg., [0,2, 5,7] is Ord0, Ext+0, Rel1, Ext-1
		self.degrees:list[int] = degrees if degrees else list(range((max(self.extended) // 4)+1))  if self.extended else [0] # homological degrees
		self.n_jobs=n_jobs 
		self.progress = progress # progress bar
		self.threshold = threshold # Threshold value
		return
	def fit(self, X:list[gd.SimplexTree], y=None):
		if self.threshold <= 0:
			self.threshold = max( (abs(f) for simplextree in get_simplextrees(X) for s,f in simplextree.get_simplices()) )  ## MAX FILTRATION VALUE
			print(f"Setting threshold to {self.threshold}.")
		return self
	def transform(self,X:list[gd.SimplexTree]):
		# Todo computes the diagrams
		def reshape(dgm:np.ndarray|list)->np.ndarray:
			out = np.array(dgm) if len(dgm) > 0 else np.empty((0,2)) 
			if self.threshold != np.inf:
				out[out>self.threshold] = self.threshold
				out[out<-self.threshold] = -self.threshold
			return out
		def todo_standard(st):
			st.compute_persistence()
			return [reshape(st.persistence_intervals_in_dimension(d)) for d in self.degrees]
		def todo_extended(st):
			st.extend_filtration()
			dgms = st.extended_persistence()
#			print(dgms, self.degrees)
			return [reshape([bar for j,dgm in enumerate(dgms) for d, bar in dgm if d in self.degrees and j+4*d in self.extended])]
		todo = todo_extended if self.extended else todo_standard

		if isinstance(X[0],gd.SimplexTree): # simplextree aren't pickleable, no parallel
			# if self.n_jobs != 1:	warn("Cannot parallelize. Use dtype=None in previous pipe.")
			return Parallel(n_jobs=self.n_jobs, prefer="threads")(delayed(todo)(x) for x in tqdm(X, disable=not self.progress, desc="Computing diagrams"))
		else:
			to_st = X[0]# if to_st is None else to_st 
			dataset = X[1]# if to_st is None else X
			pickleable_todo = lambda x : todo(to_st(x))
			return Parallel(n_jobs=self.n_jobs, prefer="threads")(delayed(pickleable_todo)(x) for x in tqdm(dataset, disable=not self.progress, desc="Computing simplextrees and diagrams"))
		warn("Bad input.")
		return

# Shuffles a diagram shaped array. Input : list of (list of diagrams), output, list of (list of shuffled diagrams)
class DiagramShuffle(BaseEstimator, TransformerMixin):
	def __init__(self, ) -> None:
		super().__init__()
		return
	def fit(self, X:list[list[np.ndarray]], y=None):
		return self
	def transform(self,X:list[list[np.ndarray]]):
		def shuffle(dgm):
			shape = dgm.shape
			dgm = dgm.flatten()
			np.random.shuffle(dgm)
			dgm = dgm.reshape(shape)
			return dgm
		def todo(dgms):
			return [shuffle(dgm) for dgm in dgms]
		return [todo(dgm) for dgm in X]


class Dgms2Landscapes(BaseEstimator, TransformerMixin):
	def __init__(self, num:int=5, resolution:int=100,  n_jobs:int=1) -> None:
		super().__init__()
		self.degrees:list[int] = []
		self.num:int= num
		self.resolution:int = resolution
		self.landscapes:list[Landscape]= []
		self.n_jobs=n_jobs
		return
	def fit(self, X, y=None):
		if len(X) == 0:	return self
		self.degrees = list(range(len(X[0])))
		self.landscapes = []
		for dim in self.degrees:
			self.landscapes.append(Landscape(num_landscapes=self.num,resolution=self.resolution).fit([dgms[dim] for dgms in X]))
		return self
	def transform(self,X):
		if len(X) == 0:	return []
		return np.concatenate([landscape.transform([dgms[degree] for dgms in X]) for degree, landscape in enumerate(self.landscapes)], axis=1)

class Dgms2Image(BaseEstimator, TransformerMixin):
	def __init__(self, bandwidth:float=1, resolution:tuple[int,int]=(20,20),  n_jobs:int=1) -> None:
		super().__init__()
		self.degrees:list[int] = []
		self.bandwidth:float= bandwidth
		self.resolution = resolution
		self.PI:list[PersistenceImage]= []
		self.n_jobs=n_jobs
		return
	def fit(self, X, y=None):
		if len(X) == 0:	return self
		self.degrees = list(range(len(X[0])))
		self.PI = []
		for dim in self.degrees:
			self.PI.append(PersistenceImage(bandwidth=self.bandwidth,resolution=self.resolution).fit([dgms[dim] for dgms in X]))
		return self
	def transform(self,X):
		if len(X) == 0:	return []
		return np.concatenate([pers_image.transform([dgms[degree] for dgms in X]) for degree, pers_image in enumerate(self.PI)], axis=1)


