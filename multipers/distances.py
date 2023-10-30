import torch
import ot
import numpy as np
from multipers.simplex_tree_multi import SimplexTreeMulti
from multipers.multiparameter_module_approximation import PyMultiDiagrams, PyModule

def sm2diff(sm1,sm2):
	if isinstance(sm1[0],np.ndarray):
		backend_concatenate = lambda a,b : np.concatenate([a,b], axis=0)
		backend_tensor = lambda x : np.asarray(x, dtype=int)
	elif isinstance(sm1[0],torch.Tensor):
		backend_concatenate = lambda a,b : torch.concatenate([a,b], dim=0)
		backend_tensor = lambda x :torch.tensor(x).type(torch.int)
	else:
		raise Exception("Invalid backend. Numpy or torch.")
	pts1,w1 = sm1
	pts2,w2 = sm2
	pos_indices1 = backend_tensor([i for i,w in enumerate(w1) for _ in range(w) if w>0])
	pos_indices2 = backend_tensor([i for i,w in enumerate(w2) for _ in range(w) if w>0]) 
	neg_indices1 = backend_tensor([i for i,w in enumerate(w1) for _ in range(-w) if w<0])
	neg_indices2 = backend_tensor([i for i,w in enumerate(w2) for _ in range(-w) if w<0])
	x = backend_concatenate(pts1[pos_indices1],pts2[neg_indices2])
	y = backend_concatenate(pts1[neg_indices1],pts2[pos_indices2])
	return x,y

def sm_distance(sm1,sm2, reg=0,reg_m=0, numItermax=10000, p=1):
	x,y = sm2diff(sm1,sm2)
	loss = ot.dist(x,y, metric='sqeuclidean', p=2) # only euc + sqeuclidian are implemented in pot for the moment with torch backend # TODO : check later
	if isinstance(x,np.ndarray):
		empty_tensor = np.array([]) # uniform weights
	elif isinstance(x,torch.Tensor):
		empty_tensor = torch.tensor([]) # uniform weights

	if reg == 0:
		return ot.lp.emd2(empty_tensor,empty_tensor,M=loss)*len(x)
	if reg_m == 0:
		return ot.sinkhorn2(a=empty_tensor,b=empty_tensor,M=loss,reg=reg, numItermax=numItermax)
	return ot.sinkhorn_unbalanced2(a=empty_tensor,b=empty_tensor,M=loss,reg=reg, reg_m=reg_m, numItermax=numItermax)
	# return ot.sinkhorn2(a=onesx,b=onesy,M=loss,reg=reg, numItermax=numItermax)
	# return ot.bregman.empirical_sinkhorn2(x,y,reg=reg)







def estimate_matching(b1:PyMultiDiagrams, b2:PyMultiDiagrams):
	assert(len(b1) == len(b2))
	from gudhi.bottleneck import bottleneck_distance
	def get_bc(b:PyMultiDiagrams, i:int)->np.ndarray:
		temp = b[i].get_points()
		out = np.array(temp)[:,:,0] if len(temp) >0  else np.empty((0,2)) # GUDHI FIX
		return out
	return max((bottleneck_distance(get_bc(b1,i), get_bc(b2,i)) for i in range(len(b1))))


#### Functions to estimate precision
def estimate_error(st:SimplexTreeMulti, module:PyModule, degree:int, nlines:int = 100, verbose:bool =False):
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
	from time import perf_counter
	parameter = 0 

	def _get_bc_ST(st, basepoint, degree:int):
		"""
		Slices an mma simplextree to a gudhi simplextree, and compute its persistence on the diagonal line crossing the given basepoint.
		"""
		gst = st.project_on_line(basepoint=basepoint, parameter=parameter) # we consider only the 1rst coordinate (as )
		gst.compute_persistence()
		return gst.persistence_intervals_in_dimension(degree)
	from gudhi.bottleneck import bottleneck_distance
	low, high = module.get_box()
	nfiltration = len(low)
	basepoints = np.random.uniform(low=low, high=high, size=(nlines,nfiltration))
	# barcodes from module
	print("Computing mma barcodes...", flush=1, end="") if verbose else None
	time = perf_counter()
	bcs_from_mod = module.barcodes(degree=degree, basepoints = basepoints).get_points()
	print(f"Done. {perf_counter() - time}s.") if verbose else None
	clean = lambda dgm : np.array([[birth[parameter], death[parameter]] for birth,death in dgm if len(birth) > 0 and  birth[parameter] != np.inf])
	bcs_from_mod = [clean(dgm) for dgm in bcs_from_mod] # we only consider the 1st coordinate of the barcode
	# Computes gudhi barcodes
	from tqdm import tqdm
	bcs_from_gudhi = [_get_bc_ST(st,basepoint=basepoint, degree=degree) for basepoint in tqdm(basepoints, disable= not verbose, desc = "Computing gudhi barcodes")]
	return max((bottleneck_distance(a,b) for a,b in tqdm(zip(bcs_from_mod, bcs_from_gudhi), disable = not verbose, total=nlines, desc="Computing bottleneck distances")))
