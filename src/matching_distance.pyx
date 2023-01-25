

def estimate_matching(b1:PyMultiDiagrams, b2:PyMultiDiagrams):
	assert(len(b1) == len(b2))
	from gudhi.bottleneck import bottleneck_distance
	def get_bc(b:PyMultiDiagrams, i:int)->np.ndarray:
		temp = b[i].get_points()
		out = np.array(temp)[:,:,0] if len(temp) >0  else np.empty((0,2)) # GUDHI FIX
		return out
	return max((bottleneck_distance(get_bc(b1,i), get_bc(b2,i)) for i in range(len(b1))))


#### Functions to estimate precision
def estimate_error(st:SimplexTree, module:PyModule, degree:int, nlines = 100, verbose:bool =False):
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
		gst = st.to_gudhi(basepoint=basepoint, parameter=parameter) # we consider only the 1rst coordinate (as )
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
	bcs_from_gudhi = [_get_bc_ST(st,basepoint=basepoint, degree=degree) for basepoint in tqdm(basepoints, disable= not verbose, desc = "Computing gudhi barcodes")]
	return max((bottleneck_distance(a,b) for a,b in tqdm(zip(bcs_from_mod, bcs_from_gudhi), disable = not verbose, total=nlines, desc="Computing bottleneck distances")))
