import multipers.ml.point_clouds as mmp
import numpy as np
import multipers as mp

mp.simplex_tree_multi.SAFE_CONVERSION=True

def test_throw_test():
	pts = np.array([[1,1],[2,2]], dtype=np.float32)
	st = mmp.PointCloud2SimplexTree(masses=[.1]).fit_transform([pts])[0][0]
	assert isinstance(st, mp.simplex_tree_multi.SimplexTreeMulti)
	st= mmp.PointCloud2SimplexTree(bandwidths=[-.1], complex="alpha").fit_transform([pts])[0][0]
	assert isinstance(st, mp.simplex_tree_multi.SimplexTreeMulti)
	st1,st2= mmp.PointCloud2SimplexTree(bandwidths=[.1], masses=[.1]).fit_transform([pts])[0]
	assert isinstance(st1, mp.simplex_tree_multi.SimplexTreeMulti)
	assert isinstance(st2, mp.simplex_tree_multi.SimplexTreeMulti)
	## ensures it doesn't throw
	assert isinstance( st.persistence_approximation(), mp.multiparameter_module_approximation.PyModule)
	assert mp.signed_measure(st, degree=None, invariant='euler')[0][0].ndim ==2 
	assert mp.signed_measure(st, degree=0,    invariant='hilbert')[0][0].ndim ==2
	# assert mp.signed_measure(st, degree=0,    invariant='rank')[0][0].ndim ==2 # This needs torch which is too heavy
