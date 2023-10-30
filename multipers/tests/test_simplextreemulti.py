import multipers as mp
import numpy as np
mp.simplex_tree_multi.SAFE_CONVERSION=True
def test_1():
	st = mp.SimplexTreeMulti(num_parameters=2)
	st.insert([0],[0,1])
	st.insert([1],[1,0])
	st.insert([0,1],[1,1])
	it = [([0, 1], [1.0, 1.0]), ([0], [0.0, 1.0]), ([1], [1.0, 0.0])]
	assert np.all(
		[np.array_equal(a,b) for x,y in zip(it,st.get_simplices()) for a,b in zip(x,y)]
	)

def test_2():
	from gudhi.rips_complex import RipsComplex
	st2 = RipsComplex(points=[[0,1], [1,0], [0,0]]).create_simplex_tree()
	st2 = mp.SimplexTreeMulti(st2, num_parameters=3, default_values=[1,2]) # the gudhi filtration is placed on axis 0
	
	it = (([0, 1], [1.4142135381698608, 1.0, 2.0]),
	([0, 2], [1.0, 1.0, 2.0]),
	([0], [0.0, 1.0, 2.0]),
	([1, 2], [1.0, 1.0, 2.0]),
	([1], [0.0, 1.0, 2.0]),
	([2], [0.0, 1.0, 2.0]))
	assert np.all(
		[np.array_equal(a,b) for x,y in zip(it,st2.get_simplices()) for a,b in zip(x,y)]
	)
