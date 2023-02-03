import numpy as np
import mma
import gudhi as gd
from sympy.ntheory import factorint 


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
	from numpy.random import uniform
	from numpy.linalg import norm

	set =[]
	while len(set)<n1:
		draw=uniform(low=-r2, high=r2, size=dim)
		if norm(draw) > r1 and norm(draw) < r2:
			set.append(draw)
	annulus = np.array(set) if center == None else np.array(set) + np.array(center)
	diffuse_noise = uniform(size=(n2,dim), low=-1.1*r2,high=1.1*r2)
	if center is not None:	diffuse_noise += np.array(center)
	return np.vstack([annulus, diffuse_noise])

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
	points = noisy_annulus(**kwargs)
	points = np.unique(points, axis=0)
	from sklearn.neighbors import KernelDensity
	kde = KernelDensity(bandwidth = 1).fit(points)
	st = gd.RipsComplex(points = points).create_simplex_tree()
	st = mma.from_gudhi(st, parameters = 2)
	st.collapse_edges(num=100)
	st.collapse_edges(num=100, strong=False, max_dimension = 2)
	f2 =  - kde.score_samples(points)
	st.fill_lowerstar(f2,parameter=1)
	mod = st.persistence(**kwargs)
	return mod

def nlines_precision_box(nlines, basepoint, scale, square = False):
	import math
	from random import choice, shuffle
	h = scale
	dim = len(basepoint)
	basepoint = np.array(basepoint, 'double')
	if square:
		# here we want n^dim-1 lines (n = nlines)
		n=nlines
		basepoint = np.array(basepoint, 'double')
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

