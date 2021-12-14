from custom_vineyards import *
import gudhi as gd
import numpy as np
import matplotlib.pyplot as plt
from multipers import *
from joblib import Parallel, delayed
from multiprocessing import Pool, Manager, cpu_count
from sklearn.neighbors import KernelDensity
from joblib import parallel_backend
from numpy.polynomial.polynomial import polyfit

ncores = cpu_count()

def compute_heatmap(time_matrix,set_of_npts, set_of_nlines, save = ""):
	plt.imshow(np.flip(time_matrix,0), cmap='hot', interpolation='nearest',
		   extent=[set_of_nlines[0], set_of_nlines[-1], set_of_npts[0], set_of_npts[-1]],
		   aspect=(set_of_nlines[-1] - set_of_nlines[0])/(set_of_npts[-1] - set_of_npts[0]))
	plt.xlabel("Number of lines")
	plt.ylabel("Number of points")
	plt.colorbar()
	if save != "":
		plt.savefig(save, dpi=500)
	plt.show()


def synthetic_random_benchmark(number_of_points, number_of_tries, dimension_of_points, number_of_lines, persistence_dimension=2, filtration="rips", verbose = True, max_dimension=3, parallel_tries=True):
	number_of_lines = (int)(number_of_lines)
	X = np.random.uniform(low=0, high=2, size=[number_of_points,dimension_of_points])
	if filtration == "alpha":
		simplextree = gd.AlphaComplex(points=X).create_simplex_tree(max_alpha_square=2)
	if filtration == "rips":
		simplextree = gd.RipsComplex(points=X, max_edge_length= 0.3).create_simplex_tree(max_dimension = max_dimension)
#	 if max_dimension <= 1:
#		 simplextree.collapse_edges()
	if verbose:
		print("Number of simplices :", simplextree.num_simplices(),"and maximum dimension :", simplextree.dimension(), flush=True)
	boundary = simplextree_to_sparse_boundary(simplextree)
	filters = [np.random.uniform(low = 0, high = 1, size =[number_of_points,1]) for _ in range(persistence_dimension)]
	box = [[0 for _ in range(persistence_dimension)], [2 for _ in range(persistence_dimension)]]
	precision =  4/ number_of_lines**(1/(persistence_dimension-1))
	if verbose:
		print("Precision :",precision, flush=True)
	times = []
	if parallel_tries:
		times = Parallel(n_jobs=min(ncores,number_of_tries))(delayed(time_vine_alt)(boundary, filters, precision, box, threshold = False, multithread = 0) for i in range(number_of_tries))
	else:
		for i in range(number_of_tries):
			times+=[time_vine_alt(boundary, filters, precision, box, threshold = False, multithread = 0)]
	return np.mean(times), np.std(times) / np.sqrt(number_of_tries), len(boundary)


def synthetic_random_benchmark_time(number_of_points, number_of_tries, dimension_of_points, number_of_lines, persistence_dimension=2, filtration="rips", verbose = True, max_dimension=3):
	return synthetic_random_benchmark(number_of_points, number_of_tries, dimension_of_points, number_of_lines, persistence_dimension=2, filtration="rips", verbose = True, max_dimension=3)[0]


def noisy_annulus(r1=1, r2=2, n=50):
	set =[]
	while len(set)<n:
		x,y=2*r2*random.random()-r2,2*r2*random.random()-r2
		if r1**2 < x**2 + y**2 < r2**2:
			set += [[x,y]]
	return set[:]


def bifiltration_densisty(simplex_tree, density):
	list_splx = simplex_tree.get_filtration()
	subdivision = barycentric_subdivision(simplex_tree)
	filtration_rips = np.array([subdivision.filtration(s) for s,_ in subdivision.get_skeleton(0)])[:,None]
	for s,_ in simplex_tree.get_filtration():
		if len(s) == 1:
			simplex_tree.assign_filtration(s, -density[s[0]])
		else:
			simplex_tree.assign_filtration(s, -1e10)
	simplex_tree.make_filtration_non_decreasing()
	subdivision = barycentric_subdivision(simplex_tree, list_splx)
	filtration_dens = np.array([subdivision.filtration(s) for s,_ in subdivision.get_skeleton(0)])[:,None]
	filtration = np.hstack([filtration_rips, filtration_dens])
	return subdivision, filtration





def density_persistence_benchmark_old(X, number_of_line, n_tries=3, gaussian_var=0.3, filtration = "rips", max_dimension=1,max_edge_length=0.3, max_alpha_square=2):
	if filtration=="alpha" :
		simplex_tree = gd.AlphaComplex(points=X).create_simplex_tree(max_alpha_square=max_alpha_square)
	if filtration == "rips":
		simplex_tree = gd.RipsComplex(points=X,max_edge_length= max_edge_length).create_simplex_tree(max_dimension=max_dimension)
	if simplex_tree.dimension() <=1:
		simplex_tree.collapse_edges()
	kde = KernelDensity(kernel='gaussian', bandwidth=.3).fit(X)
	density = kde.score_samples(X)
	subdivision, filtration = bifiltration_densisty(simplex_tree, density)
	boundary = simplextree_to_sparse_boundary(subdivision)
	box = [[min(filtration[:,0]),min(filtration[:,1])],[max(filtration[:,0]),max(filtration[:,1])]]
	precision = (box[1][1] + box[1][0] - box[0][1] - box[0][0]) / (number_of_line )
#	 print("Number of simplices", len(boundary), flush=1)
	times = Parallel(n_jobs=min(ncores,n_tries))(
		delayed(time_vine_alt)(
			boundary, filtration, precision, box, threshold = False, multithread = 0
		) for i in range(n_tries))
	return np.mean(times), np.std(times)





def density_persistence_benchmark(X, nlines, ntries=10, gaussian_var = 0.3, filtration = "alpha", max_dimension = 2, max_edge_length = 0.5,max_alpha_square=2):
	if filtration=="alpha" :
		simplex_tree = gd.AlphaComplex(points=X).create_simplex_tree(max_alpha_square=max_alpha_square)
	if filtration == "rips":
		simplex_tree = gd.RipsComplex(points=X,max_edge_length= max_edge_length).create_simplex_tree(max_dimension=max_dimension)
	if simplex_tree.dimension() <=1:
		simplex_tree.collapse_edges()
	complex_filtration = np.array([simplex_tree.filtration(s) for s,_ in simplex_tree.get_simplices()])
	kde = KernelDensity(kernel='gaussian', bandwidth=gaussian_var).fit(X)
	density_filtration = -np.array(kde.score_samples(X))
	boundary, filters = ls_boundary_density(simplex_tree, [complex_filtration, density_filtration], [0])
	box = [[min(filters[:,0]),min(filters[:,1])],[max(filters[:,0]),max(filters[:,1])]]
	precision = (box[1][1] + box[1][0] - box[0][1] - box[0][0]) / (nlines)
	if ntries <= 1:
		return time_vine_alt(boundary, filters, precision, box, threshold = False, multithread = 0), 0, simplex_tree.num_simplices()
	times = Parallel(n_jobs=min(ncores,ntries))(
		delayed(time_vine_alt)(
			boundary, filters, precision, box, threshold = False, multithread = 0
		) for i in range(ntries))
	return np.mean(times), np.std(times), simplex_tree.num_simplices()

