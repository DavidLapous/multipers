from mma import *
import gudhi as gd
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool, Manager, cpu_count
from joblib import Parallel, delayed
from sklearn.neighbors import KernelDensity
from joblib import parallel_backend
from numpy.polynomial.polynomial import polyfit
from numpy.linalg import norm
from random import random, choice, shuffle, seed
import random
from math import floor
from sympy.ntheory import factorint
from tqdm import tqdm

from os.path import exists
from os import remove


ncores = cpu_count()

def compute_heatmap(time_matrix,set_of_npts, set_of_nlines, save = "", dimension=-1):
	plt.imshow(np.flip(time_matrix,0), cmap='hot', interpolation='nearest',
		   extent=[set_of_nlines[0], set_of_nlines[-1], set_of_npts[0], set_of_npts[-1]],
		   aspect=(set_of_nlines[-1] - set_of_nlines[0])/(set_of_npts[-1] - set_of_npts[0]))
	if dimension <=2:
		plt.xlabel("Number of lines")
	elif dimension == 3:
		plt.xlabel("Square root of the number of lines")
	elif dimension == 4:
		plt.xlabel("Cubic root of the number of lines")
	else:
		plt.xlabel(f"{dimension}-root of the number of lines")
	plt.ylabel("Number of points")
	plt.colorbar()
	if save != "":
		plt.savefig(save, dpi=500)
	plt.show()


def synthetic_random_benchmark(number_of_points, number_of_tries, dimension_of_points, number_of_loglines, persistence_dimension=2, filtration="rips", verbose = True, max_dimension=3, parallel_tries=True):
	number_of_loglines = (int)(number_of_loglines)
	assert(number_of_loglines > 0)
	X = np.random.uniform(low=0, high=2, size=[number_of_points,dimension_of_points])
	if filtration == "alpha":
		simplextree = gd.AlphaComplex(points=X).create_simplex_tree()
	if filtration == "rips":
		simplextree = gd.RipsComplex(points=X, max_edge_length= 0.3).create_simplex_tree(max_dimension = max_dimension)
#	 if max_dimension <= 1:
#		 simplextree.collapse_edges()
	if verbose:
		print("Number of simplices :", simplextree.num_simplices(),"and maximum dimension :", simplextree.dimension(), flush=True)
	boundary = simplextree_to_sparse_boundary(simplextree)
	filters = [np.random.uniform(low = 0, high = 1, size =[number_of_points,1]) for _ in range(persistence_dimension)]

	precision = 1
	basepoint = [0 for _ in range(persistence_dimension)]
	box = nlines_precision_box(number_of_loglines, basepoint, precision, square=True)
	if verbose:
		print("Precision :",precision, flush=True)
	times = []
	if parallel_tries:
		times = Parallel(n_jobs=min(ncores,number_of_tries))(delayed(time_approx)(boundary, filters, precision, box, threshold = False, multithread = 0, verbose = verbose) for i in range(number_of_tries))
	else:
		for i in range(number_of_tries):
			times+=[time_approx(boundary, filters, precision, box, threshold = False, multithread = 0)]
	return np.mean(times), np.std(times) / np.sqrt(number_of_tries), len(boundary)


def synthetic_random_benchmark_time(number_of_points, number_of_tries, dimension_of_points, number_of_lines, persistence_dimension=2, filtration="rips", verbose = True, max_dimension=3):
	return synthetic_random_benchmark(number_of_points, number_of_tries, dimension_of_points, number_of_lines, persistence_dimension=2, filtration="rips", verbose = True, max_dimension=3)[0]


def noisy_annulus(r1=1, r2=2, n=50, seed=None, dim=2):
    set =[]
    if seed != None :
        np.random.seed(seed)
    while len(set)<n:
        x=np.random.uniform(low=-r2, high=r2, size=dim)
        if np.linalg.norm(x) > r1 and np.linalg.norm(x) < r2:
            set.append(x)
    return set


def nlines_precision_box(nlines, basepoint, scale, square = False):
	import math
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
		return time_approx(boundary, filters, precision, box, threshold = False, multithread = 0), 0, simplex_tree.num_simplices()
	times = Parallel(n_jobs=min(ncores,ntries))(
		delayed(time_approx)(
			boundary, filters, precision, box, threshold = False, multithread = 0
		) for i in range(ntries))
	return np.mean(times), np.std(times), simplex_tree.num_simplices()



def convergence_image(boundary, filters,max_precision, bandwidth, verbose=False, box=[], save=False, show_img = True, num=100, min_precision=1, resolution=[200,200], p=2):
	baseline = persistence_image_2d(boundary, filters, precision = max_precision,bandwidth=bandwidth, verbose = verbose, plot = show_img, resolution = resolution,box=box,  save=save)
	errors = []
	precisions = np.logspace(np.log10(min_precision),np.log10(max_precision), num)
	for precision in tqdm(precisions):
		img = persistence_image_2d(boundary, filters, precision = precision,bandwidth=bandwidth, plot=False, resolution=resolution, box=box)
		error = []
		for dimension in range(len(baseline)):
			if p == np.inf:
				e = np.max(np.abs(img[dimension] - baseline[dimension]))
			else:
				e = norm(img[dimension] - baseline[dimension],p)
			error.append(e)
		errors.append(error)
	errors = np.array(errors)
	fig,ax=plt.subplots()
	for dimension in range(len(baseline)):
		plt.plot(precisions,errors[:,dimension], label = f"Dimension {dimension}")
	# Plot
	plt.xlabel("Precision")
	if p == np.inf:
		plt.ylabel("Infinity norm error")
	elif p==2:
		plt.ylabel("Quadratic error")
	else:
		plt.ylabel(f"L{p} error")
	plt.legend()
	ax.invert_xaxis()
	ax.set_xscale('log')
	if save:
		plt.savefig(save+"_convergence.svg")
	return errors


def convert_to_rivet(simplextree, kde, X,*, dimension=1, verbose = True):
	if exists("rivet_dataset.txt"):
		remove("rivet_dataset.txt")
	file = open("rivet_dataset.txt", "a")
	file.write("--datatype bifiltration\n")
	file.write(f"--homology {dimension}\n")
	file.write("--xlabel time of appearance\n")
	file.write("--ylabel density\n\n")

	to_write = ""
	if verbose:
		for s,f in tqdm(simplextree.get_simplices()):
			for i in s:
				to_write += str(i) + " "
			to_write += "; "+ str(f) + " " + str(np.max(-kde.score_samples(X[s,:])))+'\n'
	else:
		for s,f in simplextree.get_simplices():
			for i in s:
				to_write += str(i) + " "
			to_write += "; "+ str(f) + " " + str(np.max(-kde.score_samples(X[s,:])))+'\n'
	file.write(to_write)
	file.close()

def generate_annulus_dataset(n_pts = 1000, n_outliers = 200, seed=None, dim=2):
	if seed != None:
		np.random.seed(seed)
	X = np.block([[np.array(noisy_annulus(0.7,1.6,n_pts, dim=dim))],[np.random.uniform(low=-2,high=2,size=(n_outliers,dim))]])
	simplextree = gd.AlphaComplex(points=X).create_simplex_tree(max_alpha_square=2)
	kde = KernelDensity(kernel='gaussian', bandwidth=.3).fit(X)
	filtration_dens = -np.array(kde.score_samples(X))
	filtration_rips = np.array([simplextree.filtration(s) for s,_ in simplextree.get_simplices()])
	return X,simplextree,kde,filtration_dens,filtration_rips









