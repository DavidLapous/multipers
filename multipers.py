import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import time
import gudhi as gd
from sklearn.preprocessing import MinMaxScaler
from sklearn.base import BaseEstimator, TransformerMixin
import gudhi.representations as sktda
from sklearn.neighbors import KDTree
import math
import random
from concurrent import futures
from joblib import Parallel, delayed

from dionysus_vineyards import ls_vineyards as lsvine
#from custom_vineyards import ls_vineyards as lsvine

def DTM(X,query_pts,m):
	"""
	Code for computing distance to measure. Taken from https://github.com/GUDHI/TDA-tutorial/blob/master/Tuto-GUDHI-DTM-filtrations.ipynb
	"""    
	N_tot = X.shape[0]
	k = math.floor(m*N_tot)+1
	kdt = KDTree(X, leaf_size=30, metric='euclidean')
	NN_Dist, NN = kdt.query(query_pts, k, return_distance=True)
	DTM_result = np.sqrt(np.sum(NN_Dist*NN_Dist,axis=1) / k)
	return(DTM_result)


def recursive_insert(st, base_splx, splx, name_dict, filt):
	if len(splx) == 1:	st.insert(base_splx + [name_dict[tuple(splx)]], filt)
	else:
		for idx in range(len(splx)):
			coface = splx[:idx] + splx[idx+1:]
			recursive_insert(st, base_splx + [name_dict[tuple(splx)]], coface, name_dict, max(filt, st.filtration([name_dict[tuple(coface)]])))

def barycentric_subdivision(st, list_splx=None, use_sqrt=False, use_mean=False):
	"""
	Code for computing the barycentric subdivision of a Gudhi simplex tree.
	
	Inputs:
		st: input simplex tree
		list_splx: a list of simplices of st (useful if you want to give specific names for the barycentric subdivision simplices)
		use_sqrt: whether to take the square root for the barycentric subdivision filtration values (useful if st was computed from Gudhi AlphaComplex for instance)
		use_mean: whether to take the mean of the vertices for the new vertex value (useful for refining lower star for instance)
	Outputs:
		bary: barycentric subdivision of st
	"""
	bary = gd.SimplexTree()
	bary_splx = {}

	splxs = st.get_filtration() if list_splx is None else list_splx 
	count = 0
	for splx, f in splxs:
		if use_sqrt:	bary.insert([count], np.sqrt(f))
		elif use_mean:	bary.insert([count], np.mean([st.filtration([v]) for v in splx]))
		else:	bary.insert([count], f)
		bary_splx[tuple(splx)] = count
		count += 1

	for splx, f in st.get_filtration():
		if len(splx) == 1:	continue
		else:
			recursive_insert(bary, [], splx, bary_splx, bary.filtration([bary_splx[tuple(splx)]]))

	return bary

def gudhi_line_diagram(simplextree, F, homology=0, extended=False, essential=False, mode="Gudhi"):
	"""
	Wrapper code for computing the lower-star filtration of a simplex tree.

	Inputs:
		simplextree: input simplex tree
		F: array containing the filtration values of the simplex tree vertices
		homology: homological dimension
		extended: whether to compute extended persistence
		essential: whether to keep features with infinite ordinates
		mode: method for giving the simplex tree: either in the native Gudhi format ("Gudhi"), or as a list of numpy arrays containing the simplices in each dimension ("Numpy")
	Outputs:
		dgm: a Numpy array containing the persistence diagram points
	"""
	if mode == "Gudhi":
		st = gd.SimplexTree()
		for (s,_) in simplextree.get_filtration():	st.insert(s)
	elif mode == "Numpy":
		st = gd.SimplexTree()
		for ls in simplextree:
			for s in range(len(ls)):	st.insert([v for v in ls[s,:]])
	
	for (s,_) in st.get_filtration():	st.assign_filtration(s, -1e10)
	for (v,_) in st.get_skeleton(0):	st.assign_filtration(v, F[v[0]])

	st.make_filtration_non_decreasing()
	if extended:	
		st.extend_filtration()
		dgms = st.extended_persistence(min_persistence=1e-10)
		dgms = [dgm for dgm in dgms if len(dgm) > 0]
		ldgm = [[np.array([[    min(pt[1][0], pt[1][1]), max(pt[1][0], pt[1][1])    ]]) for pt in dgm if pt[0] == homology] for dgm in dgms]
		ldgm = [ np.vstack(dgm) for dgm in ldgm if len(dgm) > 0]
		dgm = np.vstack(ldgm) if len(ldgm) > 0 else np.empty([0,2])
	else:		
		st.persistence()
		dgm = st.persistence_intervals_in_dimension(homology)

	if not essential:	dgm = dgm[np.ravel(np.argwhere(dgm[:,1] != np.inf)),:]
	return dgm


def gudhi_matching(dgm1, dgm2):
	"""
	Code for computing the matching associated to the 2-Wasserstein distance.

	Inputs:
		dgm1: first persistence diagram
		dgm2: second persistence diagram
	Outputs:
		mtc: a Numpy array containing the partial matching between the inputs
	"""
	import gudhi.wasserstein
	f1, i1, f2, i2 = np.ravel(np.argwhere(dgm1[:,1] != np.inf)), np.ravel(np.argwhere(dgm1[:,1] == np.inf)), np.ravel(np.argwhere(dgm2[:,1] != np.inf)), np.ravel(np.argwhere(dgm2[:,1] == np.inf))
	dgmf1, dgmi1 = dgm1[f1], dgm1[i1]
	dgmf2, dgmi2 = dgm2[f2], dgm2[i2]
	mtci = np.hstack([i1[np.argsort(dgmi1[:,0])][:,np.newaxis], i2[np.argsort(dgmi2[:,0])][:,np.newaxis]])
	_, mtcff = gd.wasserstein.wasserstein_distance(dgmf1, dgmf2, matching=True, order=1, internal_p=2)
	mtcf = []
	for i in range(len(mtcff)):
		if mtcff[i,0] == -1:	mtcf.append(np.array([[ -1, f2[mtcff[i,1]] ]]))
		elif mtcff[i,1] == -1:	mtcf.append(np.array([[ f1[mtcff[i,0]], -1 ]]))
		else:	mtcf.append(np.array([[ f1[mtcff[i,0]], f2[mtcff[i,1]] ]]))
	mtc = np.vstack([mtci, np.vstack(mtcf)]) if len(mtcf) > 0 else mtci
	return mtc

def sublevelsets_multipersistence(matching, simplextree, filters, homology=0, num_lines=100, corner="dg", extended=False, essential=False, bnds_filt=None, epsilon=1e-10, min_bars=1, noise=0., parallel=True, nproc=4, visu=False, plot_per_bar=False, bnds_visu=None):
	"""
	Code for computing multiparameter sublevel set persistence. 

	Inputs:
		matching: function for computing matchings. Either a Python callable (accepting two diagrams as inputs and returning a partial matching) or the string "vineyards", in which case the vineyards algorithm from Dionysus is used
		simplextree: input simplex tree. Either a path to a simplicial complex file with Dionysus format (https://www.mrzv.org/software/dionysus/examples/pl-vineyard.html), or a simplex tree
		filters: Filtration values. Either a path to a filtration value file, with Dionysus format (https://www.mrzv.org/software/dionysus/examples/pl-vineyard.html), or a Numpy array
		homology: homological dimension
		num_lines: number of lines to use
		corner: which corner do you want to use to compute lines? Either "ll" (lower left), "ur" (upper right) or "dg" (lines with slope 1)
		extended: do you want extended persistence?
		essential: do you want essential features (i.e., with infinite coordinates)?
		bnds_filt: bounding rectangle limits
		epsilon: small shift to avoid horizontal / vertical lines at the beginning of slicing
		min_bars: minimal number of bars to take summand into account
		noise: float specifying the amount of random perturbations of slice endpoints
		parallel: do you want to compute the fibered barcodes and matchings in parallel?
		nproc: number of cores. Used only if parallel is True
		visu: do you want to see the decomposition?
		plot_per_bar: do you want to check each summand individually?
		bnds_visu: bounding rectangle for visualization

	Outputs:
		decomposition: the module decomposition
		lines: the lines used for the decomposition
		the bounding rectangle limits
		the bounding rectangle limits for visualization
	"""

	if type(simplextree) == str:
		if type(matching) == str:	splx = simplextree
		else:
			splx = gd.SimplexTree()
			with open(simplextree, "r") as stfile:
				all_lines = stfile.readlines()
				for line in all_lines:
					lline = line[:-1].split(" ")
					splx.insert([int(v) for v in lline])
			stfile.close()
	elif type(simplextree) == gd.SimplexTree:
		if matching == 'vineyards':
			with open("simplextree", "w") as stfile:
				for (s,_) in simplextree.get_skeleton(0):
					stfile.write(" ".join([str(v) for v in s]) + "\n")
				for (s,_) in simplextree.get_filtration():
					if len(s) > 1:	stfile.write(" ".join([str(v) for v in s]) + "\n")

			stfile.close()
			splx = "simplextree"
		else:	splx = simplextree
	else:
		print("simplextree must be string or gudhi SimplexTree")
		return 0

	if type(matching) != str:	splx_list = [np.vstack([np.array(s)[np.newaxis,:] for s,_ in splx.get_skeleton(h) if len(s) == h+1]) for h in range(splx.dimension()+1)]

	if type(filters) == str:	filts = np.loadtxt(filters)
	elif type(filters) == np.ndarray:	filts = filters
	else:
		print("filters must be string or numpy array")
		return 0


	[n_pts, n_filts] = filts.shape
	F1, F2 = filts[:,0], filts[:,1]

	# if corner == "ur", maxs should be STRICTLY greater than the maximal filtration values
	if bnds_filt is None:	mins, maxs = np.min(filts, axis=0), np.max(filts, axis=0)
	else:	mins, maxs = [bnds_filt[0], bnds_filt[2]], [bnds_filt[1], bnds_filt[3]]
	xmt, xMt, ymt, yMt = mins[0], maxs[0], mins[1], maxs[1]
	
	lines = []

	if corner == "ll":
		midal = np.arctan((yMt-ymt)/(xMt-xmt)) 
		frames = np.concatenate([  np.arctan((np.linspace(start=ymt+epsilon, stop=yMt, num=int(num_lines/2))-ymt)/(xMt-xmt)),
	                                   np.arctan((yMt-ymt)/(np.linspace(start=xMt, stop=xmt+epsilon, num=int(num_lines/2))-xmt))  ])
		for i in range(len(frames)):
			xalpha, yalpha = xmt, ymt
			xAlpha, yAlpha = min(xMt, xmt + (yMt-ymt)/np.tan(frames[i])), min(yMt, ymt + (xMt-xmt)*np.tan(frames[i]))
			lines.append(np.array([[xalpha, yalpha, xAlpha, yAlpha]]))

	if corner == "ur":
		midal = np.arctan((xMt-xmt)/(yMt-ymt))
		frames = np.concatenate([  np.arctan((xMt-np.linspace(start=xMt-epsilon, stop=xmt, num=int(num_lines/2)))/(yMt-ymt)),
	                                   np.arctan((xMt-xmt)/(yMt-np.linspace(start=ymt, stop=yMt-epsilon, num=int(num_lines/2))))  ])
		for i in range(len(frames)):
			xalpha, yalpha = max(xmt, xMt - (yMt-ymt)*np.tan(frames[i])), max(ymt, yMt - (xMt-xmt)/np.tan(frames[i]))
			xAlpha, yAlpha = xMt, yMt
			lines.append(np.array([[xalpha, yalpha, xAlpha, yAlpha]]))

	if corner == "dg":
		delta = ((xMt-xmt) + (yMt-ymt))/num_lines
		xs = np.arange(start=xmt, stop=xMt-epsilon, step=delta)[::-1]
		ys = np.arange(start=ymt+delta, stop=yMt-epsilon, step=delta)
		frames = [("x", x) for x in xs] + [("y", y) for y in ys]
		for i in range(len(frames)):
			xalpha = frames[i][1] if frames[i][0] == "x" else xmt
			yalpha = frames[i][1] if frames[i][0] == "y" else ymt
			xAlpha = xalpha + min(xMt-xalpha, yMt-yalpha)
			yAlpha = yalpha + min(xMt-xalpha, yMt-yalpha)
			lines.append(np.array([[xalpha+np.random.uniform(low=0., high=noise*delta), 
						yalpha+np.random.uniform(low=0., high=noise*delta), 
						xAlpha+np.random.uniform(low=-noise*delta, high=0.), 
						yAlpha+np.random.uniform(low=-noise*delta, high=0.)]]))

	lines = np.vstack(lines)

	NF = []
	for i in range(len(frames)):
		xalpha, yalpha, xAlpha, yAlpha = lines[i,0], lines[i,1], lines[i,2], lines[i,3]
		new_f = []
		for pt in range(n_pts):
			if corner == "ll":	u = max( (F1[pt]-xmt) / np.cos(frames[i]), (F2[pt]-ymt) / np.sin(frames[i]) )
			if corner == "ur":	u = max( max(0,F1[pt]-xalpha)/np.sin(frames[i]), max(0,F2[pt]-yalpha)/np.cos(frames[i]) )
			if corner == "dg":	u = max( max(0,F1[pt]-xalpha)/(0.5*np.sqrt(2)), max(0,F2[pt]-yalpha)/(0.5*np.sqrt(2)) )
			new_f.append(u)
		NF.append([new_f])
	NF = np.vstack(NF)
		

	if matching == "vineyards":

		if extended:
			stbase_ext, stbase = gd.SimplexTree(), gd.SimplexTree()
			with open(splx, "r") as cplxo:	S = cplxo.readlines()
			cplxo.close()
			for line in S:
				s = [int(c) for c in line.split(" ")]
				stbase.insert(s, -1e10)
				stbase_ext.insert(s, -1e10)
			for pt in range(len(filts)):	stbase_ext.assign_filtration([pt], F1[pt])
			stbase_ext.extend_filtration()
			list_splx = [(s,f) for (s,f) in stbase_ext.get_filtration()]
			bary = barycentric_subdivision(stbase_ext, list_splx)

			with open(splx + "_extended.txt", "w") as cplxe:
				for v in range(bary.num_vertices()):	cplxe.write(str(v) + "\n")
				for (s,_) in bary.get_filtration():	
					if len(s) > 1:	cplxe.write(" ".join([str(v) for v in s]) + "\n")
				cplxe.close()
			efd = []

		NNF = []
		for i in range(len(frames)):

			if extended:
				st = gd.SimplexTree()
				for (s,_) in stbase.get_filtration():	st.insert(s, -1e10)
				for pt in range(n_pts):	st.assign_filtration([pt], NF[i,pt])
				st.make_filtration_non_decreasing()
				st.extend_filtration()
				bary = barycentric_subdivision(st, [(s, st.filtration(s)) for (s, _) in list_splx])
				new_f = np.array([bary.filtration([v]) for v in range(bary.num_vertices())])
				efd.append([[min(NF[i,:]), max(NF[i,:])]])
			else:	new_f = NF[i,:]
			NNF.append(new_f[None,:])

		NNF = np.vstack(NNF)

		if extended:	efd = np.vstack(efd)
		#print("Calling Vineyard algo.")
		if extended:
			VS = lsvine(NNF, (splx + "_extended.txt").encode('utf-8'), 1)
		else:
			if essential:	VS = lsvine(NNF, splx.encode('utf-8'), 0)
			else:	VS = lsvine(NNF, splx.encode('utf-8'), 1)
		#print("Received vineyard output.")
		Vs = VS[homology]

		decomposition = []
		for seq in Vs:
			num_bars = int(len(seq)/3)
			if num_bars > min_bars:
				summand = []
				prevnf = -1
				for b in range(num_bars):
					st, ed, nf, nfi = seq[3*b], seq[3*b+1], seq[3*b+2], int(seq[3*b+2])
					if prevnf == nf or nf-nfi > 0:	continue
					else:
						prevnf = nf
						al = frames[nfi]
						xalpha, yalpha, xAlpha, yAlpha = lines[nfi,0], lines[nfi,1], lines[nfi,2], lines[nfi,3]
						if ed == np.inf:	ed = 1e10
						if extended:
							m, M = efd[nfi,0], efd[nfi,1]
							st, ed = -np.abs(st), -np.abs(ed)
							st, ed = m+(st+2)*(M-m), m+(ed+2)*(M-m)
						if corner == "ll":
							summand.append(np.array([[xmt + st*np.cos(al), xmt + ed*np.cos(al), ymt + st*np.sin(al), ymt + ed*np.sin(al), nfi]]))
						if corner == "ur":
							summand.append(np.array([[xalpha + st*np.sin(al), xalpha + ed*np.sin(al), yalpha + st*np.cos(al), yalpha + ed*np.cos(al), nfi]]))
						if corner == "dg":
							summand.append(np.array([[xalpha+st*(0.5*np.sqrt(2)), xalpha+ed*(0.5*np.sqrt(2)), yalpha+st*(0.5*np.sqrt(2)), yalpha+ed*(0.5*np.sqrt(2)), nfi]]))

				summand = np.vstack(summand) if len(summand) > 0 else np.empty([0,5])
				decomposition.append(summand)

	else:

		if parallel:
			ldgms = Parallel(n_jobs=nproc, prefer="threads")(delayed(gudhi_line_diagram)(splx_list, NF[idx,:], homology, extended, essential, "Numpy") for idx in range(len(frames)))
			lmtcs = Parallel(n_jobs=nproc, prefer="threads")(delayed(matching)(ldgms[:-1][idx], ldgms[1:][idx]) for idx in range(len(frames)-1))
		else:
			ldgms = [gudhi_line_diagram(splx, NF[idx,:], homology, extended, essential, "Gudhi") for idx in range(len(frames))]
			lmtcs = [matching(ldgms[:-1][idx], ldgms[1:][idx]) for idx in range(len(frames)-1)]

		raw_decomposition = []
		for idx, mtc in enumerate(lmtcs):
			if idx == 0:
				for pt in range(len(ldgms[idx])):	raw_decomposition.append([(idx, pt)])
			diag_pts_1 = np.ravel(np.argwhere(mtc[:,0] == -1))
			for summand in raw_decomposition:
				summandID = summand[-1][1]
				if summandID != -1:
					pos = list(mtc[:,0]).index(summandID)
					summand.append((idx+1, mtc[pos,1]))
					
			for pt in diag_pts_1:
				raw_decomposition.append([(idx+1, mtc[pt,1])])
		for idx, summand in enumerate(raw_decomposition):
			if summand[-1][1] != -1:	summand.append((idx+1,-1))

		decomposition = []
		for summand_idxs in raw_decomposition:
			num_bars = len(summand_idxs) - 1
			if num_bars > min_bars:
				summand = []
				for b in range(num_bars):
					ptID, frID = summand_idxs[b][1], summand_idxs[b][0]
					st, ed = ldgms[frID][ptID,0], ldgms[frID][ptID,1]
					al = frames[frID]
					xalpha, yalpha, xAlpha, yAlpha = lines[frID][0], lines[frID][1], lines[frID][2], lines[frID][3]
					if ed == np.inf:	ed = 1e10
					if corner == "ll":
						summand.append(np.array([[xmt + st*np.cos(al), xmt + ed*np.cos(al), ymt + st*np.sin(al), ymt + ed*np.sin(al), frID]]))
					if corner == "ur":
						summand.append(np.array([[xalpha + st*np.sin(al), xalpha + ed*np.sin(al), yalpha + st*np.cos(al), yalpha + ed*np.cos(al), frID]]))
					if corner == "dg":
						summand.append(np.array([[xalpha+st*(0.5*np.sqrt(2)), xalpha+ed*(0.5*np.sqrt(2)), yalpha+st*(0.5*np.sqrt(2)), yalpha+ed*(0.5*np.sqrt(2)), frID]]))
				summand = np.vstack(summand) if len(summand) > 0 else np.empty([0,5])
				decomposition.append(summand)

	[xm, xM, ym, yM] = [xmt, xMt, ymt, yMt] if bnds_visu is None else bnds_visu

	if visu:

		cmap = matplotlib.cm.get_cmap("Spectral")

		if not plot_per_bar:
			plt.figure()
			for idx, summand in enumerate(decomposition):
				for i in range(len(summand)):	plt.plot([summand[i,0], summand[i,1]], [summand[i,2], summand[i,3]], c=cmap(idx/len(decomposition)))
			plt.xlim(xm, xM)
			plt.ylim(ym, yM)
			plt.show()

		else:
			for idx, summand in enumerate(decomposition):
				plt.figure()
				for i in range(len(summand)):	plt.plot([summand[i,0], summand[i,1]], [summand[i,2], summand[i,3]], c=cmap(idx/len(decomposition)))
				plt.xlim(xm, xM)
				plt.ylim(ym, yM)
				plt.show()

	return decomposition, lines, [xm, xM, ym, yM], [xmt, xMt, ymt, yMt]

def interlevelsets_multipersistence(matching, simplextree, filters, basepoint=None, homology=0, num_lines=100, essential=False, bnds_filt=None, epsilon=1e-10, min_bars=1, parallel=True, nproc=4, visu=False, plot_per_bar=False, bnds_visu=None):

	"""
	Code for computing multiparameter interlevel set persistence. 

	Inputs:
		matching: function for computing matchings. Either a Python callable (accepting two diagrams as inputs and returning a partial matching) or a path to a vineyards executable
		simplextree: input simplex tree. Either a path to a simplicial complex file with Dionysus format (https://www.mrzv.org/software/dionysus/examples/pl-vineyard.html), or a simplex tree
		filters: Filtration values. Either a path to a filtration value file, with Dionysus format (https://www.mrzv.org/software/dionysus/examples/pl-vineyard.html), or a Numpy array
		basepoint: lower left corner on the diagonal: all lines will go through the point (basepoint, basepoint) 
		homology: homological dimension
		num_lines: number of lines to use
		essential: do you want essential features (i.e., with infinite coordinates)?
		bnds_filt: bounding rectangle limits
		epsilon: small shift to avoid horizontal / vertical lines at the beginning of slicing
		min_bars: minimal number of bars to take summand into account
		parallel: do you want to compute the fibered barcodes and matchings in parallel?
		nproc: number of cores. Used only if parallel is True
		visu: do you want to see the decomposition?
		plot_per_bar: do you want to check each summand individually?
		bnds_visu: bounding rectangle for visualization

	Outputs:
		decomposition: the module decomposition
		lines: the lines used for the decomposition
		the bounding rectangle limits
		the bounding rectangle limits for visualization
	"""
	if type(simplextree) == str:
		if type(matching) == str:	splx = simplextree
		else:
			splx = gd.SimplexTree()
			with open(simplextree, "r") as stfile:
				all_lines = stfile.readlines()
				for line in all_lines:
					lline = line[:-1].split(" ")
					splx.insert([int(v) for v in lline])
			stfile.close()
	elif type(simplextree) == gd.SimplexTree:
		if type(matching) == str:
			with open("simplextree", "w") as stfile:
				for (s,_) in simplextree.get_skeleton(0):
					stfile.write(" ".join([str(v) for v in s]) + "\n")
				for (s,_) in simplextree.get_filtration():
					if len(s) > 1:	stfile.write(" ".join([str(v) for v in s]) + "\n")
			stfile.close()
			splx = "simplextree"
		else:	splx = simplextree
	else:
		print("simplextree must be string or gudhi SimplexTree")
		return 0

	if type(matching) != str:	splx_list = [np.vstack([np.array(s)[np.newaxis,:] for s,_ in splx.get_skeleton(h) if len(s) == h+1]) for h in range(splx.dimension()+1)]

	if type(filters) == str:	filts = np.loadtxt(filters)
	elif type(filters) == np.ndarray:	filts = filters
	else:
		print("filters must be string or numpy array")
		return 0


	n_pts = len(filts)

	[xmt, xMt] = [basepoint, filts.max()] if bnds_filt is None else bnds_filt
	ymt, yMt = xmt, xMt

	midal = np.pi/4
	frames = np.linspace(start=epsilon, stop=np.pi/2-epsilon, num=num_lines)

	lines = []

	for i in range(len(frames)):
		xalpha, yalpha = xmt, ymt
		xAlpha, yAlpha = min(xMt, xmt + (yMt-ymt)/np.tan(frames[i])), min(yMt, ymt + (xMt-xmt)*np.tan(frames[i]))
		lines.append(np.array([[xalpha, yalpha, xAlpha, yAlpha]]))

	lines = np.vstack(lines)

	if basepoint is None:	basepoint = np.mean(filts)
	NF = []
	for i in range(len(frames)):
		new_f = []
		for pt in range(n_pts):
			if filts[pt] >= basepoint:	new_f.append(  (filts[pt] - basepoint) / np.cos(frames[i])  )
			else:	new_f.append(  (basepoint - filts[pt]) / np.sin(frames[i])  )
		NF.append([new_f])
	NF = np.vstack(NF)

	if matching == 'vineyards':

		#vinef = open(splx + "_homotopy.txt", "w")

		NNF = []
		for i in range(len(frames)):
			NNF.append(NF[i,:][None,:])			
		NNF = np.vstack(NNF)
		
		if essential:	VS = lsvine(NNF, splx.encode('utf-8'), 0)
		else:	VS = lsvine(NNF, splx.encode('utf-8'), 1)

		Vs = VS[homology]


		decomposition = []
		for seq in Vs:
			num_bars = int(len(seq)/3)
			if num_bars > min_bars:
				summand = []
				prevnf = -1
				for b in range(num_bars):
					st, ed, nf, nfi = seq[3*b], seq[3*b+1], seq[3*b+2], int(seq[3*b+2])
					if prevnf == nf or nf-nfi > 0:	continue
					else:
						prevnf = nf
						al = frames[nfi]
						if ed == np.inf:	ed = 1e10
						summand.append(np.array([[basepoint + st*np.sin(al), basepoint + ed*np.sin(al), basepoint + st*np.cos(al), basepoint + ed*np.cos(al), nfi]]))

				summand = np.vstack(summand) if len(summand) > 0 else np.empty([0,5])
				decomposition.append(summand)

	else:

		if parallel:
			ldgms = Parallel(n_jobs=nproc, prefer="threads")(delayed(gudhi_line_diagram)(splx_list, NF[idx,:], homology, False, essential, "Numpy") for idx in range(len(frames)))
			lmtcs = Parallel(n_jobs=nproc, prefer="threads")(delayed(matching)(ldgms[:-1][idx], ldgms[1:][idx]) for idx in range(len(frames)-1))
		else:
			ldgms = [gudhi_line_diagram(splx, NF[idx,:], homology, False, essential, "Gudhi") for idx in range(len(frames))]
			lmtcs = [matching(ldgms[:-1][idx], ldgms[1:][idx]) for idx in range(len(frames)-1)]

		raw_decomposition = []
		for idx, mtc in enumerate(lmtcs):
			if idx == 0:
				for pt in range(len(ldgms[idx])):	raw_decomposition.append([(idx, pt)])
			diag_pts_1 = np.ravel(np.argwhere(mtc[:,0] == -1))
			for summand in raw_decomposition:
				summandID = summand[-1][1]
				if summandID != -1:
					pos = np.ravel(np.argwhere(mtc[:,0] == summandID))[0]
					summand.append((idx+1, mtc[pos,1]))
			for pt in diag_pts_1:
				raw_decomposition.append([(idx+1, mtc[pt,1])])
		for idx, summand in enumerate(raw_decomposition):
			if summand[-1][1] != -1:	summand.append((idx+1,-1))

		decomposition = []
		for summand_idxs in raw_decomposition:
			num_bars = len(summand_idxs) - 1
			if num_bars > min_bars:
				summand = []
				for b in range(num_bars):
					ptID, frID = summand_idxs[b][1], summand_idxs[b][0]
					st, ed = ldgms[frID][ptID,0], ldgms[frID][ptID,1]
					al = frames[frID]
					xalpha, yalpha, xAlpha, yAlpha = lines[frID][0], lines[frID][1], lines[frID][2], lines[frID][3]
					if ed == np.inf:	ed = 1e10
					summand.append(np.array([[basepoint + st*np.sin(al), basepoint + ed*np.sin(al), basepoint + st*np.cos(al), basepoint + ed*np.cos(al), frID]]))
				summand = np.vstack(summand) if len(summand) > 0 else np.empty([0,5])
				decomposition.append(summand)


	[xm, xM] = [basepoint, max(filts)] if bnds_visu is None else bnds_visu

	if visu:

		cmap = matplotlib.cm.get_cmap("Spectral")

		if not plot_per_bar:
			plt.figure()
			for idx, summand in enumerate(decomposition):
				for i in range(len(summand)):
					plt.plot([summand[i,0], summand[i,1]], [summand[i,2], summand[i,3]], c=cmap(idx/len(decomposition)))
			plt.xlim(xm, xM)
			plt.ylim(xm, xM)
			plt.show()

		else:
			for idx, summand in enumerate(decomposition):
				plt.figure()
				for i in range(len(summand)):
					plt.plot([summand[i,0], summand[i,1]], [summand[i,2], summand[i,3]], c=cmap(idx/len(decomposition)))
				plt.xlim(xm, xM)
				plt.ylim(xm, xM)
				plt.show()
	return decomposition, lines, [xm, xM], [xmt, xMt]










def intersect_boundaries(summand, bnds, visu=False):

	xm, xM, ym, yM = bnds[0], bnds[1], bnds[2], bnds[3]

	# Select good bars
	good_idxs = np.argwhere(np.abs(summand[:,1]-summand[:,0]) > 0.)[:,0]
	summand = summand[good_idxs]
	good_idxs = np.argwhere(np.abs(summand[:,3]-summand[:,2]) > 0.)[:,0]
	summand = summand[good_idxs]
		
	# Compute intersection with boundaries
	Ts = np.hstack([ np.multiply( xm-summand[:,0:1], 1./(summand[:,1:2]-summand[:,0:1]) ),
			 np.multiply( xM-summand[:,0:1], 1./(summand[:,1:2]-summand[:,0:1]) ),
			 np.multiply( ym-summand[:,2:3], 1./(summand[:,3:4]-summand[:,2:3]) ),
			 np.multiply( yM-summand[:,2:3], 1./(summand[:,3:4]-summand[:,2:3]) ) ])
	Ts = np.hstack([ np.minimum(Ts[:,0:1], Ts[:,1:2]), np.maximum(Ts[:,0:1], Ts[:,1:2]), np.minimum(Ts[:,2:3], Ts[:,3:4]), np.maximum(Ts[:,2:3], Ts[:,3:4]) ])
	Ts = np.hstack([ np.maximum(Ts[:,0:1], Ts[:,2:3]), np.minimum(Ts[:,1:2], Ts[:,3:4]) ])
	good_idxs = np.argwhere(Ts[:,1]-Ts[:,0] > 0.)[:,0]
	summand, Ts = summand[good_idxs], Ts[good_idxs]
	good_idxs = np.argwhere(Ts[:,0] < 1.)[:,0]
	summand, Ts = summand[good_idxs], Ts[good_idxs]
	good_idxs = np.argwhere(Ts[:,1] > 0.)[:,0]
	summand, Ts = summand[good_idxs], Ts[good_idxs]
	Ts = np.hstack([  np.maximum(Ts[:,0:1], np.zeros(Ts[:,0:1].shape)), np.minimum(Ts[:,1:2], np.ones(Ts[:,0:1].shape))  ])
	P1x, P2x, P1y, P2y = summand[:,0:1], summand[:,1:2], summand[:,2:3], summand[:,3:4]
	Ta, Tb = Ts[:,0:1], Ts[:,1:2]
	summand = np.hstack([  np.multiply(1.-Ta, P1x) + np.multiply(Ta, P2x), 
                               np.multiply(1.-Tb, P1x) + np.multiply(Tb, P2x), 
                               np.multiply(1.-Ta, P1y) + np.multiply(Ta, P2y), 
                               np.multiply(1.-Tb, P1y) + np.multiply(Tb, P2y)  ])

	if visu:
		plt.figure()
		for i in range(len(summand)):	plt.plot([summand[i,0], summand[i,1]], [summand[i,2], summand[i,3]], c="red")
		plt.xlim(xm-1., xM+1.)
		plt.ylim(ym-1., yM+1.)
		plt.show()

	return summand










def persistence_image(dgm, bnds, resolution=[100,100], return_raw=False, bandwidth=1., power=1.):
	"""
	Code for computing 1D persistence images.
	"""
	xm, xM, ym, yM = bnds[0], bnds[1], bnds[2], bnds[3]
	x = np.linspace(xm, xM, resolution[0])
	y = np.linspace(ym, yM, resolution[1])
	X, Y = np.meshgrid(x, y)
	Zfinal = np.zeros(X.shape)
	X, Y = X[:,:,np.newaxis], Y[:,:,np.newaxis]

	# Compute image
	P0, P1 = np.reshape(dgm[:,0], [1,1,-1]), np.reshape(dgm[:,1], [1,1,-1])
	weight = np.abs(P1-P0)
	distpts = np.sqrt((X-P0)**2+(Y-P1)**2)

	if return_raw:
		lw = [weight[0,0,pt] for pt in range(weight.shape[2])]
		lsum = [distpts[:,:,pt] for pt in range(distpts.shape[2])]
	else:
		weight = weight**power
		Zfinal = (np.multiply(weight, np.exp(-distpts**2/bandwidth))).sum(axis=2)

	output = [lw, lsum] if return_raw else Zfinal
	return output

class PersistenceImageWrapper(BaseEstimator, TransformerMixin):
	"""
	Scikit-Learn wrapper for cross-validating 1D persistence images.
	"""
	def __init__(self, bdw=1., power=0, step=1):
		self.bdw, self.power, self.step = bdw, power, step

	def fit(self, X, y=None):
		return self

	def transform(self, X):
		final = []
		bdw = self.bdw
		for nf in range(len(X[0])):
			XX = [X[idx][nf] for idx in range(len(X))]
			lpi = []
			for idx, _ in enumerate(XX):
				im = sum([(XX[idx][0][i]**self.power)*np.exp(-XX[idx][1][i]**2/bdw) for i in range(len(XX[idx][0]))]) if len(XX[idx][0]) > 0 else np.zeros([50,50])
				im = np.reshape(im, [1,-1])
				lpi.append(im)
			Y = np.vstack(lpi)
			res = int(np.sqrt(Y.shape[1]))
			nr = int(res/self.step)
			Y = np.reshape(np.transpose(np.reshape(np.transpose(np.reshape(Y,[-1,res,nr,self.step]).sum(axis=3),(0,2,1)),[-1,nr,nr,self.step]).sum(axis=3),(0,2,1)),[-1,nr**2])
			final.append(Y)

		return np.hstack(final)










def multipersistence_image(decomposition, bnds=None, resolution=[100,100], return_raw=False, bandwidth=1., power=1., line_weight=lambda x: 1):
	"""
	Code for computing Multiparameter Persistence Images.

	Inputs:
		decomposition: vineyard decomposition as provided with interlevelset_multipersistence or sublevelset_multipersistence
		bnds: bounding rectangle
		resolution: number of pixels for each image axis
		return_raw: whether to return the raw images and weights for each summand (useful to save time when cross validating the image parameters) or the usual image
		bandwidth: image bandwidth
		power: exponent for summand weight
		line_weight: weight function for the lines in the vineyard decomposition

	Outputs:
		image (as a numpy array) if return_raw is False otherwise list of images and weights for each summand 
	"""
	if np.all(~np.isnan(np.array(bnds))):
		bnds = bnds
	else:
		full = np.vstack(decomposition)
		maxs, mins = full.max(axis=0), full.min(axis=0)
		bnds = list(np.where(np.isnan(np.array(bnds)), np.array([min(mins[0],mins[1]), max(maxs[0],maxs[1]), min(mins[2],mins[3]), max(maxs[2],maxs[3])]), np.array(bnds)))

	xm, xM, ym, yM = bnds[0], bnds[1], bnds[2], bnds[3]
	x = np.linspace(xm, xM, resolution[0])
	y = np.linspace(ym, yM, resolution[1])
	X, Y = np.meshgrid(x, y)
	Zfinal = np.zeros(X.shape)
	X, Y = X[:,:,np.newaxis], Y[:,:,np.newaxis]

	if return_raw:	lw, lsum = [], []

	for summand in decomposition:

		summand = intersect_boundaries(summand, bnds)

		# Compute weight
		if return_raw or power > 0.:
			bars   = np.linalg.norm(summand[:,[0,2]]  -summand[:,[1,3]],  axis=1)		
			consm  = np.linalg.norm(summand[:-1,[0,2]]-summand[1:,[0,2]], axis=1)
			consM  = np.linalg.norm(summand[:-1,[1,3]]-summand[1:,[1,3]], axis=1)
			diags  = np.linalg.norm(summand[:-1,[0,2]]-summand[1:,[1,3]], axis=1)
			s1, s2 = .5 * (bars[:-1] + diags + consM), .5 * (bars[1:] + diags + consm)
			weight = np.sum(np.sqrt(np.abs(np.multiply(np.multiply(np.multiply(s1,s1-bars[:-1]),s1-diags),s1-consM)))+np.sqrt(np.abs(np.multiply(np.multiply(np.multiply(s2,s2-bars[1:]),s2-diags),s2-consm))))
			weight /= ((xM-xm)*(yM-ym))	
		else:	weight = 1.

		# Compute image
		P00, P01, P10, P11 = np.reshape(summand[:,0], [1,1,-1]), np.reshape(summand[:,2], [1,1,-1]), np.reshape(summand[:,1], [1,1,-1]), np.reshape(summand[:,3], [1,1,-1])
		good_xidxs, good_yidxs = np.argwhere(P00 != P10), np.argwhere(P01 != P11)
		good_idxs = np.unique(np.reshape(np.vstack([good_xidxs[:,2:3], good_yidxs[:,2:3]]), [-1]))
		if len(good_idxs) > 0:
			P00, P01, P10, P11 = P00[:,:,good_idxs], P01[:,:,good_idxs], P10[:,:,good_idxs], P11[:,:,good_idxs]
			vectors = [ P10[0,0,:]-P00[0,0,:], P11[0,0,:]-P01[0,0,:] ]
			vectors = np.hstack([v[:,np.newaxis] for v in vectors])
			norm_vectors = np.linalg.norm(vectors, axis=1)
			unit_vectors = np.multiply( vectors, 1./norm_vectors[:,np.newaxis] )
			W = np.array([line_weight(unit_vectors[i,:]) for i in range(len(unit_vectors))])
			T = np.maximum(np.minimum(np.multiply(np.multiply(P00-X,P00-P10)+np.multiply(P01-Y,P01-P11),1./np.square(np.reshape(norm_vectors,[1,1,-1])) ),1),0)
			distlines = np.sqrt((X-P00+np.multiply(T,P00-P10))**2+(Y-P01+np.multiply(T,P01-P11))**2 )
			Zsummand = distlines.min(axis=2)
			arglines = np.argmin(distlines, axis=2)
			weightlines = W[arglines]
			#Zsummand = np.multiply(Zsummand, weightlines)
			if return_raw:
				lw.append(weight)
				lsum.append(Zsummand)
			else:
				weight = weight**power
				Zfinal += weight * np.multiply(np.exp(-Zsummand**2/bandwidth), weightlines)

	output = [lw, lsum] if return_raw else Zfinal
	return output


def convert_summand(summand):
	num_lines = len(summand)
	dimension = len(summand[0][0])
	#assert dimension == 2
	new_summand = np.zeros((num_lines,4))
	for i in range(num_lines):
		if summand[i][0] == []:	continue
		new_summand[i,0] = summand[i][0][0]
		new_summand[i,1] = summand[i][1][0]
		new_summand[i,2] = summand[i][0][1]
		new_summand[i,3] = summand[i][1][1]
	return new_summand

def convert_barcodes(barcodes):
	new_barcodes = []
	for matching in barcodes:
		new_barcodes.append(convert_summand(matching))
	return new_barcodes





class MultiPersistenceImageWrapper(BaseEstimator, TransformerMixin):
	"""
	Scikit-Learn wrapper for cross-validating Multiparameter Persistence Images.
	"""
	def __init__(self, bdw=1., power=0, step=1):
		self.bdw, self.power, self.step = bdw, power, step

	def fit(self, X, y=None):
		return self

	def transform(self, X):
		final = []
		bdw = self.bdw
		for nf in range(len(X[0])):
			XX = [X[idx][nf] for idx in range(len(X))]
			lmpi = []
			for idx, _ in enumerate(XX):
				im = sum([(XX[idx][0][i]**self.power)*np.exp(-XX[idx][1][i]**2/bdw) for i in range(len(XX[idx][0]))]) if len(XX[idx][0]) > 0 else np.zeros([50,50])
				im = np.reshape(im, [1,-1])
				lmpi.append(im)
			Y = np.vstack(lmpi)
			res = int(np.sqrt(Y.shape[1]))
			nr = int(res/self.step)	
			Y = np.reshape(np.transpose(np.reshape(np.transpose(np.reshape(Y,[-1,res,nr,self.step]).sum(axis=3),(0,2,1)),[-1,nr,nr,self.step]).sum(axis=3),(0,2,1)),[-1,nr**2])
			final.append(Y)

		return np.hstack(final)










def multipersistence_landscape(decomposition, bnds, delta, resolution=[100,100], k=None, return_raw=False, power=1.):
	"""
	Code for computing Multiparameter Persistence Landscapes.

	Inputs:
		decomposition: decomposition as provided with sublevelset_multipersistence with corner=="dg"
		bnds: bounding rectangle
		delta: distance between consecutive lines in the vineyard decomposition. It can be computed from the second output of sublevelset_multipersistence
		resolution: number of pixels for each landscape axis
		k: number of landscapes
		return_raw: whether to return the raw landscapes and weights for each summand (useful to save time when cross validating the landscape parameters) or the usual landscape
		power: exponent for summand weight (useful if silhouettes are computed)

	Outputs:
		landscape (as a numpy array) if return_raw is False otherwise list of landscapes and weights for each summand 
	"""
	if np.all(~np.isnan(np.array(bnds))):
		bnds = bnds
	else:
		full = np.vstack(decomposition)
		maxs, mins = full.max(axis=0), full.min(axis=0)
		bnds = list(np.where(np.isnan(np.array(bnds)), np.array([min(mins[0],mins[1]), max(maxs[0],maxs[1]), min(mins[2],mins[3]), max(maxs[2],maxs[3])]), np.array(bnds)))

	xm, xM, ym, yM = bnds[0], bnds[1], bnds[2], bnds[3]
	x = np.linspace(xm, xM, resolution[0])
	y = np.linspace(ym, yM, resolution[1])
	X, Y = np.meshgrid(x, y)
	X, Y = X[:,:,np.newaxis], Y[:,:,np.newaxis]
	mesh = np.reshape(np.concatenate([X,Y], axis=2), [-1,2])

	final = []

	if len(decomposition) > 0:

		agl = np.sort(np.unique(np.concatenate([summand[:,4] for summand in decomposition])))

		for al in agl:

			tris = np.vstack( [ summand[np.argwhere(summand[:,4] == int(al))[:,0]] for summand in decomposition ] )

			if len(tris) > 0:

				tris = intersect_boundaries(tris, bnds)
				P1x, P2x, P1y, P2y = tris[:,0:1], tris[:,1:2], tris[:,2:3], tris[:,3:4]
				bars = np.linalg.norm(np.hstack([P2x-P1x, P2y-P1y]), axis=1)
				good_idxs = np.argwhere(bars > 0)[:,0]
				P1x, P2x, P1y, P2y, bars = P1x[good_idxs], P2x[good_idxs], P1y[good_idxs], P2y[good_idxs], np.reshape(bars[good_idxs], [1,-1])
				e1s, e2s = np.array([[ -delta/2, delta/2 ]]), np.hstack([ P2x-P1x, P2y-P1y ])
				e1s = np.reshape(np.multiply(e1s, 1./np.linalg.norm(e1s, axis=1)[:,np.newaxis]).T, [1,2,-1])
				e2s = np.reshape(np.multiply(e2s, 1./np.linalg.norm(e2s, axis=1)[:,np.newaxis]).T, [1,2,-1])
				pts = mesh[:,:,np.newaxis] - np.reshape(np.hstack([P1x+delta/4, P1y-delta/4]).T, [1,2,-1])
				scal1, scal2 = np.multiply(pts, e1s).sum(axis=1), np.multiply(pts, e2s).sum(axis=1)
				output = np.where( (scal1 >= 0) & (scal1 < np.sqrt(2)*delta/2) & (scal2 >= 0) & (scal2 <= bars), np.minimum(scal2, bars-scal2), np.zeros(scal2.shape))
				LS = np.reshape(output, [X.shape[0], X.shape[1], len(P1x)])

				if k is None:
					if return_raw:	final.append([LS, bars])
					else:	final.append( np.multiply(LS, np.reshape(bars**power, [1,1,-1])).sum(axis=2) )
				else:
					pLS = np.concatenate([np.zeros([LS.shape[0], LS.shape[1], 1]), LS], axis=2)
					num = LS.shape[2]
					final.append(np.concatenate([np.partition(pLS, kth=max(num-(kk-1),0), axis=2)[:,:,max(num-(kk-1),0):max(num-(kk-1),0)+1] for kk in range(1,k+1) ], axis=2))

		if k is None:
			if return_raw:	return final
			else:	return np.maximum.reduce(final)
		else:	return  np.maximum.reduce(final)

	else:
		if k is None:
			if return_raw:	return [  [np.zeros([X.shape[0], X.shape[1], 1]), np.zeros([1])]  ]
			else:	np.zeros([X.shape[0], X.shape[1]])
		else:	return np.zeros([X.shape[0], X.shape[1], 1])

class MultiPersistenceLandscapeWrapper(BaseEstimator, TransformerMixin):
	"""
	Scikit-Learn wrapper for cross-validating Multiparameter Persistence Landscapes.
	"""
	def __init__(self, power=0, step=1, k=None):
		self.power, self.step, self.k = power, step, k

	def fit(self, X, y=None):
		return self

	def transform(self, X):
		final = []
		for nf in range(len(X[0])):
			XX = [X[idx][nf] for idx in range(len(X))]
			if self.k is None:
				Y = np.vstack([  np.maximum.reduce([np.multiply(im, np.reshape(w**self.power, [1,1,-1])).sum(axis=2).flatten()[np.newaxis,:] for [im,w] in L])  for L in XX  ])
			else:
				Y = np.vstack([  LS[:,:,:self.k].sum(axis=2).flatten()[np.newaxis,:] for LS in XX  ])
			res = int(np.sqrt(Y.shape[1]))
			nr = int(res/self.step)	
			Y = np.reshape(np.transpose(np.reshape(np.transpose(np.reshape(Y,[-1,res,nr,self.step]).sum(axis=3),(0,2,1)),[-1,nr,nr,self.step]).sum(axis=3),(0,2,1)),[-1,nr**2])
			final.append(Y)
		return np.hstack(final)










def extract_diagrams(decomposition, bnds, lines):
	"""
	Code for extracting all persistence diagrams from a decomposition.

	Inputs:
		decomposition: decomposition as provided with interlevelset_multipersistence or sublevelset_multipersistence
		bnds: bounding rectangle
		lines: lines used for computing decompositions

	Outputs:
		ldgms: list of persistence diagrams		
	"""
	if len(decomposition) > 0:

		mdgm = np.vstack(decomposition)
		agl = np.arange(len(lines))
		ldgms, limits = [], []

		for al in agl:

			dg = []
			idxs = np.argwhere(mdgm[:,4] == al)[:,0]
			if len(idxs) > 0:	dg.append(mdgm[idxs][:,:4])
			if len(dg) > 0:
				dg = np.vstack(dg)
				dg = intersect_boundaries(dg, bnds)
				if len(dg) > 0:
					xalpha, yalpha, xAlpha, yAlpha = lines[al,0], lines[al,1], lines[al,2], lines[al,3]
					pt = np.array([[xalpha, yalpha]])
					st, ed = dg[:,[0,2]], dg[:,[1,3]]
					dgm = np.hstack([ np.linalg.norm(st-pt, axis=1)[:,np.newaxis], np.linalg.norm(ed-pt, axis=1)[:,np.newaxis] ])
				else:	dgm = np.array([[.5*(bnds[0]+bnds[1]), .5*(bnds[2]+bnds[3])]])
			else:	dgm = np.array([[.5*(bnds[0]+bnds[1]), .5*(bnds[2]+bnds[3])]])
	
			ldgms.append(dgm)
	else:	ldgms = [np.array([[.5*(bnds[0]+bnds[1]), .5*(bnds[2]+bnds[3])]]) for _ in range(len(lines))]

	return ldgms

def multipersistence_kernel(X, Y, lines, kernel, line_weight=lambda x: 1, same=False, metric=True, return_raw=False, power=1.):
	"""
	Code for computing Multiparameter Persistence Kernel.

	Inputs:
		X: first list of persistence diagrams extracted from decompositions as provided with extract_diagrams
		Y: second list of persistence diagrams extracted from decompositions as provided with extract_diagrams
		lines: lines used for computing decompositions, as provided with interlevelset_multipersistence or sublevelset_multipersistence
		kernel: kernel function between persistence diagrams if metric == True otherwise CNSD distance function between persistence diagrams 
		line_weight: weight function for the lines in the decomposition
		same: are X and Y the same list?
		metric: do you want to use CNSD distances or direct kernels?
		return_raw: whether to return the raw kernel matrices and weights for each line (useful to save time when cross validating the multiparam. kernel) or the usual kernel matrix
		power: exponent for line weight

	Outputs:
		kernel matrix (as a numpy array) if return_raw is False otherwise list of matrices and weights for each line 
	"""
	M = np.zeros([len(X), len(Y), len(lines)])
	vectors = np.hstack([ lines[:,2:3]-lines[:,0:1], lines[:,3:4]-lines[:,1:2]])
	unit_vectors = np.multiply(vectors, 1./np.linalg.norm(vectors, axis=1)[:,np.newaxis])
	W = np.zeros([len(lines)])

	for l in range(len(lines)):
		W[l] = line_weight(unit_vectors[l,:])
		if same:
			for i in range(len(X)):
				ldgmsi = X[i]
				for j in range(i, len(X)):
					#print(i,j)
					ldgmsj = X[j]
					M[i,j,l] = kernel(ldgmsi[l], ldgmsj[l])
					M[j,i,l] = M[i,j,l]
		else:
			for i in range(len(X)):
				ldgmsi = X[i]
				for j in range(len(Y)):
					ldgmsj = Y[j]
					M[i,j,l] = kernel(ldgmsi[l], ldgmsj[l])
	if metric:
		med = 1 if np.median(M) == 0 else np.median(M)
	if not return_raw:
		if metric:	return np.multiply( W[np.newaxis, np.newaxis, :]**power, np.exp(-M/med) ).sum(axis=2)
		else:	return np.multiply( W[np.newaxis, np.newaxis, :]**power, M).sum(axis=2)
	else:	
		if metric:	return np.exp(-M/med), W
		else:	return M, W


class SubsampleWrapper(BaseEstimator, TransformerMixin):
	"""
	Scikit-Learn wrapper for cross-validating resolutions.
	"""
	def __init__(self, step=1):
		self.step = step

	def fit(self, X, y=None):
		return self

	def transform(self, X):
		return X[:,::self.step]
