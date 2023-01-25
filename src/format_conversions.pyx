cdef extern from "mma_cpp/format_python-cpp.h":
	#list_simplicies_to_sparse_boundary_matrix
	vector[vector[unsigned int]] build_sparse_boundary_matrix_from_simplex_list(vector[vector[unsigned int]] list_simplices)
	#list_simplices_ls_filtration_to_sparse_boundary_filtration
	pair[vector[vector[unsigned int]], vector[vector[double]]] build_boundary_matrix_from_simplex_list(vector[vector[unsigned int]] list_simplices, vector[vector[double]] filtrations, vector[unsigned int] filters_to_permute)
	# pair[vector[vector[unsigned int]], vector[double]] simplextree_to_boundary_filtration(vector[boundary_type] &simplexList, filtration_type &filtration)
	pair[boundary_matrix, multifiltration] simplextree_to_boundary_filtration(uintptr_t)

	pair[vector[vector[unsigned int]], vector[double]] __old__simplextree_to_boundary_filtration(vector[boundary_type]&, filtration_type&)

	string __old__simplextree2rivet(const uintptr_t, const vector[filtration_type]&)
	void simplextree2rivet(const string&, const uintptr_t, const vector[filtration_type]&)



def splx2bf_old(simplextree:GudhiSimplexTree):
	boundaries = [s for s,f in simplextree.get_simplices()]
	filtration = [f for s,f in simplextree.get_simplices()]
	return __old__simplextree_to_boundary_filtration(boundaries,filtration)
	

def splx2bf(simplextree:SimplexTree | GudhiSimplexTree):
	"""Computes a (sparse) boundary matrix, with associated filtration. Can be used as an input of approx afterwards.
	
	Parameters
	----------
	simplextree: Gudhi or mma simplextree
		The simplextree defining the filtration to convert to boundary-filtration.
	
	Returns
	-------
	B:List of lists of ints
		The boundary matrix.
	F: List of 1D filtration
		The filtrations aligned with B; the i-th simplex of this simplextree has boundary B[i] and filtration(s) F[i].
	
	"""
	if type(simplextree) == type(SimplexTree()):
		return simplextree_to_boundary_filtration(simplextree.thisptr)
	else:
		temp_st = from_gudhi(simplextree, parameters=1)
		b,f=simplextree_to_boundary_filtration(temp_st.thisptr)
		temp_st
		return b, f[0]


def simplextree_to_sparse_boundary(st:GudhiSimplexTree):
	return build_sparse_boundary_matrix_from_simplex_list([simplex[0] for simplex in st.get_simplices()])





def __old__convert_to_rivet(simplextree:GudhiSimplexTree, kde, X,*, dimension=1, verbose = True)->None:
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

def __old__splx2rivet(simplextree:GudhiSimplexTree, F, **kwargs):
	"""Converts an input of approx to a file (rivet_dataset.txt) that can be opened by rivet.

	Parameters
	----------
	simplextree: gudhi.SimplexTree
		A gudhi simplextree defining the chain complex
	bifiltration: pair of filtrations. Same format as the approx function.
		list of 1-dimensional filtrations that encode the multiparameter filtration.
		Given an index i, filters[i] should be the list of filtration values of
		the simplices, in lexical order, of the i-th filtration.

	Returns
	-------
	Nothing.
	The file created is located at <current_working_directory>/rivet_dataset.txt; and can be directly imported to rivet.
	"""
	if(type(F) == np.ndarray):
		#assert filters.shape[1] == 2
		G = [F[:,i] for i in range(F.shape[1])]
	else:
		G = F
	if exists("rivet_dataset.txt"):
		remove("rivet_dataset.txt")
	file = open("rivet_dataset.txt", "a")
	file.write("--datatype bifiltration\n")
	file.write("--xbins " + str(kwargs.get("xbins", 0))+"\n")
	file.write("--ybins " + str(kwargs.get("xbins", 0))+"\n")
	file.write("--xlabel "+ kwargs.get("xlabel", "")+"\n")
	file.write("--ylabel " + kwargs.get("ylabel", "") + "\n\n")

	simplices = __old__simplextree2rivet(simplextree.thisptr, G).decode("UTF-8")
	file.write(simplices)
	#return simplices

def splx2rivet(simplextree:GudhiSimplexTree, F, **kwargs):
	"""Converts an input of approx to a file (rivet_dataset.txt) that can be opened by rivet.

	Parameters
	----------
	simplextree: gudhi.SimplexTree
		A gudhi simplextree defining the chain complex
	bifiltration: pair of filtrations. Same format as the approx function.
		list of 1-dimensional filtrations that encode the multiparameter filtration.
		Given an index i, filters[i] should be the list of filtration values of
		the simplices, in lexical order, of the i-th filtration.

	Returns
	-------
	Nothing.
	The file created is located at <current_working_directory>/rivet_dataset.txt; and can be directly imported to rivet.
	"""
	from os import getcwd
	if(type(F) == np.ndarray):
		#assert filters.shape[1] == 2
		G = [F[:,i] for i in range(F.shape[1])]
	else:
		G = F
	path = getcwd()+"/rivet_dataset.txt"
	if exists(path):
		remove(path)
	file = open(path, "a")
	file.write("--datatype bifiltration\n")
	file.write("--xbins " + str(kwargs.get("xbins", 0))+"\n")
	file.write("--ybins " + str(kwargs.get("xbins", 0))+"\n")
	file.write("--xlabel "+ kwargs.get("xlabel", "")+"\n")
	file.write("--ylabel " + kwargs.get("ylabel", "") + "\n\n")
	file.close()

	simplextree2rivet(path.encode("UTF-8"), simplextree.thisptr, G)
	return
