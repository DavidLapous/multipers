import numpy as np
from os.path import expanduser
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from os.path import expanduser
from os import listdir
import os
import MDAnalysis as mda
import matplotlib.pyplot as plt
from MDAnalysis.topology.guessers import guess_masses
import multipers as mp
# from numba import njit
from tqdm import tqdm
from typing import Iterable
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator,TransformerMixin


DATASET_PATH = expanduser("~/Datasets/")
JC_path = DATASET_PATH + "Cleves-Jain/"
DUDE_path = DATASET_PATH + "DUD-E/"


#pathes = get_data_path()
#imgs = apply_pipeline(pathes=pathes, pipeline=pipeline_img)
#distances_to_letter, ytest = img_distances(imgs)


def _get_mols_in_path(folder):
	with open(folder+"/TargetList", "r") as f:
		train_data =  [folder + "/" + mol.strip() for mol in f.readlines()]
	criterion = lambda dataset : dataset.endswith(".mol2") and not dataset.startswith("final") and dataset not in train_data
	test_data = [folder + "/" + dataset for dataset in listdir(folder) if criterion(folder + "/" + dataset)]
	return train_data, test_data
def get_data_path_JC(type="dict"):
	if type == "dict": out = {}
	elif type == "list": out = []
	else: raise TypeError(f"Type {out} not supported")
	for stuff in listdir(JC_path):
		if stuff.startswith("target_"):
			current_letter = stuff[-1]
			to_add = _get_mols_in_path(JC_path + stuff)
			if type == "dict":	out[current_letter] = to_add
			elif type == "list": out.append(to_add)
	decoy_folder = JC_path + "RognanRing850/"
	to_add = [decoy_folder + mol for mol in listdir(decoy_folder) if mol.endswith(".mol2")]
	if type == "dict":	out["decoy"] = to_add
	elif type == "list": out.append(to_add)
	return out
def get_all_JC_path():
	out = []
	for stuff in listdir(JC_path):
		if stuff.startswith("target_"):
			train_data, test_data =  _get_mols_in_path(JC_path + stuff)
			out += train_data
			out += test_data
	decoy_folder = JC_path + "RognanRing850/"
	out +=[decoy_folder + mol for mol in listdir(decoy_folder) if mol.endswith(".mol2")]
	return out
		

def split_multimol(path:str, mol_name:str, out_folder_name:str = "splitted", enforce_charges:bool=False):
	with open(path + mol_name, "r") as f:
		lines = f.readlines()
	splitted_mols = []
	index = 0
	for i,line in enumerate(lines):
		is_last = i == len(lines)-1
		if line.strip() == "@<TRIPOS>MOLECULE" or is_last:
			if i != index:
				molecule = "".join(lines[index:i + is_last])
				if enforce_charges:
					# print(f"Replaced molecule {i}")
					molecule = molecule.replace("NO_CHARGES","USER_CHARGES")
					# print(molecule)
					# return
				index = i
				splitted_mols.append(molecule)
	if not os.path.exists(path + out_folder_name):
		os.mkdir(path + out_folder_name)
	for i,mol in enumerate(splitted_mols):
		with open(path + out_folder_name + f"/{i}.mol2", "w") as f:
			f.write(mol)
	return [path+out_folder_name + f"/{i}.mol2" for i in range(len(splitted_mols))]

# @njit(parallel=True)
def apply_pipeline(pathes:dict, pipeline):
	img_dict = {}
	for key, value in tqdm(pathes.items(), desc="Applying pipeline"):
		if len(key) == 1:
			train_paths, test_paths = value
			train_imgs = pipeline.transform(train_paths)
			test_imgs = pipeline.transform(test_paths)
			img_dict[key] = (train_imgs, test_imgs)
		else:
			assert key == "decoy"
			img_dict[key] = pipeline.transform(value)
	return img_dict

from sklearn.metrics import pairwise_distances
def img_distances(img_dict:dict):
	distances_to_anchors = []
	ytest = []
	decoy_list = img_dict["decoy"]
	for letter, imgs in img_dict.items():
		if len(letter) != 1 : continue # decoy
		xtrain, xtest = imgs
		assert len(xtest)>0
		train_data, test_data = xtrain, np.concatenate([xtest ,decoy_list])
		D = pairwise_distances(train_data, test_data)
		distances_to_anchors.append(D)
		letter_ytest = np.array([letter]*len(xtest) + ['0']*len(decoy_list), dtype="<U1")
		ytest.append(letter_ytest)
	return distances_to_anchors, ytest
	
def get_EF_vector_from_distances(distances, ytest, alpha=0.05):
	EF = []
	for distance_to_anchors, letter_ytest in zip(distances, ytest):
		indices = np.argsort(distance_to_anchors, axis=1)
		n = indices.shape[1]
		n_max = int(alpha*n)
		good_indices = (letter_ytest[indices[:,:n_max]] == letter_ytest[0]) ## assumes that ytest[:,0] are the good letters
		EF_letter = good_indices.sum(axis=1) / (letter_ytest == letter_ytest[0]).sum()
		EF_letter /= alpha
		EF.append(EF_letter.mean())
	return np.mean(EF)

def EF_from_distance_matrix(distances:np.ndarray, labels:list|np.ndarray, alpha:float, anchors_in_test=True):
	"""
	Computes the Enrichment Factor from a distance matrix, and its labels.
	 - First axis of the distance matrix is the anchors on which to compute the EF
	 - Second axis is the test. For convenience, anchors can be put in test, if the flag anchors_in_test is set to true.
	 - labels is a table of bools, representing the the labels of the test axis of the distance matrix.
	 - alpha : the EF alpha parameter.
	"""
	n = len(labels)
	n_max = int(alpha*n)
	indices = np.argsort(distances, axis=1)
	EF_ = [((labels[idx[:n_max]]).sum()-anchors_in_test)/(labels.sum()-anchors_in_test) for idx in indices]
	return np.mean(EF_)/alpha

def EF_AUC(distances:np.ndarray, labels:np.ndarray, anchors_in_test=0):
	if distances.ndim == 1:
		distances = distances[None,:]
	assert distances.ndim == 2
	indices = np.argsort(distances, axis=1)
	out = []
	for i in range(1,distances.size):
		proportion_of_good_indices = (labels[indices[:,:i]].sum(axis=1).mean() -anchors_in_test)/min(i,labels.sum() -anchors_in_test)
		out.append(proportion_of_good_indices)
	# print(out)
	return np.mean(out)


def theorical_max_EF(distances,labels, alpha):
	n = len(labels)
	n_max = int(alpha*n)
	num_true_labels = np.sum(labels == labels[0]) ## if labels are not True / False, assumes that the first one is a good one
	return min(n_max, num_true_labels)/alpha


def theorical_max_EF_from_distances(list_of_distances,list_of_labels, alpha):
	return np.mean([theorical_max_EF(distances, labels,alpha) for distances, labels in zip(list_of_distances, list_of_labels)])

def plot_EF_from_distances(alphas = [0.01, 0.02, 0.05, 0.1], EF = EF_from_distance_matrix, plot:bool=True):
	y = np.round([EF(alpha=alpha) for alpha in alphas], decimals=2)
	if plot:
		_alphas = np.linspace(0.01, 1., 100)
		plt.figure()
		plt.plot(_alphas, [EF(alpha=alpha) for alpha in _alphas])
		plt.scatter(alphas, y, c='r')
		plt.title("Enrichment Factor")
		plt.xlabel(r"$\alpha$" + f" = {alphas}")
		plt.ylabel(r"$\mathrm{EF}_\alpha$" + f" = {y}")
	return y


def lines2bonds(mol:mda.Universe, bond_types = ['ar','am',3,2,1,0], molecule_format=None):
	extension = mol.filename.split('.')[-1].lower() if molecule_format is None else molecule_format
	match extension:
		case 'mol2':
			out = lines2bonds_MOL2(mol)['bond_type']
		case 'pdb':
			out = lines2bonds_PDB(mol)
		case _:
			raise Exception('Invalid, or not supported molecule format.')
	return LabelEncoder().fit(bond_types).transform(out)
	

def lines2bonds_MOL2(mol:mda.Universe):
	_lines = open(mol.filename, "r").readlines()
	out = []
	index = 0
	while index < len(_lines) and  _lines[index].strip() != "@<TRIPOS>BOND":
		index += 1
	index += 1
	while index < len(_lines) and  _lines[index].strip()[0] != "@":
		line = _lines[index].strip().split(" ")
		for j,truc in enumerate(line):
			line[j] = truc.strip()
		# try:
		out.append([stuff for stuff in line if len(stuff) > 0])
		# except:
		# 	print_lin
		index +=1
	out = pd.DataFrame(out, columns=["bond_id","atom1", "atom2", "bond_type"])
	out.set_index(["bond_id"],inplace=True)
	return  out


def lines2bonds_PDB(mol:mda.Universe):
	raise Exception('Not yet implemented.')
	return  

def _mol2graphst(path:str|mda.Universe, filtrations:Iterable[str], molecule_format=None):
	molecule = path if isinstance(path, mda.Universe) else mda.Universe(path) 

	num_filtrations = len(filtrations)
	nodes = molecule.atoms.indices.reshape(1,-1)
	edges = molecule.bonds.dump_contents().T
	num_vertices = nodes.shape[1]
	num_edges = edges.shape[1]
	
	st = mp.SimplexTreeMulti(num_parameters = num_filtrations)
	
	## Edges filtration
	# edges = np.array(bonds_df[["atom1", "atom2"]]).T
	edges_filtration = np.zeros((num_edges, num_filtrations), dtype=np.float32) - np.inf
	for i, filtration in enumerate(filtrations):
		match filtration:
			case "bond_length":
				bond_lengths = molecule.bonds.bonds()
				edges_filtration[:,i] = bond_lengths
			case "bond_type":
				bond_types = lines2bonds(mol=molecule, molecule_format=molecule_format)
				edges_filtration[:,i] = bond_types
			case _:
				pass
	
	## Nodes filtration
	nodes_filtrations = np.zeros((num_vertices,num_filtrations), dtype=np.float32) + np.min(edges_filtration, axis=0) # better than - np.inf
	st.insert_batch(nodes, nodes_filtrations)

	st.insert_batch(edges, edges_filtration)
	for i, filtration in enumerate(filtrations):
		match filtration:
			case "charge":
				charges = molecule.atoms.charges
				st.fill_lowerstar(charges, parameter=i)
			case "atomic_mass":
				masses = molecule.atoms.masses
				null_indices = masses == 0
				if np.any(null_indices): # guess if necessary
					masses[null_indices] = guess_masses(molecule.atoms.types)[null_indices]
				st.fill_lowerstar(-masses, parameter=i)
			case _:
				pass
	st.make_filtration_non_decreasing() # Necessary ?
	return st


def _mol2ripsst(path:str, filtrations:Iterable[str], threshold=np.inf, bond_types:list=['ar','am',3,2,1,0]):
	import gudhi as gd
	assert 'bond_length' == filtrations[0], "Bond length has to be first for rips."
	molecule = path if isinstance(path, mda.Universe) else mda.Universe(path) 
	num_parameters = len(filtrations)
	st_rips = gd.RipsComplex(points = molecule.atoms.positions, max_edge_length=threshold).create_simplex_tree()
	st = mp.SimplexTreeMulti(st_rips, num_parameters=num_parameters, 
			  default_values = [bond_types.index(0) if f == "bond_type" else -np.inf for f in filtrations[1:]] # the 0 index is the label of 'no bond' in bond_types
	)

	## Edges filtration
	mol_bonds = molecule.bonds.indices.T
	edges_filtration = np.zeros((mol_bonds.shape[1], num_parameters), dtype=np.float32) - np.inf
	for i, filtration in enumerate(filtrations):
		match filtration:
			case "bond_type":
				edges_filtration[:,i] = lines2bonds(mol=molecule, bond_types=bond_types)
			case "atomic_mass":
				continue
			case "charge":
				continue
			case 'bond_length':
				edges_filtration[:,i] = [st_rips.filtration(s) for s in mol_bonds.T]
			case _:
				raise Exception(f"Invalid filtration {filtration}. Available ones : bond_type, atomic_mass, charge, bond_length.")
	st.assign_batch_filtration(mol_bonds, edges_filtration, propagate=False)
	min_filtration = edges_filtration.min(axis=0)
	st.assign_batch_filtration(np.asarray([list(range(st.num_vertices))], dtype=int), np.asarray([min_filtration]*st.num_vertices, dtype=np.float32), propagate=False)
	## Nodes filtration
	for i, filtration in enumerate(filtrations):
		match filtration:
			case "charge":
				charges = molecule.atoms.charges
				st.fill_lowerstar(charges, parameter=i)
			case "atomic_mass":
				masses = molecule.atoms.masses
				null_indices = masses == 0
				if np.any(null_indices): # guess if necessary
					masses[null_indices] = guess_masses(molecule.atoms.types)[null_indices]
				# print(masses)
				st.fill_lowerstar(-masses, parameter=i)
			case _:
				pass
	st.make_filtration_non_decreasing() # Necessary ?
	return st


class Molecule2SimplexTree(BaseEstimator, TransformerMixin):
	"""
	Transforms a list of MDA-compatible files into a list of mulitparameter simplextrees
	
	Input
	-----
	 X: Iterable[path_to_files:str]
	
	Output
	------
	 Iterable[multipers.SimplexTreeMulti]
	
	Parameters
	----------
	 - filtrations : list of filtration names. Available ones : 'charge', 'atomic_mass', 'bond_length', 'bond_type'. Others are ignored.
	 - graph : bool. If true, will use the graph given by the molecule, otherwise, a Rips Complex Based on the distance. '
	 In that case bond_length is ignored (it's the 1rst parameter).
	"""
	def __init__(self, 
			delayed:bool=False, 
			filtrations:Iterable[str]=[],
			graph:bool=True,
			n_jobs:int=1) -> None:
		super().__init__()
		self.delayed=delayed
		self.n_jobs = n_jobs
		self.filtrations=filtrations
		self.graph=graph
		self._molecule_format=None
		return
	def fit(self, X:Iterable[str], y=None):
		if len(X) == 0:	return self
		test_mol = mda.Universe(X[0])
		self._molecule_format = test_mol.filename.split('.')[-1].lower()
		return self
	def transform(self,X:Iterable[str]):
		_to_simplextree = _mol2graphst if self.graph else _mol2ripsst
		to_simplex_tree = lambda path_to_mol2_file : _to_simplextree(path=path_to_mol2_file, filtrations=self.filtrations)
		if self.delayed:
			return [delayed(to_simplex_tree)(path) for path in X]
		return Parallel(n_jobs=self.n_jobs, prefer="threads")(delayed(to_simplex_tree)(path) for path in X)

