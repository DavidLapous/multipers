import numpy as np
from os.path import expanduser, exists
import networkx as nx
from warnings import warn
import pickle
from joblib import Parallel, delayed
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from scipy.spatial import distance_matrix
from sklearn.base import BaseEstimator, TransformerMixin, clone
import multipers as mp
from typing import Iterable

DATASET_PATH=expanduser("~/Datasets/")


def get(dataset:str, filtration:str):
	graphs, labels = get_graphs(dataset)
	try:
		for g in graphs:	
			for node in g.nodes:
				g.nodes[node][filtration]
	except:
		print(f"Filtration {filtration} not computed, trying to compute it ...", flush=1)
		compute_filtration(dataset, filtration)
	return get_graphs(dataset)



def get_from_file_old(dataset:str, label="lb"):
	from os import walk
	from scipy.io import loadmat
	from warnings import warn
	path = DATASET_PATH + dataset  +"/mat/"
	labels:list[int] = []
	gs:list[nx.Graph] = []
	for root, dir, files in walk(path):
		for file in files:
			file_ppties  = file.split("_")
			gid = file_ppties[5]
			i=0
			while i+1 < len(file_ppties) and file_ppties[i] != label :
				i+=1
			if i+1 >= len(file_ppties):
				warn(f"Cannot find label {label} on file {file}.")
			else:
				labels += [file_ppties[i+1]]
			adj_mat = np.array(loadmat(path + file)['A'], dtype=np.float32)
			gs.append(nx.Graph(adj_mat))
	return gs, labels


def get_from_file(dataset:str):
	from os.path import expanduser, exists
	path = DATASET_PATH + f"{dataset}/{dataset[7:]}."
	try:
		graphs_ids = np.loadtxt(path+"graph_idx")
	except:
		return get_from_file_old(dataset=dataset)
	labels:list[int] = LabelEncoder().fit_transform(np.loadtxt(path+"graph_labels"))
	edges = np.loadtxt(path+"edges", delimiter=',', dtype=int)-1
	has_intrinsic_filtration = exists(path+"node_attrs")
	graphs:list[nx.Graph] = []
	if has_intrinsic_filtration:
		F = np.loadtxt(path+"node_attrs", delimiter=',')
	for graph_id in tqdm(np.unique(graphs_ids), desc="Reading graphs from file"):
		nodes, = np.where(graphs_ids == graph_id)
		def graph_has_edge(u:int,v:int)->bool:
			if u in nodes or v in nodes:
				assert u in nodes and v in nodes, f"Nodes {u} and {v} are not in the same graph"
				return True
			return False
		graph_edges = [(u,v) for u,v in edges if graph_has_edge(u,v)]
		g = nx.Graph(graph_edges)
		if has_intrinsic_filtration:
			node_attrs = {node:F[node] for node in nodes}
			nx.set_node_attributes(g,node_attrs, "intrinsic")
		graphs.append(g)
	return graphs, labels


def get_graphs(dataset:str, N:int|str="")->tuple[list[nx.Graph], list[int]]:
	graphs_path = f"{DATASET_PATH}{dataset}/graphs{N}.pkl"
	labels_path = f"{DATASET_PATH}{dataset}/labels{N}.pkl"
	if not exists(graphs_path) or not exists(labels_path):
		if dataset.startswith("3dshapes/"):
			return get_from_file_old(dataset,)
		graphs, labels = get_from_file(dataset,)
		print("Saving graphs at :", graphs_path)
		set_graphs(graphs = graphs, labels = labels, dataset = dataset)
	else:
		graphs = pickle.load(open(graphs_path, "rb"))
		labels = pickle.load(open(labels_path, "rb"))
	from sklearn.preprocessing import LabelEncoder
	return graphs, LabelEncoder().fit_transform(labels)


def set_graphs(graphs:list[nx.Graph], labels:list, dataset:str, N:int|str=""): # saves graphs (and filtration values) into a file
	graphs_path = f"{DATASET_PATH}{dataset}/graphs{N}.pkl"
	labels_path = f"{DATASET_PATH}{dataset}/labels{N}.pkl"
	pickle.dump(graphs, open(graphs_path, "wb"))
	pickle.dump(labels, open(labels_path, "wb"))
	return

def reset_graphs(dataset:str, N=None): # Resets filtrations values on graphs
	graphs, labels = get_from_file(dataset)
	set_graphs(graphs,labels, dataset)
	return




def compute_ricci(graphs:list[nx.Graph], alpha=0.5, progress = 1):
	from GraphRicciCurvature.OllivierRicci import OllivierRicci
	def ricci(graph, alpha=alpha):
		return OllivierRicci(graph,alpha=alpha).compute_ricci_curvature()
	graphs = [ricci(g) for g in tqdm(graphs, disable = not progress, desc="Computing ricci")]
	def push_back_node(graph):
		# for node in graph.nodes:
			# graph.nodes[node]['ricciCurvature'] = np.min([graph[node][node2]['ricciCurvature'] for node2 in graph[node]] + [graph.nodes[node]['ricciCurvature']])
		node_filtrations = {
			node: -1 if len(graph[node]) == 0 else np.min([graph[node][node2]['ricciCurvature'] for node2 in graph[node]]) 
			for node in graph.nodes
		}
		nx.set_node_attributes(graph,node_filtrations,"ricciCurvature")
		return graph
	graphs = [push_back_node(g) for g in graphs]
	return graphs

def compute_cc(graphs:list[nx.Graph], progress = 1):
	def _cc(g):
		cc = nx.closeness_centrality(g)
		nx.set_node_attributes(g,cc,"cc")
		edges_cc = {(u,v):max(cc[u], cc[v]) for u,v in g.edges}
		nx.set_edge_attributes(g,edges_cc, "cc")
		return g
	graphs = Parallel(n_jobs=1, prefer="threads")(delayed(_cc)(g) for g in tqdm(graphs, disable = not progress, desc="Computing cc"))
	return graphs
	# for g in tqdm(graphs, desc="Computing cc"):
	# 	_cc(g)
	# return graphs

def compute_degree(graphs:list[nx.Graph], progress=1):
	def _degree(g):
		degrees = {i:1.1 if degree == 0 else 1 / degree   for i, degree in g.degree}
		nx.set_node_attributes(g,degrees,"degree")
		edges_dg = {(u,v):max(degrees[u], degrees[v]) for u,v in g.edges}
		nx.set_edge_attributes(g,edges_dg, "degree")
		return g
	graphs = Parallel(n_jobs=1, prefer="threads")(delayed(_degree)(g) for g in tqdm(graphs, disable = not progress, desc="Computing degree"))
	return graphs
	# for g in tqdm(graphs, desc="Computing degree"):
	# 	_degree(g)
	# return graphs

def compute_fiedler(graphs:list[nx.Graph], progress = 1): # TODO : make it compatible with non-connexe graphs
	def _fiedler(g):
		connected_graphs = [nx.subgraph(g, nodes) for nodes in nx.connected_components(g)]
		fiedler_vectors = [nx.fiedler_vector(g)**2 if g.number_of_nodes() > 2 else np.zeros(g.number_of_nodes()) for g in connected_graphs] # order of nx.fiedler_vector correspond to nx.laplacian -> g.nodes
		fiedler_dict = {
			node:fiedler_vector[node_index]
			for g,fiedler_vector in zip(connected_graphs, fiedler_vectors)
			for node_index,node in enumerate(list(g.nodes))
		}
		nx.set_node_attributes(g,fiedler_dict,"fiedler")
		edges_fiedler = {(u,v):max(fiedler_dict[u], fiedler_dict[v]) for u,v in g.edges}
		nx.set_edge_attributes(g,edges_fiedler, "fiedler")
		return g
	graphs = Parallel(n_jobs=1, prefer="threads")(delayed(_fiedler)(g) for g in tqdm(graphs, disable = not progress, desc="Computing fiedler"))
	return graphs
	# for g in tqdm(graphs, desc="Computing fiedler"):
	# 	_fiedler(g)
	# return graphs

def compute_hks(graphs:list[nx.Graph],t:float, progress = 1):
	def _hks(g:nx.Graph):
		w, vps = np.linalg.eig(nx.laplacianmatrix.normalized_laplacian_matrix(g, nodelist=g.nodes()).toarray()) # order is given by g.nodes order
		w = w.view(dtype=float)
		vps= vps.view(dtype=float)
		node_hks = {node:np.sum(np.exp(-t*w)*np.square(vps[node_index,:])) for node_index,node in enumerate(g.nodes)}
		nx.set_node_attributes(g, node_hks, f"hks_{t}")
		edges_hks = {(u,v):max(node_hks[u], node_hks[v]) for u,v in g.edges}
		nx.set_edge_attributes(g,edges_hks, f"hks_{t}")
		return g
	graphs = Parallel(n_jobs=1, prefer="threads")(delayed(_hks)(g) for g in tqdm(graphs, disable = not progress, desc=f"Computing hks_{t}"))
	return graphs

def compute_geodesic(graphs:list[nx.Graph], progress=1):
	def _f(g:nx.Graph):
		try:
			nodes_intrinsic = {i:n["intrinsic"] for i,n in g.nodes.data()}
		except:
			warn("This graph doesn't have an intrinsic filtration, will use 0 instead ...")
			nodes_intrinsic = {i:0 for i,n in g.nodes.data()}
			# return g
		node_geodesic = {i:0 for i in g.nodes}
		nx.set_node_attributes(g, node_geodesic, f"geodesic")
		edges_geodesic = {(u,v):np.linalg.norm(nodes_intrinsic[u] - nodes_intrinsic[v]) for u,v in g.edges}
		nx.set_edge_attributes(g,edges_geodesic, f"geodesic")
		return g 
	graphs = Parallel(n_jobs=1, prefer="threads")(delayed(_f)(g) for g in tqdm(graphs, disable = not progress, desc=f"Computing geodesic distances on graphs"))
	return graphs

def compute_filtration(dataset:str, filtration:str, **kwargs):
	if filtration == "ALL":
		reset_graphs(dataset) # not necessary
		graphs,labels = get_graphs(dataset, **kwargs)
		graphs = compute_geodesic(graphs)
		graphs = compute_cc(graphs)
		graphs = compute_degree(graphs)
		graphs = compute_ricci(graphs)
		graphs = compute_fiedler(graphs)
		graphs = compute_hks(graphs, 10)
		set_graphs(graphs=graphs, labels=labels, dataset=dataset)
		return
	graphs,labels = get_graphs(dataset, **kwargs)
	if filtration == "dijkstra":
		return
	elif filtration == "cc":
		graphs = compute_cc(graphs)
	elif filtration == "degree":
		graphs = compute_degree(graphs)
	elif filtration == "ricciCurvature":
		graphs = compute_ricci(graphs)
	elif filtration == "fiedler":
		graphs = compute_fiedler(graphs)
	elif filtration == "geodesic":
		graphs = compute_geodesic(graphs)
	elif filtration.startswith('hks_'):
		t = int(filtration[4:]) # don't want do deal with floats, makes dots in title...
		graphs = compute_hks(graphs=graphs, t=t)
	else:
		warn(f"Filtration {filtration} not implemented !")
		return
	set_graphs(graphs=graphs, labels=labels, dataset=dataset)
	return



class Graph2SimplexTrees(BaseEstimator,TransformerMixin):
	"""
	Transforms a list of networkx graphs into a list of simplextree multi
	
	Usual Filtrations
	-----------------
	- "cc" closeness centrality
	- "geodesic" if the graph provides data to compute it, e.g., BZR, COX2, PROTEINS
	- "degree" 
	- "ricciCurvature" the ricci curvature
	- "fiedler" the square of the fiedler vector
	"""
	def __init__(self, filtrations:Iterable[str]=[], delayed=False, num_collapses=100, progress:bool=False):
		super().__init__()
		self.filtrations=filtrations # filtration to search in graph
		self.delayed = delayed # reverses the filtration #TODO
		self.num_collapses=num_collapses
		self.progress=progress
	def fit(self, X, y=None):
		return self
	def transform(self,X:list[nx.Graph]):
		def todo(graph, filtrations=self.filtrations) -> mp.SimplexTreeMulti: 
			st = mp.SimplexTreeMulti(num_parameters=len(filtrations))
			nodes = np.asarray(graph.nodes, dtype=int).reshape(1,-1)
			nodes_filtrations = np.asarray([[graph.nodes[node][filtration] for filtration in filtrations] for node in graph.nodes], dtype=np.float32)
			st.insert_batch(nodes, nodes_filtrations)
			edges = np.asarray(graph.edges, dtype=int).T
			edges_filtrations = np.asarray([[graph[u][v][filtration] for filtration in filtrations] for u,v in graph.edges], dtype=np.float32)
			st.insert_batch(edges,edges_filtrations)
			if st.num_parameters == 2:	st.collapse_edges(num=self.num_collapses) # TODO : wait for a filtration domination update
			# st.make_filtration_non_decreasing() ## Ricci is not safe ...
			return [st] # same output for each pipelines, some have a supplementary axis.
		return [delayed(todo)(graph) for graph in X] if self.delayed else Parallel(n_jobs=-1, prefer="threads")(delayed(todo)(graph) for graph in tqdm(X, desc="Computing simplextrees from graphs", disable=not self.progress))
