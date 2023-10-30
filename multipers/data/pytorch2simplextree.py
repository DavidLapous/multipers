from tqdm import tqdm
import numpy as np
from torch_geometric.data.data import Data
import networkx as nx
from sklearn.base import BaseEstimator, TransformerMixin
from typing import Iterable


def modelnet2pts2gs(train_dataset, test_dataset , nbr_size = 8, exp_flag = True, labels_only = False,n=100, n_jobs=1, random=False):
	from sklearn.neighbors import kneighbors_graph
	""" 
    sample points and create neighborhoold graph
	"""	
	dataset = train_dataset + test_dataset
	indices = np.random.choice(range(len(dataset)),replace=False, size=n) if random else range(n)

	dataset:list[Data] = [dataset[i] for i in indices]
	_,labels = torch_geometric_2nx(dataset, labels_only=True)
	if labels_only: return labels
	
	def data2graph(data:Data):
		pos = data.pos.numpy()
		adj = kneighbors_graph(pos, nbr_size, mode='distance', n_jobs=n_jobs) 
		g = nx.from_scipy_sparse_array(adj, edge_attribute= 'weight')
		if exp_flag:
			for u, v in g.edges(): # TODO optimize
				g[u][v]['weight'] = np.exp(-g[u][v]['weight'])
		return g
		#TODO : nx.set_edge_attributes()

	return [data2graph(data) for data in dataset], labels
def torch_geometric_2nx(dataset, labels_only = False, print_flag = False, weight_flag = False):
	"""
	:param dataset:
	:param labels_only: return labels only
	:param print_flag:
	:param weight_flag: whether computing distance as weights or not
	:return:
	"""
	if labels_only:
		return None, [int(data.y) for data in dataset]
	def data2graph(data:Data):
		edges = np.unique(data.edge_index.numpy().T, axis=0)
		g = nx.from_edgelist(edges)
		edge_filtration = {(u,v):np.linalg.norm(data.pos[u] - data.pos[v]) for u,v in g.edges}
		nx.set_node_attributes(g,{node:0 for node in g.nodes}, "geodesic")
		nx.set_edge_attributes(g, edge_filtration, "geodesic")
		return g
	return [data2graph(data) for data in tqdm(dataset, desc="Turning Data to graphs")], [int(data.y) for data in dataset]


def modelnet2graphs(version = '10', print_flag = False, labels_only = False, a = 0, b = 10, weight_flag = False):
	""" load modelnet 10 or 40 and convert to graphs"""
	from torch_geometric.transforms import FaceToEdge
	from .shape3d import load_modelnet
	train_dataset, test_dataset = load_modelnet(version, point_flag = False)
	dataset = train_dataset + test_dataset
	if b>0:	dataset = [dataset[i] for i in range(a,b)]
	if labels_only:
		return torch_geometric_2nx(dataset, labels_only=True)
	dataset = [FaceToEdge(remove_faces=False)(data) for data in dataset]
	graphs, labels = torch_geometric_2nx(dataset, print_flag=print_flag, weight_flag= weight_flag)
	return graphs, labels




class Torch2SimplexTree(BaseEstimator,TransformerMixin):
	"""
	WARNING : build in progress
	PyTorch Data-like to simplextree.
	
	Input
	-----
	Class having `pos`, `edges`, `faces` methods
    
	Filtrations
	-----------
	 - Geodesic (geodesic rips)
	 - eccentricity 
	"""
	import multipers as mp
	
	def __init__(self, filtrations:Iterable[str]=[]):
		super().__init__()
		
	def fit(self, X, y=None):
		return self
	
	def transform(self,X:list[nx.Graph]):
		return