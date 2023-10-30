import numpy as np
from os.path import expanduser
from torch_geometric.datasets import ModelNet

DATASET_PATH = expanduser("~/Datasets/")
import os


####################### MODELNET
def load_modelnet(version='10', sample_points = False, reset:bool=False, remove_faces=False):
	from torch_geometric.transforms import FaceToEdge, SamplePoints
	"""
	:param point_flag: Sample points if point_flag true. Otherwise load mesh
	:return: train_dataset, test_dataset
	"""
	assert version in ['10', '40']
	if sample_points:
		pre_transform, transform = FaceToEdge(remove_faces=remove_faces), SamplePoints(num=sample_points)
	else:
		pre_transform, transform = FaceToEdge(remove_faces=remove_faces), None
	path = f"{DATASET_PATH}/ModelNet{version}"
	if reset:
		# print(f"rm -rf {path}")
		os.system(f"rm -rf {path+'/processed/'}")
	train_dataset = ModelNet(path, name=version, train=True, transform=transform, pre_transform=pre_transform)
	test_dataset = ModelNet(path, name=version, train=False, transform=transform, pre_transform=pre_transform)
	return train_dataset, test_dataset


def get_ModelNet(dataset, num_graph, seed):
	train,test = load_modelnet(version=dataset[8:])
	test_size = len(test) / len(train)
	if num_graph >0:
		np.random.seed(seed)
		indices = np.random.choice(len(train), num_graph, replace=False)
		train = train[indices]
		indices = np.random.choice(len(test), int(num_graph*test_size), replace=False)
		test = test[indices]
		np.random.seed() # resets seed
	return train, test
	

def get(dataset:str, num_graph=0, seed=0, node_per_graph=0):
	if dataset.startswith("ModelNet"):
		return get_ModelNet(dataset=dataset, num_graph=num_graph, seed=seed)
	datasets = get_(dataset=dataset, num_sample=num_graph)
	graphs = []
	labels = []
	np.random.seed(seed)
	for data, ls in datasets:
		nodes = np.random.choice(range(len(data.pos)), replace=False, size=node_per_graph)
		for i,node in enumerate(nodes):
			data_ = data # if i == 0 else None # prevents doing copies
			graphs.append([data_, node])
			labels.append(ls[node])
	return graphs, labels


def get_(dataset:str, dataset_num:int|None=None, num_sample:int=0, DATASET_PATH = expanduser("~/Datasets/")):
	from torch_geometric.io import read_off
	if dataset.startswith("3dshapes/"):
		dataset_ = dataset[len("3dshapes/"):]
	else:
		dataset_ = dataset
	if dataset_num is None and "/" in dataset_:
		position = dataset_.rfind("/")
		dataset_num = int(dataset_[position+1:-4]) # cuts the "<dataset>/" and the ".off"
		dataset_ = dataset_[:position]

	if dataset_num is None: # gets a random (available) number for this dataset
		from os import listdir
		from random import choice
		files = listdir(DATASET_PATH+f"3dshapes/{dataset_}")
		if num_sample <= 0:
			files = [file for file in files if "label" not in file]
		else:
			files = np.random.choice([file for file in files if "label" not in file], replace=False, size=num_sample)
		dataset_nums = np.sort([int("".join([char for  char in file  if char.isnumeric()])) for file in files])
		
		print("Dataset nums : ", *dataset_nums)
		out = [get_(dataset_, dataset_num=num) for num in dataset_nums]
		return out

	path = DATASET_PATH+f"3dshapes/{dataset_}/{dataset_num}.off"
	data = read_off(path)
	faces = data.face.numpy().T
	# data = FaceToEdge(remove_faces=remove_faces)(data)
	#labels 
	label_path = path.split(".")[0] + "_labels.txt"
	f = open(label_path, "r")
	labels = np.zeros(len(data.pos), dtype="<U10") # Assumes labels are of size at most 10 chars
	current_label=""
	for i, line in enumerate(f.readlines()):
		if i %  2 == 0:
			current_label = line.strip()
			continue
		faces_of_label = np.array(line.strip().split(" "), dtype=int) -1 # this starts at 1, python starts at 0
		# print(faces_of_label.min())
		nodes_of_label = np.unique(faces[faces_of_label].flatten())
		labels[nodes_of_label] = current_label  # les labels sont sur les faces
	return data, labels
