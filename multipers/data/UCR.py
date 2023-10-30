import numpy as np
from os.path import expanduser
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def get(dataset:str="UCR/Coffee", test:bool=False, DATASET_PATH:str=expanduser("~/Datasets/"), dim=3,delay=1,skip=1):
	from gudhi.point_cloud.timedelay import TimeDelayEmbedding
	dataset_path = DATASET_PATH + dataset + "/" + dataset[4:]
	dataset_path +=  "_TEST.tsv" if test else "_TRAIN.tsv"
	data = np.array(pd.read_csv(dataset_path, delimiter='\t', header=None, index_col=None))
	Y = LabelEncoder().fit_transform(data[:,0])
	data = data[:,1:]
	tde = TimeDelayEmbedding(dim=dim, delay=delay, skip=skip).transform(data)
	return tde, Y
def get_train(*args, **kwargs):
	return get(*args, **kwargs, test=False)
def get_test(*args, **kwargs):
	return get(*args, **kwargs, test=True)
