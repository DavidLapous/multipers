
from shapely.geometry import box as _rectangle_box
from shapely.geometry import Polygon as _Polygon
from shapely.ops import unary_union as _unary_union
shapely = True
from matplotlib.patches import Rectangle as RectanglePatch
import numpy as np
from matplotlib.cm import get_cmap
import matplotlib.pyplot as plt



def _rectangle(x,y,color, alpha):
	"""
	Defines a rectangle patch in the format {z | x  ≤ z ≤ y} with color and alpha
	"""
	return RectanglePatch(x, max(y[0]-x[0],0),max(y[1]-x[1],0), color=color, alpha=alpha)

def _d_inf(a,b):
	if type(a) != np.ndarray or type(b) != np.ndarray:
		a = np.array(a)
		b = np.array(b)
	return np.min(np.abs(b-a))

	

def plot2d(corners, box = [],*,dimension=-1, separated=False, min_persistence = 0, alpha=1, verbose = False, save=False, dpi=200, shapely = True, xlabel=None, ylabel=None, cmap=None):
	cmap = get_cmap("Spectral") if cmap is None else get_cmap(cmap)
	if not(separated):
		# fig, ax = plt.subplots()
		ax = plt.gca()
		ax.set(xlim=[box[0][0],box[1][0]],ylim=[box[0][1],box[1][1]])
	n_summands = len(corners)
	for i in range(n_summands):
		trivial_summand = True
		list_of_rect = []
		for birth in corners[i][0]:
			for death in corners[i][1]:
				death[0] = min(death[0],box[1][0])
				death[1] = min(death[1],box[1][1])
				if death[1]>birth[1] and death[0]>birth[0]:
					if trivial_summand and _d_inf(birth,death)>min_persistence:
						trivial_summand = False
					if shapely:
						list_of_rect.append(_rectangle_box(birth[0], birth[1], death[0],death[1]))
					else:
						list_of_rect.append(_rectangle(birth,death,cmap(i/n_summands),alpha))
		if not(trivial_summand):	
			if separated:
				fig,ax= plt.subplots()
				ax.set(xlim=[box[0][0],box[1][0]],ylim=[box[0][1],box[1][1]])
			if shapely:
				summand_shape = _unary_union(list_of_rect)
				if type(summand_shape) is _Polygon:
					xs,ys=summand_shape.exterior.xy
					ax.fill(xs,ys,alpha=alpha, fc=cmap(i/n_summands), ec='None')
				else:
					for polygon in summand_shape.geoms:
						xs,ys=polygon.exterior.xy
						ax.fill(xs,ys,alpha=alpha, fc=cmap(i/n_summands), ec='None')
			else:
				for rectangle in list_of_rect:
					ax.add_patch(rectangle)
			if separated:
				if xlabel:
					plt.xlabel(xlabel)
				if ylabel:
					plt.ylabel(ylabel)
				if dimension>=0:
					plt.title(rf"$H_{dimension}$ $2$-persistence")
	if not(separated):
		if xlabel != None:
			plt.xlabel(xlabel)
		if ylabel != None:
			plt.ylabel(ylabel)
		if dimension>=0:
			plt.title(rf"$H_{dimension}$ $2$-persistence")
	return

