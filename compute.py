from argparse import ArgumentParser, Namespace
from sys import argv
from joblib import cpu_count
import numpy as np
from os import system

if "self_test" in argv:
	script_name = argv[0]
    # compiles pykeops
	for _pipeline in ("multismk", "multismi", "multisurface", "mmaimg"):
		# n_jobs = cpu_count() // 2
		n_jobs = 1

		## GRAPHS
		easy_graph="graphs/COX2"
		assert system(f"python {script_name} --dataset {easy_graph} "
			f"--filtrations degree --filtrations cc --pipeline {_pipeline} "
			f"--train_k 2 --test_k .2 --degrees 0  --degrees -1 --in_strategy exact "
			f"--num_rescales 2 --out_strategy regular_closest --out_resolution 5 "
			f"--n_jobs {n_jobs} --test 1") == 0
		## More number of parameters
		assert system(f"python {script_name} --dataset {easy_graph} --filtrations degree "
			f" --filtrations cc --filtrations hks_10 --filtrations hks_5 --pipeline {_pipeline} "
			f"--train_k 2 --test_k .2 --degrees 0  --in_strategy exact --num_rescales 1 "
			f"--out_strategy regular_closest --out_resolution 5 --n_jobs {n_jobs} --test 1") == 0
		
		## point clouds
		n_jobs = 1 
		easypoint_cloud="orbit"
		for _complex in ("rips", "alpha"):
			for _kernel in ["gaussian"]:
				assert system(f"python {script_name} --dataset {easypoint_cloud} "
					f"--pipeline {_pipeline} --train_k 2 --test_k .2 --degrees 1 --in_strategy exact "
					f" --out_strategy regular --out_resolution 5 --complex {_complex} "
					f"--rips_threshold -1 --kernel {_kernel} --kde_bandwidths -.1 "
					f"--dtm_masses .1 --n_jobs {n_jobs} --num_pts 50 --num_samples 5 --test 1") == 0
	
	print("\n\n\n -------------- All tests passed. ")
	exit()
if "--test" not in argv:
	## This tests the scripts, and also precompiles pykeops, without entering the gridsearch (test=1)
	print("----------------------------------------------")
	print("----------------------------------------------")
	print("Testing script")
	exit_status = system(" ".join(["python ",*argv," --test 1 --n_jobs 1"]))
	assert  exit_status == 0, f"Testing script failed. error {exit_status}"
	print("----------------------------------------------")
	print("----------------------------------------------")
	print("Script launched with arguments:\npython", *argv)
	print("----------------------------------------------")
	print("----------------------------------------------", flush=True)
else:
    print("----------------------------------------------")
    print("----------------------------------------------")
    print("Testing...")
    print("----------------------------------------------")
    print("----------------------------------------------", flush=True)




############################################ARGS PARSER
p = ArgumentParser()
p.add_argument("-d","--dataset", type=str, required=True, help="The dataset on which to do the computation. Either UCR, e.g., UCR/Coffee, graphs : graphs/BZR, orbit, 3dshapes, e.g., 3dshapes/Airplane ") #Threshold infinite values to compute diagram distance
p.add_argument("-p", "--pipeline", required=True,type=str, help="The pipeline to apply to the dataset. Available : dummy, filvec, pervec, sw, {rd,dr,multi}_{smi, smk, hilbert}, sw, smk, smh, smi, pl,pl_p, pi, pi_p. Where sm -> signed measure, sm{i,k,h} -> image, kernel, hilbert, rd -> rips+density bifiltration, dr-> degree+rips bifiltration, multi -> custom 1critical multi filtration (eg. graphs, molecules), sw -> sliced wasserstein, pl -> persistence landscape, pi -> persistance image.") # pipeline
p.add_argument("-fc","--final_classifier", default="rf", type=str, help="When the final input is a vector, this defines the final classifier") 
p.add_argument("-f","--filtration", default="", type=str, help="For 1 parameter filtration, the custom filtration. for example for graphs : ricciCurvature") # filtration on the graph (1-parameter)
p.add_argument("-fs","--filtrations", default=[], type=str, action="append", help = "For multifiltration, the filtrations to consider. e.g. --filtrations ricciCurvature --filtrations cc --filtrations geodesic. Depending on the dataset, available ones for graphs are cc,degree,fiedler,ricciCurvature,geodesic.") # filtrations on the graph (multi-parameter)
p.add_argument("-tk", "--train_k", default=10, type=int, help="Number of cross validations to choose the parameters during the training") # number of kfold for cross validation
p.add_argument("-k", "--test_k", default=10, type=float, help="Number of n-folds for testing. If 0<x<1, will do a train-test-split with a proportion of x for the test.") # number of kfold for test

p.add_argument("-t","--diagram_threshold", default=np.inf, type=float, help="For 1 parameter, thresholds persistence values to this threshold.") #Threshold infinite values to compute diagram distance
p.add_argument("-ns", "--num_samples", default=-1, type=int, help="number of data for orbit5k, and 3dshapes") # number of data (e.g. graph, or orbit data) samples
p.add_argument("-npts", "--num_pts", default=0, type=int, help="number of points / nodes in each data for 3dshapes / orbit") # number of pts per sample, if (synthetic)
p.add_argument("-res", "--in_resolution", type=int, default=100, help="For multiparameter pipelines, the resolution to compute the signed measure. e.g. 100 will do the computation on a [100]*num_parameter grid.")

p.add_argument("-is", "--in_strategy", default="exact", help="Infers the grid on which to compute the topological invariant.")
p.add_argument("--in_individual_grid", default=True, type=int, help="Whether or not to compute signed meaasure on individual grids. Significantly faster if true.")


p.add_argument("-os", "--out_strategy", nargs='+', action='extend', help="Infers the grid on which to compute the topological invariant. Available : regular, quantile, exact.")

p.add_argument("-ores", "--out_resolution", type=int, action='append', help="For multiparameter, vectorized pipelines, e.g. *_{smi, hilbert} the resolution of these vectors.")

p.add_argument("-numdir", "--num_directions", type=int, default=100, help="For multiparameter, vectorized pipelines, e.g. *_{smi, hilbert} the resolution of these vectors.")




p.add_argument("-cplx", "--complex", default="rips", help="Simplicial complex used on the point cloud.")
p.add_argument("-krnl", "--kernel", default="gaussian", help="Codensity-like kernel to use, e.g., gaussian, exponential.")

p.add_argument("--kde_bandwidths", type=float, action='append',default=[], help="For point cloud dataset, the bandwidths of the Kernel Density Estimation to cross validate.")
p.add_argument("--dtm_masses", type=float, action='append',default=[], help="For point cloud dataset, the selected masses of the DistanceToMeasure to cross validate to compute a codensity filtration.")

p.add_argument("--drop_quantile", default=0, help="When inferring the filtrations, drop filtration values lower than this q and greater than 1-q.", type=float)
p.add_argument("--num_rescales", default=1, help="Number of rescales per filtration for Kernel.", type=int)

p.add_argument("--rips_threshold", type=float, default=np.inf, help="Maximum radius value for rips, when using a pipeline using rips.")

# p.add_argument("-rb", "--rips_bandwidth", type=float, default=0., help="")
p.add_argument("--sparse_rips", type=float, default=None, help="Value of the sparse rips, if using it.")
p.add_argument('--extended', action='append', type=int, default=[], help="Extended persistence for 1 parameter filtrations. if -1 : will use [0,2,5,7]. (Order given by gudhi)") #TODO remove
p.add_argument("--geodesic_backend", default="torch_geometric")
p.add_argument("-s", "--seed", default=None, type=int, help="Some pipeline have randomized fit, this controls their seed.") # node selection seed
p.add_argument("--test", default=False, type=bool, help="Reduces the number of input, to ensure the pipelines are working. DO NOT USE WHEN NOT TESTING.") 

p.add_argument('--degrees', action='append', type=int, help="The homological degrees to consider. Note : `None` represent the euler characteristic.")
p.add_argument('--rank_degrees', action='append', type=int, help="The homological degrees, for the rank invariant, to consider. ")


p.add_argument("--n_jobs", default=cpu_count(), type=int, help="The number of threads to use.")

p.add_argument("--self_test",default=False, type=bool, help="Computes small tests to ensure the script works.")
args = p.parse_args()
np.random.seed(args.seed)

args.degrees = [] if args.degrees is None else [None if d <0 else d for d in args.degrees]
args.rank_degrees = [] if args.rank_degrees is None else args.rank_degrees
assert len(args.degrees) >0  or len(args.rank_degrees) >0, "Provide homological degree to compute."


print("Loading core dependencies...", end="", flush=True)
import multipers as mp
mp.simplex_tree_multi.SAFE_CONVERSION=True
from sklearn.base import BaseEstimator, TransformerMixin
import multipers.ml.signed_measures as mms
import multipers.ml.mma as mma
from sklearn.model_selection import GridSearchCV
from multipers.ml.tools import get_filtration_weights_grid
import multipers.ml.one as mmo ## only for 1param
from sklearn.svm import SVC
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from random import choice
from copy import deepcopy
from multipers.ml.accuracies import accuracy_to_csv
from joblib import parallel_backend
print("Done.", flush=True)



## FROM https://stackoverflow.com/questions/71845452/convert-argparse-namespace-to-dict-recursively
def namespace_to_dict(namespace):
	return {
		k: namespace_to_dict(v) if isinstance(v, Namespace) else v
		for k, v in vars(namespace).items()
	}
results_kwargs = namespace_to_dict(args) ## To be written in the end in the csv
dataset = results_kwargs.pop("dataset")






## ARGS magic
num_parameters = len(args.filtrations) if "graphs/" in args.dataset else 2
args.grid_shape= [args.in_resolution]*num_parameters
shuffle = True if args.filtration != "dijkstra" else False
extended = args.extended
if len(extended) == 1 and extended[0] == -1:
	extended = [0,2,5,7] # ord0, ext+0, rel1, Ext-1
	degrees = list(range((max(extended) // 4)+1))
elif len(extended) > 0:
	extended = extended[1:]
	degrees = list(range((max(extended) // 4)+1))
else:
	degrees = args.degrees
	extended = False
args.extended = extended
args.degrees = degrees

### Final classifiers
match args.final_classifier:
	case "rf":
		from sklearn.ensemble import RandomForestClassifier
		final_classifier = RandomForestClassifier()
		final_classifier_parameters={}
	case "xgboost":
		from xgboost import XGBClassifier
		final_classifier = XGBClassifier()
		final_classifier_parameters={}
	case "mlp":
		from sklearn.neural_network import MLPClassifier
		final_classifier = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000)
		final_classifier_parameters={}
	case "svm":
		final_classifier = SVC(kernel="rbf")
		final_classifier_parameters={}
	case "adaboost":
		from sklearn.ensemble import AdaBoostClassifier
		final_classifier = AdaBoostClassifier()
		final_classifier_parameters={}
	case "knn":
		from sklearn.neighbors import KNeighborsClassifier
		final_classifier = KNeighborsClassifier()
		final_classifier_parameters={}
	case "lda":
		from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
		final_classifier = LinearDiscriminantAnalysis()
		final_classifier_parameters={}
	case "qda":
		from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
		final_classifier = QuadraticDiscriminantAnalysis()
		final_classifier_parameters={}
	case "naivebayes":
		from sklearn.naive_bayes import GaussianNB
		final_classifier = GaussianNB()
		final_classifier_parameters={}
	case "lgbm":
		from lightgbm import LGBMClassifier
		final_classifier = LGBMClassifier(verbose=-1, n_jobs=1)
		final_classifier_parameters={}
	case _:
		raise Exception(f"Classifier {args.final_classifier} not implemented.")



###########################################DATASET / FILTRATIONS
SM_parameters = {
	"degrees":args.degrees,
	"rank_degrees":args.rank_degrees,
	"progress":True,
	"n_jobs":args.n_jobs,
	"filtration_quantile":args.drop_quantile, 
	"resolution":args.grid_shape, 
	"_möbius_inversion": True,
	"normalize_filtrations":False, ## Will be done in SMF
	"fit_fraction":1, ## Why not
	"expand":args.complex == "rips" and not "graphs" in args.dataset,
	# "out_resolution": args.out_resolution, ## not needed, integration is done afterward
	"grid_strategy":args.in_strategy,
	"enforce_null_mass": False, ## DEFINED AT THE DATASET LEVEL
	"individual_grid": args.in_individual_grid,
}

print("Getting dataset", flush=True)
if args.dataset == "orbit":
	from multipers.data import get_orbit5k
	X,Y = get_orbit5k(num_data=args.num_samples, num_pts=args.num_pts)
	SM_parameters["enforce_null_mass"] = "smk" in args.pipeline and not args.rips_threshold in [np.inf, -1]
elif args.dataset.startswith("UCR/"):
	from multipers.data import UCR
	xtrain, ytrain = UCR.get_train(dataset=args.dataset)
	xtest, ytest = UCR.get_test(dataset = args.dataset)
	## FOR ACCURACY 2 CSV
	args.test_k = len(xtest) / (len(xtrain) + len(xtest))
	shuffle = False
	X = xtrain + xtest
	Y = np.concatenate([ytrain, ytest])
	## TESTS THAT ACCURACY TO CSV RETRIEVES THE SAME
	from sklearn.model_selection import train_test_split
	_xtrain, _xtest, _ytrain, _ytest = train_test_split(X, Y, shuffle=shuffle, test_size=args.test_k)
	assert np.array_equal(xtrain, _xtrain)
	assert np.array_equal(ytrain,_ytrain)
	assert np.array_equal(xtest, _xtest)
	assert np.array_equal(ytest,_ytest)
	SM_parameters["enforce_null_mass"] = "smk" in args.pipeline and not args.rips_threshold in [np.inf, -1]
elif args.dataset == "immuno":
	from multipers.data import immuno_regions
	X,Y = immuno_regions.get()
	SM_parameters["enforce_null_mass"] = "smk" in args.pipeline and not args.rips_threshold in [np.inf, -1]
elif args.dataset.startswith("3dshapes/"):
	from multipers.data import shape3d
	X,Y = shape3d.get(dataset = args.dataset, num_graph=args.num_samples, node_per_graph = args.num_pts)
	args.kde_bandwidths=[];args.masses = []
	
elif args.dataset.startswith("graphs/"):
	from multipers.data import graphs
	args.kde_bandwidths=[];args.masses = []
	print("Checking graphs filtrations ...", flush=True)
	filtrations = args.filtrations
	if args.filtrations == ["reset"]:
		print("Computing all filtrations and leaving...", end="\n")
		graphs.compute_filtration(args.dataset, "ALL")
		print("Done")
		exit()
	if len(filtrations) == 0:
		assert args.filtration != "", "Provide a filtration for graph data!"
		filtrations = [args.filtration]
	for f in filtrations:
		graphs.get(dataset = args.dataset, filtration=f) # Ensures that the filtration f is computed on this dataset
	X,Y = graphs.get(dataset = args.dataset, filtration=filtrations[0]) # Fills X and Y
	SM_parameters["enforce_null_mass"] = "smk" in args.pipeline
	SM_parameters["expand"]=False
# elif args.datasets == "ModelNet10":
# 	import os
# 	if not os.path.exists()
# 	train, test = shape3d.load_modelnet('10')
# train_graphs, train_labels = shape3d.torch_geometric_2nx(train)
else:
	raise Exception(f"Dataset {args.dataset} not yet supported.")


if args.test:
	indices = range(min(10, len(X)))
	
	X=[X[i] for i in indices]
	Y=[i%2 for i in indices]
	# args.test_k = .2 ## UCR overwrites this
	shuffle=False

	print(indices, Y)



print("Classes :", np.unique(Y))


############################# SIGNED MEASURES MULTI PIPELINES

print("------------- SimplexTree 2 Signed Measure parameters")
print(SM_parameters)
print("------------- ")
STM2SM = mms.SimplexTree2SignedMeasure(
		**SM_parameters
	)
STMs2SMs = mms.SimplexTrees2SignedMeasures(
		**SM_parameters
	)
# DR2SM = p2.DegreeRips2SignedMeasure(
# 		degrees=args.degrees,
# 		min_rips_value=0,
# 		max_rips_value=args.rips_threshold,
# 		min_normalized_degree=0,
# 		max_normalized_degree=0.3, # TODO, make a threshold for that
# 		grid_granularity=args.in_resolution,
# 		n_jobs=args.n_jobs, 
# 		progress=True,
# 		_möbius_inversion= True,
# 	)
SMD1 = mms.SignedMeasure2SlicedWassersteinDistance(num_directions=args.num_directions, n_jobs=args.n_jobs, progress=True)
SMDs = mms.SignedMeasures2SlicedWassersteinDistances(
	num_directions=args.num_directions, n_jobs=args.n_jobs, 
	progress=True,
	scales = None if args.num_rescales <= 1 else get_filtration_weights_grid(
		num_parameters=num_parameters,
		weights=np.unique([1.]+list(np.linspace(.1,10.,args.num_rescales -1))),
		remove_homothetie=False,
	)
)
#### NUMBER of axis ?
num_bandwidth=len(args.kde_bandwidths)
num_masses=len(args.dtm_masses)
num_bandwidth += num_masses
num_bandwidth = max(num_bandwidth,1)
num_kernel_rescale = 1 if SMDs.scales is None else len(SMDs.scales)
num_axes = (num_bandwidth)*num_kernel_rescale
print(f"Number of axis : bandwidths ({num_bandwidth}) x num_scales ({num_kernel_rescale}) = {num_axes}", flush=1)



############################# MMA Pipelines 
int_degrees = [d for d in args.degrees if d is not None]
ST2MMA = mma.SimplexTree2MMA(
	nlines=args.num_directions, 
	n_jobs=args.n_jobs,
	prune_degrees_above=np.max(int_degrees)+1 if len(int_degrees)>0 else None,
	progress=True,
)
ST2MMA_parameters={
	"nlines":[args.num_directions],
}
MMAF = mma.MMAFormatter() 
MMAF_parameters={
	"MMAF__normalize":				[False], # Done at the preprosess stage
	"MMAF__degrees":				[int_degrees],
	"MMAF__weights":				[None] if args.num_rescales <= 1 else get_filtration_weights_grid(
										num_parameters=num_parameters,
										weights=np.unique([1.]+list(np.linspace(.5,10.,args.num_rescales-1))),
										remove_homothetie=False,
									),
	"MMAF__axis": 					list(range(num_bandwidth)),
	"MMAF__quantiles":				[[args.drop_quantile]*2],
}

MMA2IMG = mma.MMA2IMG(n_jobs=1,progress=False,flatten=True,degrees=int_degrees)
MMA2IMG_parameters = {
	"MMA2IMG__degrees":						[int_degrees],
	"MMA2IMG__bandwidth":					[
											0.0001,0.001,0.01,0.1,0.2,
											# -0.001,-0.01,-0.1,-0.2
											], ## Normalized filtrations, negative is rectangle
	"MMA2IMG__power":						[0,1],
	"MMA2IMG__resolution":					args.out_resolution,
	"MMA2IMG__grid_strategy":				args.out_strategy, # should be fine in every cases
	"MMA2IMG__normalize":					[False,True],
}







############################ VERSION WITH DISCRETE CONVOLUTION : Faster but smaller precision
# SMF = p2.SignedMeasureFormatter(unsparse=True)
# SMF_parameters = {
# 	# "SMF__filtrations_weights": 	[None] if "hilbert" in args.pipeline else p2.get_filtration_weights_grid(num_parameters=num_parameters, weights=[1,.1,10]),
# 	"SMF__axis": 					list(range(num_bandwidth)) if args.pipeline.startswith("rd_") else [None],
# 	"SMF__resolution":				[20, 50, 100] if num_parameters == 2 else [20],
# }
# SMM2CV = p2.SignedMeasure2Convolution(flatten=True, n_jobs=-1)
# print("Num parameters", num_parameters)
# SMM2CV_parameters = {
# 	"SMM2CV__bandwidth":		p2.get_filtration_weights_grid(num_parameters=num_parameters, weights=[1.]+list(np.linspace(.1,10.,args.num_rescales-1)), remove_homothetie=False),
# 	# "SMM2CV__resolution": 		[args.out_resolution],
# 	# "SMM2CV__infer_grid_strategy":	["exact"], # should be fine in every cases
# }

########## VERSION WITH SPARSE CONVOLUTION : Slower but better precision (With pykeops speed is fine)
print("Num parameters", num_parameters)
print("Num bandwidths", num_bandwidth)


class Identity(BaseEstimator, TransformerMixin):
	def __init__(self):
		pass

	def fit(self, x, y=None):
		return self

	def transform(self, X, y=None):
		# X= np.asarray(X).squeeze()
		# return np.asarray(X).reshape((len(X),-1))
		print([x[0].shape for x in X])
		return [np.asarray([np.asarray(stuff).flatten() for stuff in x]).flatten() for x in X]
		# return X


SMF = mms.SignedMeasureFormatter(verbose=False) 
SMF_parameters = {
	"SMF__axis": 					list(range(num_bandwidth)),
	# "SMF__resolution":				[20, 50, 100] if num_parameters == 2 else [5],
}
if "surface" in args.pipeline:
	SMF_parameters["SMF__integrate"] = [True]
	SMF.flatten = True
	SMF_parameters["SMF__grid_strategy"] = args.out_strategy
	SMF_parameters["SMF__filtrations_weights"] = [None]
	SMF_parameters["SMF__resolution"] = args.out_resolution
else:
	SMF_parameters["SMF__normalize"] = [True]
	SMF_parameters["SMF__filtrations_weights"] = get_filtration_weights_grid(
		num_parameters=num_parameters, 
		weights=np.unique([1.]+list(np.linspace(.01,1.,args.num_rescales-1))),
		remove_homothetie=False,
		)
	
if "smi" in args.pipeline:
	SMM2CV = mms.SignedMeasure2Convolution(flatten=True, n_jobs=1)
	SMM2CV_parameters = {
		"SMM2CV__bandwidth":		[0.001,0.01,0.1,0.2], ## Normalized filtrations
		"SMM2CV__resolution": 		args.out_resolution,
		"SMM2CV__grid_strategy":	args.out_strategy, # should be fine in every cases
	}



############################## DISTANCE MATRIX MAGIC
if "smk" in args.pipeline:
	import multipers.ml.kernels as mmk


	D2DL = mmk.DistanceMatrices2DistancesList()
	DL2D = mmk.DistancesLists2DistanceMatrices()

	DM2K = mmk.DistanceMatrix2Kernel()

	DM2K_parameters= {
		"DM2K__sigma":[0.001, 0.01,1, 10, 100, 1000], # TODO : With measures between 0,1 distances should also be very small ... 
		"DM2K__axis":list(range(num_axes)), # if args.pipeline == "rd_smk" else [None], # Only for rips+density
		"DM2K__weights":get_filtration_weights_grid(num_parameters=len(SM_parameters["degrees"]) + len(SM_parameters["rank_degrees"]), weights=[1,.1,10]),
	}



SVMP = SVC(kernel = "precomputed")
SVMP_parameters = {
	"SVMP__kernel" : 	["precomputed"],
	"SVMP__C" : 		[0.001, 0.01,1, 10, 100, 1000],
}


########################################### SimplexTree and Diagram Transformers

if args.dataset.startswith("3dshapes/"):
	true_geodesic = args.geodesic_backend == "torch_geometric"
	ToSimplexTree = mmo.TorchData2DijkstraSimplexTree(true_geodesic=true_geodesic, progress=True) # dtype=None delays the computation for multithread with simplextrees
	ToSimplexTreeMulti = None # TODO ?
	ToSignedMeasure = None
	SMD=None
elif args.dataset in ["orbit", "immuno"] or args.dataset.startswith("UCR/"): # point clouds
	import multipers.ml.point_clouds as mmp
	RDs2STs = mmp.PointCloud2SimplexTree(
		bandwidths=args.kde_bandwidths,
		masses=args.dtm_masses,
		num_collapses='full' if args.complex == "rips" else 0, 
		progress=True,
		sparse=args.sparse_rips,  
		threshold=args.rips_threshold,
		n_jobs=args.n_jobs,
		complex=args.complex,
		kernel=args.kernel,
	)
	ToSimplexTree = mmo.PointCloud2SimplexTree(threshold=args.diagram_threshold)
	ToSimplexTreeMulti = RDs2STs
	ToSignedMeasure = STMs2SMs
	SMD = SMDs
	assert num_bandwidth>0 or "one" in args.pipeline, "Need a bandwidth/mass parameter to compute a codensity axis."
	# TODO pop filtrations, ... from args
elif args.dataset.startswith("graphs/") or args.dataset.startswith("ModelNet"):
	ToSimplexTree = mmo.Graph2SimplexTree(f=args.filtration)
	import multipers.data.graphs as mdg
	ToSimplexTreeMulti = mdg.Graph2SimplexTrees(filtrations=args.filtrations)
	# STM2SM.infer_filtration_strategy = "exact"
	STMs2SMs.num_collapses = 0
	STMs2SMs.expand = False
	# STM2SM.sparse = True
	ToSignedMeasure = STMs2SMs
	SMD=SMDs
else:
	raise Exception(f"Dataset {args.dataset} not yet supported.")

print("Transformers : ", ToSimplexTree, ToSimplexTreeMulti, ToSignedMeasure, SMD,sep="\n    ")

print("Initializing diagrams pipeline", flush=True)
# The other pipelines are taking diagrams as an input, so we can factorize the pipeline from here. This allows for multithread computation of the dgms
compute_diagram_pipe = Pipeline([
	("st", ToSimplexTree),
	("dgm", mmo.SimplexTree2Dgm(n_jobs=args.n_jobs, threshold=args.diagram_threshold, extended=extended, degrees=degrees, progress=True))
])




# ## Final args
print("Arguments", args)
###########################################PIPELINES PARAMETERS
print("Initializing pipeline", flush=True)
to_switch = args.pipeline.lower()
match to_switch:
	case "dummy": # Dummy
		pipeline = DummyClassifier()
		parameters = {}
	case "mmaimg":
		assert len(int_degrees)>0, "Provide degrees to compute."
		ModuleTransformer = Pipeline([
			('st',ToSimplexTreeMulti),
			('mma',ST2MMA),
			('normalize_step',mma.MMAFormatter(dump=True,normalize=True,verbose=True,degrees=int_degrees,quantiles=[args.drop_quantile]*2))
		])
		X = ModuleTransformer.fit_transform(X)
		pipeline = Pipeline([
			("MMAF",MMAF),
			("MMA2IMG",MMA2IMG),
			("final_classifier", final_classifier),
		])
		parameters = {}
		parameters.update(MMAF_parameters)
		parameters.update(MMA2IMG_parameters)
		parameters.update(final_classifier_parameters)
	case "filvec":
		print("------------filvec pipeline")
		svm = SVC(kernel="rbf")
		parameters = {
			"hist__quantile":[0.],
			"hist__bins":[100,200,300],
			"svm__kernel" : ["rbf"],
			"svm__gamma" : [0.01, 0.1, 1, 10, 100],
			"svm__C" : [0.001,0.01,1, 10, 100, 1000],
		}
		pipeline = Pipeline([
			("st",ToSimplexTree), 
			("hist", mmo.SimplexTree2Histogram()),
			("svm",svm)
		])
	case "msmi"|"multismi":
		print("------------smi pipeline")
		### PREPROCESSING : transform to signed measure
		SignedMeasureTransformer = Pipeline([('st', ToSimplexTreeMulti), ("sm",ToSignedMeasure)])
		X = SignedMeasureTransformer.fit_transform(X=X)
		SMF.verbose=False
		pipeline = Pipeline([
			("SMF", SMF),
			("SMM2CV", SMM2CV),
			("final_classifier", final_classifier),
		],
		# memory=memory
		)
		# SMF_parameters["SMF__filtrations_weights"] = [[1,1]]
		parameters = {}
		parameters.update(SMF_parameters)
		parameters.update(SMM2CV_parameters)
		parameters.update(final_classifier_parameters)
	case "msurface"|"multisurface":
		print("------------surface pipeline")
		# ToSignedMeasure._möbius_inversion = False ## integrates in the SMF
		# ToSignedMeasure.flatten=True
		
		SignedMeasureTransformer = Pipeline([('st', ToSimplexTreeMulti), ("sm",ToSignedMeasure)])
		X = SignedMeasureTransformer.fit_transform(X=X)
		pipeline = Pipeline([
			("SMF", SMF),
			("final_classifier", final_classifier),
		])
		parameters = {}
		parameters.update(SMF_parameters)
		parameters.update(final_classifier_parameters)
	case "msmk"|"multismk":
		print("------------smk pipeline")
		# ToSignedMeasure.sparse = True
		SMD = SMDs
		SMF.verbose=True
		SMF.axis = -1
		SMF.normalize=True
		SignedMeasureDistancesTransformer = Pipeline([
			('st', ToSimplexTreeMulti), 
			("sm",ToSignedMeasure),
			('smf',SMF), 
			("smd", SMD), 
			("smdl",D2DL),
		])
		X = SignedMeasureDistancesTransformer.fit_transform(X=X)
		print(f"Num axes of computed measure : {len(X[0])}")
		pipeline = Pipeline([
			("DL2D",DL2D),
			("DM2K",DM2K),
			("SVMP",SVMP),
		])
		parameters = {}
		parameters.update(DM2K_parameters)
		parameters.update(SVMP_parameters)
	case "dr"|"degreerips":
		### Preprocessing: compute the signed measure
		raise Exception("TODO reimplement")
		# X = DR2SM.fit_transform(X)

		### CLASSIFICATION PIPELINE
		pipeline = Pipeline([
			("SMM2CV", SMM2CV),
			("final_classifier", final_classifier)
		])
		parameters = {}
		parameters.update(SMM2CV_parameters)
		parameters.update(final_classifier_parameters)
	case "sw"|"slicedwasserstein":
		svm = SVC(kernel = "precomputed")
		print("Computing Sliced Wassertstein Distances", flush=True)
		diagrams = compute_diagram_pipe.fit_transform(X)
		swds = mmo.Dgms2SlicedWassersteinDistanceMatrices(num_directions=10, n_jobs=args.n_jobs).fit_transform(diagrams)
		print("Formatting Distance Matrix", flush=True)
		X = mmo.DistanceMatrices2DistancesList().fit_transform(swds)
		pipeline = Pipeline([
			("dms",DL2D),
			("DM2K", DM2K),
			("SVMP",SVMP)
		])
		parameters = {}
		DM2K_parameters.pop("DM2K__axis")
		DM2K_parameters.pop("DM2K__weights")
		parameters.update(DM2K_parameters)
		parameters.update(SVMP_parameters)
# elif args.pipeline == "sw_p": # Graph -> SimplexTree -> Diagram -> Shuffled Diagram -> SW -> SVM
# 	svm = SVC(kernel = "precomputed")
# 	params={
# 		"sw__bandwidth":[0.01, 0.1, 1, 10, 100],
# 		"sw__num_directions":[10],
# 		"svm__kernel" : ["precomputed"],
# 		"svm__C" : [0.01,1, 10, 100, 1000],
# 	}
# 	pipe = Pipeline([
# 		("shuffle", DiagramShuffle()),
# 		("sw", Dgms2SWK()),
# 		("svm",svm)
# 	])
	case "pervec":
		X = compute_diagram_pipe.fit_transform(X)
		svm = SVC(kernel="rbf")
		parameters = {
			"hist__quantile":[0.],
			"hist__bins":[100,200,300],
			"svm__kernel" : ["rbf"],
			"svm__gamma" : [0.01, 0.1, 1, 10, 100],
			"svm__C" : [0.001,0.01,1, 10, 100, 1000],
		}
		pipeline = Pipeline([
			("hist", mmo.Dgm2Histogram()),
			("svm",svm)
		])
	case "onesmk":
		diagrams = compute_diagram_pipe.fit_transform(X)
		# print(diagrams)
		smds = mmo.Dgms2SignedMeasureDistance(n_jobs=args.n_jobs, progress=True).fit_transform(diagrams)
		# print(smds)
		X = mmo.DistanceMatrices2DistancesList().fit_transform(smds)
		pipeline = Pipeline([
			("dms",DL2D),
			("DM2K", DM2K),
			("SVMP",SVMP)
		])
		parameters = {}
		DM2K_parameters.pop("DM2K__axis")
		DM2K_parameters.pop("DM2K__weights")
		parameters.update(DM2K_parameters)
		parameters.update(SVMP_parameters)
	case "onesmh":
		X = compute_diagram_pipe.fit_transform(X)
		hist = mmo.Dgms2SignedMeasureHistogram()
		svm = SVC(kernel="rbf")
		pipeline = Pipeline([
			("hist", hist),
			("svm",svm),
		])
		parameters = {
			"hist__quantile" : [0,0.01,0.1],
			"hist__bins":[50,100,200,300],
			"svm__kernel" : ["rbf"],
			"svm__gamma" : [0.01, 0.1, 1, 10, 100],
			"svm__C" : [0.001,0.01,1, 10, 100, 1000],
		}
	case "onesmi":
		X = compute_diagram_pipe.fit_transform(X)
		img = mmo.Dgms2SignedMeasureImage()
		svm = SVC(kernel="rbf")
		pipeline = Pipeline([
			("img", img),
			("svm",svm),
		])
		parameters = {
			"img__quantile" : [0,0.01,0.1],
			"img__bandwidth" : [0.01, 0.1, 1., 10., 100.],
			"img__resolution":[50,100,200,300],
			"svm__kernel" : ["rbf"],
			"svm__gamma" : [0.01, 0.1, 1, 10, 100],
			"svm__C" : [0.001,0.01,1, 10, 100, 1000],
		}
	case "onepl":
		X = compute_diagram_pipe.fit_transform(X)
		pipeline = Pipeline([("pl", mmo.Dgms2Landscapes()), ("svm", SVC(kernel="rbf"))])
		parameters = {
			"svm__kernel" : ["rbf"],
			"svm__gamma" : [0.01, 0.1, 1, 10, 100],
			"svm__C" : [0.001,0.01,1, 10, 100, 1000],
			"pl__num": [3,4,5,6,7,8], #num landscapes
			"pl__resolution": [50,100,200,300],
		}
# elif args.pipeline == "pl_p": # Shuffled Landscapes
# 	X = compute_diagram_pipe.fit_transform(X)
# 	pipeline = Pipeline([("shuffle", mmo.DiagramShuffle()),("pl", mmo.Dgms2Landscapes()), ("svm", SVC(kernel="rbf"))])
# 	parameters = {
# 		"svm__kernel" : ["rbf"],
# 		"svm__gamma" : [0.01, 0.1, 1, 10, 100],
# 		"svm__C" : [0.001,0.01,1, 10, 100, 1000],
# 		"pl__num": [3,4,5,6,7,8], #num landscapes
# 		"pl__resolution": [50,100,200,300],
# 	}
	case "onepi":
		X = compute_diagram_pipe.fit_transform(X)
		pipeline = Pipeline([("pi", mmo.Dgms2Image()), ("svm", SVC(kernel="rbf"))])
		parameters = {
			"svm__kernel" : ["rbf"],
			"svm__gamma" : [0.01, 0.1, 1, 10, 100],
			"svm__C" : [0.001,0.01,1, 10, 100, 1000],
			"pi__bandwidth": [0.01,0.1,1,10,100], 
			"pi__resolution": [[20,20], [30,30]],
		}
# elif args.pipeline == "pi_p": # Shuffled Immages
# 	X = compute_diagram_pipe.fit_transform(X)
# 	pipeline = Pipeline([("shuffle", mmo.DiagramShuffle()),("pi", mmo.Dgms2Image()), ("svm", SVC(kernel="rbf"))])
# 	parameters = {
# 		"svm__kernel" : ["rbf"],
# 		"svm__gamma" : [0.01, 0.1, 1, 10, 100],
# 		"svm__C" : [0.001,0.01,1, 10, 100, 1000],
# 		"pi__bandwidth": [0.01,0.1,1,10,100], 
# 		"pi__resolution": [[20,20], [30,30]],
# 	}	
	case unimplemented:
		raise Exception(f"Pipeline {unimplemented} not supported.")

print("Initializing classification pipeline", flush=True)
classifier = GridSearchCV(
	estimator=pipeline, param_grid=parameters,
	n_jobs = args.n_jobs, cv=args.train_k, verbose=10)

######################################SCORE
print("Computing classification, with pipeline", flush=True)
print(pipeline, flush=True)
print("Final parameters : ", parameters)


# try: 
# 	filtration_grid = ToSignedMeasure.filtration_grid
# 	print("Signed Measure Filtration grid : ", filtration_grid)
# except:
# 	None

# PRECOMPILES PYKEOPS if necessary
k=3
print(f"------------ Running {k} times small fit... ")
for _ in range(k):
	example_parameter = {a:choice(b) for a,b in parameters.items()}
	pipeline_ = clone(pipeline).set_params(**example_parameter)
	SMF.verbose=True
	print("Parameters", example_parameter)
	_X = deepcopy(X[:3])

	assert len(_X) == 3, len(_X)
	_Y = [0,1,0] ## svm needs at least 2 classes
	pipeline_.fit(_X, _Y).score(_X, _Y)
SMF.verbose = False
print("------------ Done.", flush=True)
if args.test:
	print("------------ Done testing.")
	exit()


## mma modules are hard to pickle. Signed measures are trivial to pickle.
## internal pipelines are threading hardcoded.
#backend = "loky" if "mma" not in args.pipeline else "threading"
# backend="loky" # loky seems to be faster anyway...
# with parallel_backend(backend, n_jobs=args.n_jobs):
accuracy_to_csv(
	X=X, Y=Y, dataset = dataset, cl=classifier, k=args.test_k,
	shuffle = shuffle,
	**results_kwargs
)
# os.system(f"rm -rf {memory}") # removes cache
