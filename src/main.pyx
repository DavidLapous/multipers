"""!
@package mma
@brief Files containing the C++ cythonized functions.
@author David Loiseaux, Mathieu Carrière, Hannah Schreiber
@copyright Copyright (c) 2022 Inria.
"""

# distutils: language = c++
# distutils: include_dirs = mma_cpp

###########################################################################
#PYTHON LIBRARIES
import gudhi as gd
from gudhi.simplex_tree import SimplexTree as GudhiSimplexTree
import numpy as np
from typing import List, Union
from os.path import exists
from os import remove 
from tqdm import tqdm 
from cycler import cycler
from joblib import Parallel, delayed
import pickle as pk

###########################################################################
#CPP CLASSES
from cython.operator import dereference, preincrement
from libc.stdint cimport intptr_t
from libc.stdint cimport uintptr_t

###########################################################################
#CYTHON TYPES
from cython cimport numeric
from libcpp.vector cimport vector
from libcpp.utility cimport pair
#from libcpp.list cimport list as clist
from libcpp cimport bool
from libcpp cimport int
from libcpp.string cimport string


###########################################################################
#PYX MODULES
include "mma.pyx"
include "format_conversions.pyx"
include "matching_distance.pyx"
include "simplex_tree_multi.pyx"

include "plots.pyx"
include "tests.pyx"



