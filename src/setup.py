"""
Multipersistence Module Approximation Library cython file.

	Author(s):       David Loiseaux, Mathieu Carrière
	Copyright (C) 2022  Inria
"""


__author__ = "David Loiseaux, Mathieu Carrière"
__copyright__ = "Copyright (C) 2022  Inria"
__license__ = ""

#from distutils.core import setup
#from distutils.extension import Extension
from setuptools import Extension, setup, find_packages
from Cython.Build import cythonize

import sys as sys
from multiprocessing import cpu_count

py_modules=[]
# curl micro.mamba.pm/install.sh | bash # Installs micromamba
# micromamba create -n python310
# micromamba activate python310
# micromamba install python=3.10 numpy matplotlib gudhi scikit-learn cython sympy tqdm cycler typing shapely
# pip install filtration-domination
python_requirements = [
#	"numpy",
#	"matplotlib",
#	"gudhi",
#	"scikit-learn",
	"cython",
#	"sympy",
#	"tqdm",
#	"cycler",
#	"typing",
#	"shapely",
#	"filtration-domination",
]

cython_compiler_directives = {
	"language_level":3,
	"embedsignature":True,
	"binding":True,
	"infer_types":True,
	# "nthreads":max(1, cpu_count()//2),
}
cython_flags= {
	# "annotate":True, # This prevents compilation in parallel
	# "language":"c++", # DEPRECATED
	#"annotate-fullc":True,
	#"depfile":True,
	"nthreads":(int)(max(1, cpu_count()//2)),
	# "show_all_warnings":True,
}

extensions = [Extension('mma',
						sources=['main.pyx',],
						language='c++',
						extra_compile_args=[
							"-Ofast",
							"-march=native",
							#"-g0",
							"-std=c++20",
							'-fopenmp',
							"-Wall"
						],
						extra_link_args=['-fopenmp'],
)]
setup(
	name='mma',
	author="David Loiseaux, Mathieu Carrière",
	author_email="david.loiseaux@inria.fr",
	url="https://gitlab.inria.fr/dloiseau/multipers",
	description="Open source library for multipersistence module approximation.",
	install_requires=python_requirements,
	# packages=(find_packages(where=".")),
	ext_modules=cythonize(extensions, compiler_directives=cython_compiler_directives,**cython_flags),
	include_dirs=['mma_cpp', "gudhi"],
	)
