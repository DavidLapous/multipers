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
from setuptools import Extension, setup
from Cython.Build import cythonize
import sys as _sys

extensions = [Extension('mma',
						sources=['custom_vineyards.pyx'],
						language='c++',
						extra_compile_args=[
							"-O3",
							"-march=native",
							"-g0",
							"-std=c++17"
						  ,'-fopenmp'
						  ,"-pthread"
						  ],
						extra_link_args=['-fopenmp', '-pthread'],
)]
setup(
	name='mma',
	author="David Loiseaux, Mathieu Carrière",
	author_email="david.loiseaux@inria.fr",
	url="https://gitlab.inria.fr/dloiseau/multipers",
	description="Open source library for multipersistence module approximation.",
	ext_modules=cythonize(extensions, language_level = str(_sys.version_info[0])), include_dirs=['.']
	)
