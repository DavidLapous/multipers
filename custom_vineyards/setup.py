#from distutils.core import setup
#from distutils.extension import Extension
from setuptools import Extension, setup
from Cython.Build import cythonize
extensions = [Extension('custom_vineyards',
						sources=['custom_vineyards.pyx'],
						language='c++',
						extra_compile_args=[
							"-O3",
							"-march=native",
							"-g0",
							"-std=c++17"
						  #,'-fopenmp'
						  ],
						#extra_link_args=['-fopenmp'],
)]
setup(name='custom_vineyards', ext_modules=cythonize(extensions, language_level = "3"), include_dirs=['.'])
