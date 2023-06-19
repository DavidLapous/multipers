
from os.path import exists
from setuptools import Extension, setup, find_packages
import numpy as np
from Cython.Build import cythonize
from Cython.Compiler import Options

Options.docstrings = True
Options.embed_pos_in_docstring = True
Options.fast_fail = True

cython_compiler_directives = {
    "language_level": 3,
    "embedsignature": True,
    "binding": True,
    "infer_types": True,
    # "show_all_warnings": True,
    # "nthreads":max(1, cpu_count()//2),
}

cythonize_flags = {
    # "depfile":True,
    # "nthreads": (int)((cpu_count()//2) + 1),
    # "show_all_warnings":True,
}

cython_modules = ["simplex_tree_multi", "multiparameter_module_approximation"]
extensions = [Extension(f"mma.{module}",
                        sources=[f"mma/{module}.pyx"],
                        language='c++',
                        extra_compile_args=[
                            "-Ofast",
                            "-march=native",
                            "-std=c++20",
                            '-ltbb',
                            "-Wall",
                        ],
                        extra_link_args=['-ltbb'],
                        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
                        ) for module in cython_modules]
setup(
    name='mma',
    author="David Loiseaux",
    author_email="david.loiseaux@inria.fr",
    description="Multiparameter Module Approximation",
	ext_modules=cythonize(
		extensions, compiler_directives=cython_compiler_directives, **cythonize_flags),
	packages=find_packages(include=['mma', "mma.*"]),
	package_data={"mma":["*.pyi"]},
	python_requires=">=3.10",
	include_dirs = ['mma', "mma/gudhi", np.get_include()],
)
