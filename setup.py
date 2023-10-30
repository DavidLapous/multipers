
from setuptools import Extension, setup, find_packages
import numpy as np
from Cython.Build import cythonize
from Cython.Compiler import Options


Options.docstrings = True
Options.embed_pos_in_docstring = True
Options.fast_fail = True
# Options.warning_errors = True

cython_modules = [
    "simplex_tree_multi", 
    "rank_invariant",
	"function_rips",
    "multiparameter_module_approximation", 
    # 'diff_helper',
	'hilbert_function',
	'euler_characteristic',
    # 'cubical_multi_complex',
	'point_measure_integration',
]

cythonize_flags = {
	# "depfile":True,
	# "nthreads": len(cython_modules), # Broken on mac
	# "show_all_warnings":True,
}

cython_compiler_directives = {
	"language_level": 3,
	"embedsignature": True,
	"binding": True,
	"infer_types": True,
	"boundscheck":False,
	"wraparound":True,
	# "profile":True,
	# "unraisable_tracebacks":True,
	"annotation_typing":True,
	"emit_code_comments":True,
}


extensions = [Extension(f"multipers.{module}",
		sources=[f"multipers/{module}.pyx"],
		language='c++',
		extra_compile_args=[
			"-Ofast",
			#"-march=native",
			"-std=c++20",
            # Uncomment this if you have trouble compiling on macos.
			# "-fno-aligned-new", 
			"-Wall",
		],
		define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
		libraries=["tbb", "tbbmalloc"]
	) for module in cython_modules
]
setup(
	name='multipers',
	author="David Loiseaux",
	author_email="david.loiseaux@inria.fr",
	description="Multiparameter persistence toolkit",
	ext_modules=cythonize(
		extensions, compiler_directives=cython_compiler_directives, **cythonize_flags),
	packages=find_packages(include=['multipers', "multipers.*"]),
	package_data={"multipers":["*.pyi", "*.pyx", "*.pxd"]},
	python_requires=">=3.10",
	include_dirs = ['multipers', np.get_include(), 'multipers/gudhi'],
)
