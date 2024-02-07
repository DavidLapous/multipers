import contextlib
import site
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
    "io",
    "rank_invariant",
    "function_rips",
    "mma_structures",
    "multiparameter_module_approximation",
    "hilbert_function",
    "euler_characteristic",
    # 'cubical_multi_complex',
    "point_measure_integration",
    "slicer",
]
n_jobs = 1
with contextlib.suppress(ImportError):
    import joblib

    n_jobs = joblib.cpu_count()

cythonize_flags = {
    # "depfile":True,
    # "nthreads": n_jobs,  # Broken on mac
    # "show_all_warnings":True,
}

cython_compiler_directives = {
    "language_level": 3,
    "embedsignature": True,
    "binding": True,
    "infer_types": True,
    "boundscheck": False,
    "wraparound": True,
    # "profile":True,
    # "unraisable_tracebacks":True,
    "annotation_typing": True,
    "emit_code_comments": True,
    "initializedcheck": False,
    # "nonecheck": False,
    "cdivision": True,
    "profile": False,
}

# When venv is not properly set, we have to add the current python path
# removes lib / python3.x / site-packages
PYTHON_ENV_PATH = "/".join((site.getsitepackages()[0]).split("/")[:-3])
INCLUDE_PATH = PYTHON_ENV_PATH + "/include/"
LIBRARY_PATH = PYTHON_ENV_PATH + "/lib/"

cpp_dirs = [
    "multipers/gudhi",
    "multipers",
    np.get_include(),
    INCLUDE_PATH,
]
library_dirs = [
    LIBRARY_PATH,
]

python_dependencies = [
    "gudhi",  # Waiting for gudhi with python 3.12
    "numpy",
    "Cython",  # needed for compilation
    # "scikit-learn",
    "tbb",  # needed for compilation
    # "tbb-devel", # needed for compilation
    # boost,
    # boost-cpp,
    # "tqdm",
    "setuptools",
]


extensions = [
    Extension(
        f"multipers.{module}",
        sources=[
            f"multipers/{module}.pyx",
        ],
        language="c++",
        extra_compile_args=[
            "-Ofast",
            # "-g",
            # "-march=native",
            # Windows doesn't support this yet. TODO: Wait (haha).
            "-std=c++20",
            # "-fno-aligned-new", # Uncomment this if you have trouble compiling on macos.
            "-Wall",
        ],
        include_dirs=cpp_dirs,
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
        libraries=["tbb", "tbbmalloc"],
        library_dirs=library_dirs,
    )
    for module in cython_modules
]
setup(
    name="multipers",
    author="David Loiseaux",
    author_email="david.loiseaux@inria.fr",
    description="Scikit-style Multiparameter persistence toolkit",
    url="https://github.com/DavidLapous/multipers",
    # long_description=long_description,
    # long_description_content_type='text/markdown'
    version="1.2.0",
    license="MIT",
    keywords="TDA Persistence Multiparameter sklearn",
    ext_modules=cythonize(
        extensions, compiler_directives=cython_compiler_directives, **cythonize_flags
    ),
    packages=find_packages(),
    package_data={
        "multipers": ["*.pyi", "*.pyx", "*.pxd"],
    },
    python_requires=">=3.10",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: Implementation :: CPython",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Visualization",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: MIT License",
    ],
)
