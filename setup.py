import contextlib
import os
import site

import numpy as np
import sklearn._build_utils
from Cython.Build import cythonize
from Cython.Compiler import Options
from setuptools import Extension, find_packages, setup

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
    # "diff_helper",
    # "hilbert_function",
    # "euler_characteristic",
    # 'cubical_multi_complex',
    "point_measure_integration",
    "grids",
    "slicer",
]


templated_cython_modules = [
    "filtration_conversions.pxd",
    "slicer.pxd",
    "mma_structures.pyx",
    "simplex_tree_multi.pyx",
    "slicer.pyx",
]


os.system(
    "python _tempita_grid_gen.py"
)  ## generates some parameter files (Tempita fails with python<3.12)
sklearn._build_utils.gen_from_templates(
    (f"multipers/{mod}.tp" for mod in templated_cython_modules)
)


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
    "embedsignature.format": "python",
    "binding": True,
    "infer_types": True,
    "boundscheck": False,
    "wraparound": True,
    "iterable_coroutine": True,
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

build_dependencies = [
    "gudhi",
    "numpy",
    "Cython",  # needed for compilation
    # "scikit-learn",
    # "tbb",  # needed for compilation, but doesn't exist on mac
    # "tbb-devel", # needed for compilation
    # boost,
    # boost-cpp,
    # "tqdm",
    "scikit-learn",
    "setuptools",
    # "joblib",
]
python_dependencies = [
    "gudhi",
    "numpy",
    "filtration-domination",
    "pykeops",
    "scikit-learn",
    "joblib",
    "pot",
    "tqdm",
    "matplotlib",
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
            "-std=c++20",  # Windows doesn't support this yet. TODO: Wait.
            # "-fno-aligned-new", # Uncomment this if you have trouble compiling on macos.
            "-Wall",
        ],
        extra_link_args=[],  ## mvec for python312
        include_dirs=cpp_dirs,
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
        libraries=["tbb", "m"],
        library_dirs=library_dirs,
    )
    for module in cython_modules
]

if __name__ == "__main__":
    # os.system("rm *.pkl")
    setup(
        name="multipers",
        author="David Loiseaux",
        author_email="david.loiseaux@inria.fr",
        description="Scikit-style Multiparameter persistence toolkit",
        url="https://github.com/DavidLapous/multipers",
        # long_description=long_description,
        # long_description_content_type='text/markdown'
        version="2.0.3",
        license="MIT",
        keywords="TDA Persistence Multiparameter sklearn",
        ext_modules=cythonize(
            extensions,
            compiler_directives=cython_compiler_directives,
            **cythonize_flags,
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
        install_requires=python_dependencies,
        setup_requires=build_dependencies,
    )
    # os.system("rm *.pkl")
