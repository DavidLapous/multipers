# import contextlib
import filecmp
import os
import shutil
import site

import numpy as np
from Cython import Tempita
from Cython.Build import cythonize
from Cython.Compiler import Options
from setuptools import Extension, setup

Options.docstrings = True
Options.embed_pos_in_docstring = True
Options.fast_fail = True
# Options.warning_errors = True

os.system("mkdir -p ./build/tmp")


def was_modified(file):
    tail = os.path.basename(file)
    new_file = "build/tmp/" + tail
    if not os.path.isfile(new_file):
        print(f"File {file} was modified.")
        shutil.copyfile(file, new_file)
        return True
    else:
        res = not filecmp.cmp(new_file, file, shallow=False)
        if res:
            print(f"File {file} was modified.")
            shutil.copyfile(file, new_file)
        return res


full_build = False
if was_modified("_tempita_grid_gen.py"):
    full_build = True


# credit to sklearn with just a few modifications:
# https://github.com/scikit-learn/scikit-learn/blob/156ef1b7fe9bc0ee5b281634cfd56b9c54e83277/sklearn/_build_utils/tempita.py
# took it out to not having to depend on a sklearn version in addition to a cython version
def process_tempita(fromfile):
    """Process tempita templated file and write out the result.

    The template file is expected to end in `.c.tp` or `.pyx.tp`:
    E.g. processing `template.c.tp` generates `template.c`.

    """
    if not was_modified(fromfile) and not full_build:
        return
    with open(fromfile, "r", encoding="utf-8") as f:
        template_content = f.read()

    template = Tempita.Template(template_content)
    content = template.substitute()

    outfile = os.path.splitext(fromfile)[0]
    with open(outfile, "w", encoding="utf-8") as f:
        f.write(content)


cython_modules = [
    "simplex_tree_multi",
    "io",
    # "rank_invariant",
    "function_rips",
    "mma_structures",
    "multiparameter_module_approximation",
    "point_measure",
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

## generates some parameter files (Tempita fails with python<3.12)
# TODO: see if there is a way to avoid _tempita_grid_gen.py or a nicer way to do it
os.system("python _tempita_grid_gen.py")

for mod in templated_cython_modules:
    process_tempita(f"multipers/{mod}.tp")

## Broken on mac
# n_jobs = 1
# with contextlib.suppress(ImportError):
#     import joblib
#     n_jobs = joblib.cpu_count()

cythonize_flags = {
    # "depfile": True,
    # "nthreads": n_jobs,  # Broken on mac
    # "show_all_warnings": True,
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
    "multipers/multiparameter_module_approximation",
    "multipers/multi_parameter_rank_invariant",
    "multipers/tensor",
    np.get_include(),
    INCLUDE_PATH,
]

library_dirs = [
    LIBRARY_PATH,
]

extensions = [
    Extension(
        f"multipers.{module}",
        sources=[
            f"multipers/{module}.pyx",
        ],
        language="c++",
        extra_compile_args=[
            "-O3",  # -Ofast disables infinity values for filtration values
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
    setup(
        name="multipers",
        ext_modules=cythonize(
            extensions,
            compiler_directives=cython_compiler_directives,
            **cythonize_flags,
        ),
    )
