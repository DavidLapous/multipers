# import contextlib
import filecmp
import os
import shutil
import platform
import sys

from pathlib import Path
import numpy as np
from Cython import Tempita
from Cython.Build import cythonize
from Cython.Compiler import Options
from setuptools import Extension, setup

Options.docstrings = True
Options.embed_pos_in_docstring = True
Options.fast_fail = True
# Options.warning_errors = True

try:
    os.mkdir("build")
except FileExistsError:
    pass

try:
    os.mkdir("build/tmp")
except FileExistsError:
    pass


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
PYTHON_ENV_PATH = sys.prefix

cpp_dirs = [
    "multipers/gudhi",
    "multipers",
    # "multipers/multiparameter_module_approximation",
    # "multipers/multi_parameter_rank_invariant",
    # "multipers/tensor",
    np.get_include(),
    PYTHON_ENV_PATH + "/include/", # Unix
    PYTHON_ENV_PATH + "/Library/include/", # Windows
]
cpp_dirs = [str(Path(stuff).expanduser().resolve()) for stuff in cpp_dirs]

library_dirs = [
    PYTHON_ENV_PATH + "/lib/", # Unix
    PYTHON_ENV_PATH + "/Library/lib/", # Windows
]

library_dirs = [str(Path(stuff).expanduser().resolve()) for stuff in library_dirs]

print("Include dirs:")
print(cpp_dirs)

print("Library dirs:")
print(library_dirs)

extensions = [
    Extension(
        f"multipers.{module}",
        sources=[
            f"multipers/{module}.pyx",
        ],
        language="c++",
        extra_compile_args=[
            "-O3",  # -Ofast disables infinity values for filtration values
            "-fassociative-math",
            "-funsafe-math-optimizations",
            # "-g",
            # "-march=native",
            "/std:c++20" if platform.system() == "Windows" else "-std=c++20",
            # "-fno-aligned-new", # Uncomment this if you have trouble compiling on macos.
            "-Wall",
            "-Wextra" if platform.system() != "Windows" else "",
            # "-Werror" if platform.system() != "Windows" else "",
        ],
        extra_link_args=[],  ## mvec for python312
        include_dirs=cpp_dirs,
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
        libraries=["tbb"],
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
