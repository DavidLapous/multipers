import os
import platform
import subprocess
import sys
import filecmp
import shutil

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

os.makedirs("build/tmp", exist_ok=True)


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
    print("#-----------------------------------")
    print(f"processing {fromfile}.")
    print("#-----------------------------------")
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
    "vector_interface",
]


def arch_module_blacklist(module: str):
    if platform.system() == "Windows" and module == "vector_interface":
        # Persistence-Algebra doesn't compile yet here
        return False
    return True


templated_cython_modules = [
    "filtrations.pxd",
    "filtration_conversions.pxd",
    "slicer.pxd",
    "mma_structures.pyx",
    "simplex_tree_multi.pyx",
    "slicer.pyx",
]

# generates some parameter files (Tempita fails with python<3.12)
# TODO: see if there is a way to avoid _tempita_grid_gen.py or a nicer way to do it
subprocess.run([sys.executable, "_tempita_grid_gen.py"], check=True)

for mod in templated_cython_modules:
    process_tempita(f"multipers/{mod}.tp")


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
PYTHON_ENV_PATH = Path(sys.prefix)

cpp_dirs = [
    "multipers/gudhi",
    "multipers",
    # "multipers/multiparameter_module_approximation",
    # "multipers/multi_parameter_rank_invariant",
    # "multipers/tensor",
    np.get_include(),
    PYTHON_ENV_PATH / "include",  # Unix
    PYTHON_ENV_PATH / "Library" / "include",  # Windows
    "AIDA/src",
    "AIDA/include",
    "Persistence-Algebra/include",
]
cpp_dirs = [str(Path(stuff).expanduser().resolve()) for stuff in cpp_dirs]

library_dirs = [
    PYTHON_ENV_PATH / "lib",  # Unix
    PYTHON_ENV_PATH / "Library" / "lib",  # Windows
]

library_dirs = [str(Path(stuff).expanduser().resolve()) for stuff in library_dirs]


## AIDA stuff

AIDA_PATHS = [
    Path("AIDA/src"),
    Path("AIDA/include"),
]

# Recursively collect all .cpp files from the AIDA directories
AIDA_CPP_SOURCES = []
for p in AIDA_PATHS:
    # Use Path.rglob('*.cpp') to recursively find all .cpp files
    AIDA_CPP_SOURCES.extend([str(file) for file in p.rglob("*.cpp")])

print("AIDA files:")
print(AIDA_CPP_SOURCES)

print("Include dirs:")
print(cpp_dirs)

print("Library dirs:")
print(library_dirs)


def cpp_lib_deps(module):
    if module == "vector_interface":
        return ["boost_system", "boost_timer"]
    else:
        return ["tbb"]


extensions = [
    Extension(
        f"multipers.{module}",
        sources=(
            [
                f"multipers/{module}.pyx",
            ]
            + (AIDA_CPP_SOURCES if module == "vector_interface" else [])
        ),
        language="c++",
        extra_compile_args=[
            "-O3"
            if platform.system() != "Windows"
            else "/O2",  # -Ofast disables infinity values for filtration values
            "-fassociative-math",
            "-funsafe-math-optimizations",
            "-DGUDHI_USE_TBB",
            "-DWITH_TBB=ON",
            # "-g",
            # "-march=native",
            "-DNDEBUG" if platform.system() != "Windows" else "/DNDEBUG",
            "-std=c++20" if platform.system() != "Windows" else "/std:c++20",
            # "-fno-aligned-new", # Uncomment this if you have trouble compiling on macos.
            "-Wall",
            "-Wextra" if platform.system() != "Windows" else "",
            # "-Werror" if platform.system() != "Windows" else "",
        ],
        extra_link_args=[],
        include_dirs=cpp_dirs,
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
        libraries=cpp_lib_deps(module),
        library_dirs=library_dirs,
    )
    for module in cython_modules
    if arch_module_blacklist(module)
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
