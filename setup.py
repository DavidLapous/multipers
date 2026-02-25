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

IS_WINDOWS = platform.system() == "Windows"


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
    "ops",
    "ext_interface._mpfree_interface",
    "ext_interface._aida_interface",
    "ext_interface._function_delaunay_interface",
    "ext_interface._multi_critical_interface",
]


def should_build_module(module: str):
    if IS_WINDOWS and module in [
        "ext_interface._aida_interface",
    ]:
        # Persistence-Algebra and AIDA extension do not compile on Windows.
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

base_cpp_dirs = [
    "multipers/gudhi",
    "multipers",
    np.get_include(),
    PYTHON_ENV_PATH / "include",  # Unix
    PYTHON_ENV_PATH / "include" / "eigen3",  # Eigen
    PYTHON_ENV_PATH / "Library" / "include",  # Windows
    PYTHON_ENV_PATH / "Library" / "include" / "eigen3",  # Windows
    # "/usr/include/",
    # "/usr/local/include",
]
base_cpp_dirs = [str(Path(stuff).expanduser().resolve()) for stuff in base_cpp_dirs]

# Include directories follow each backend's own CMake include layout.
ext_cpp_dirs = {
    "aida": [
        "ext/AIDA/src",
        "ext/AIDA/include",
        "ext/Persistence-Algebra/include",
    ],
    "mpfree": [
        "ext/mpfree/include",
        "ext/mpfree/phat_mod/include",
        "ext/mpfree/mpp_utils_mod/include",
        "ext/mpfree/scc_mod/include",
    ],
    "multi_critical": [
        # Keep SCC include first for this backend.
        "ext/multi_critical/scc_mod/include",
        "ext/multi_critical/include",
        "ext/multi_critical/phat_mod/include",
        "ext/multi_critical/mpp_utils_mod/include",
        "ext/multi_critical/mpfree_mod/include",
        "ext/multi_critical/multi_chunk_mod/include",
    ],
    "function_delaunay": [
        "ext/function_delaunay/include",
        "ext/function_delaunay/phat/include",
        "ext/function_delaunay/mpp_utils_mod/include",
        "ext/function_delaunay/scc_mod/include",
        "ext/function_delaunay/mpfree_mod/include",
        "ext/function_delaunay/multi_chunk_mod/include",
    ],
}
ext_cpp_dirs = {
    key: [str(Path(stuff).expanduser().resolve()) for stuff in values]
    for key, values in ext_cpp_dirs.items()
}


def module_cpp_dirs(module: str):
    dirs = list(base_cpp_dirs)
    if module == "ext_interface._aida_interface":
        dirs.extend(ext_cpp_dirs["aida"])
    elif module == "ext_interface._mpfree_interface":
        dirs.extend(ext_cpp_dirs["mpfree"])
    elif module == "ext_interface._multi_critical_interface":
        dirs.extend(ext_cpp_dirs["multi_critical"])
    elif module == "ext_interface._function_delaunay_interface":
        dirs.extend(ext_cpp_dirs["function_delaunay"])
    return dirs


ext_interface_disable_macros = {
    "ext_interface._aida_interface": "MULTIPERS_DISABLE_AIDA_INTERFACE",
    "ext_interface._mpfree_interface": "MULTIPERS_DISABLE_MPFREE_INTERFACE",
    "ext_interface._function_delaunay_interface": "MULTIPERS_DISABLE_FUNCTION_DELAUNAY_INTERFACE",
    "ext_interface._multi_critical_interface": "MULTIPERS_DISABLE_MULTI_CRITICAL_INTERFACE",
}


def module_define_macros(module: str):
    macros: list[tuple[str, str | None]] = [
        ("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")
    ]
    disable_macro = ext_interface_disable_macros.get(module)
    if IS_WINDOWS and disable_macro is not None:
        macros.append((disable_macro, "1"))
    return macros


library_dirs = [
    PYTHON_ENV_PATH / "lib",  # Unix
    PYTHON_ENV_PATH / "Library" / "lib",  # Windows
]

library_dirs = [str(Path(stuff).expanduser().resolve()) for stuff in library_dirs]


## AIDA stuff


def build_aida_static_library():
    source_dir = Path("ext/AIDA").resolve()
    build_dir = Path("build") / "aida"
    cache_file = build_dir / "CMakeCache.txt"
    if cache_file.exists():
        cache_text = cache_file.read_text(encoding="utf-8", errors="ignore")
        if str(source_dir) not in cache_text:
            shutil.rmtree(build_dir)
    build_dir.mkdir(parents=True, exist_ok=True)

    configure_cmd = [
        "cmake",
        "-S",
        str(source_dir),
        "-B",
        str(build_dir),
        "-DCMAKE_BUILD_TYPE=Release",
        "-DCMAKE_POSITION_INDEPENDENT_CODE=ON",
    ]
    build_cmd = [
        "cmake",
        "--build",
        str(build_dir),
        "--target",
        "aida_static",
        "--config",
        "Release",
        "-j4",
    ]

    print("Configuring AIDA static library")
    subprocess.run(configure_cmd, check=True)

    print("Building AIDA static library")
    subprocess.run(build_cmd, check=True)

    candidates = [
        build_dir / "libaida.a",
        build_dir / "aida.lib",
        build_dir / "Release" / "aida.lib",
        build_dir / "Release" / "libaida.a",
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate.resolve())

    # Fallback for generators that place the artifact in nested directories
    fallback_candidates = list(build_dir.rglob("libaida.a")) + list(
        build_dir.rglob("aida.lib")
    )
    if fallback_candidates:
        return str(fallback_candidates[0].resolve())

    raise RuntimeError("Could not locate built AIDA static library in build/aida.")


AIDA_STATIC_LIBRARY = None
requested_commands = set(sys.argv[1:])
build_commands = {
    "build_ext",
    "install",
    "bdist",
    "bdist_wheel",
    "develop",
    "editable_wheel",
}
should_build_aida = any(cmd in requested_commands for cmd in build_commands)
if should_build_module("ext_interface._aida_interface") and should_build_aida and not IS_WINDOWS:
    AIDA_STATIC_LIBRARY = build_aida_static_library()
    print("AIDA static library:")
    print(AIDA_STATIC_LIBRARY)

print("Base include dirs:")
print(base_cpp_dirs)

print("Library dirs:")
print(library_dirs)


def cpp_lib_deps(module):
    if module.startswith("ext_interface._"):
        return ["boost_system", "boost_timer", "boost_chrono", "omp", "gmp"]
    elif module in ["ops", "io"]:
        return []
    else:
        return ["tbb"]


extensions = [
    Extension(
        f"multipers.{module}",
        sources=(
            [
                f"multipers/{module.replace('.', '/')}.pyx",
            ]
        ),
        language="c++",
        extra_compile_args=(
            [
                "-DGUDHI_USE_TBB",
                "-DWITH_TBB=ON",
                # "-g",
                # "-march=native",
                # "-fno-aligned-new", # Uncomment this if you have trouble compiling on macos.
                # "-Werror",
            ]
            + (
                [
                    "/O2",
                    "/DNDEBUG",
                    "/std:c++20",
                    "/W1",
                    "/WX-",
                ]
                if IS_WINDOWS
                else [
                    "-O3",  # -Ofast disables infinity values for filtration values
                    "-fassociative-math",
                    "-funsafe-math-optimizations",
                    "-DNDEBUG",
                    "-std=c++20",
                    "-Wall",
                    "-Wextra",
                ]
            )
        ),
        extra_link_args=[],
        extra_objects=[AIDA_STATIC_LIBRARY]
        if module == "ext_interface._aida_interface" and AIDA_STATIC_LIBRARY
        else [],
        include_dirs=module_cpp_dirs(module),
        define_macros=module_define_macros(module),
        libraries=cpp_lib_deps(module),
        library_dirs=library_dirs,
    )
    for module in cython_modules
    if should_build_module(module)
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
