Compilation from sources
========================

`multipers` compilation
***********************

To build from the source files, some more dependencies are needed.
The following commands should provide a working environment

.. code-block:: bash

  micromamba install python=3.12 gudhi>=3.10 numpy>=2  libboost tbb pytest pot scikit-learn matplotlib joblib tqdm scipy cmake ninja -c conda-forge
  pip install pykeops filtration-domination --upgrade

Source builds require the Ninja generator. If you do not already have a compiler toolchain,
install one in the same environment, for example:

.. code-block:: bash

   micromamba install cxx-compiler -c conda-forge

Then clone the repository and pip install it.

.. code-block:: bash

  git clone https://github.com/DavidLapous/multipers
  cd multipers
  pip install . 

  # or with cmake

  cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
  cmake --build build --parallel # this may take some time
  cp build/compile_commands.json ./compile_commands.json
  pip install --no-build-isolation .


  # tests
  for f in tests/test_*.py; do pytest "$f" || break; done

The exported ``compile_commands.json`` records each backend interface flag as
an explicit ``0`` or ``1`` definition (for example
``MULTIPERS_DISABLE_MPFREE_INTERFACE=0``), which helps ``clangd`` follow the
same preprocessor paths as the configured build.
The shared ``multipers/ext_interface`` runtime translation unit also carries the
optional backend include directories so ``clangd`` can evaluate
``__has_include(...)`` checks in those headers the same way as the configured
build.


For ``mpfree``, ``multi_critical``, and ``function_delaunay``, build 
auto-applies generated runtime-log patch overlays from:

.. code-block:: bash

  ext/patches/function_delaunay_runtime_logs.patch
  ext/patches/mpfree_runtime_logs.patch
  ext/patches/multi_critical_runtime_logs.patch

onto build-local header overlays before compiling the native extensions. The
vendored checkouts under ``ext/`` are not edited in place.
Vendored log patch workflow lives under ``ext/patches/``. Generate the tracked
patch artifacts from a configured build tree with:

.. code-block:: bash

  cmake --build build --target multipers_generate_ext_patches

Normal native builds also pull those generated patch files in as dependencies
before the patch-overlay steps. Direct script regen still works for one-off
single-backend updates:

.. code-block:: bash

  python ext/patches/generate_log_patch.py function_delaunay
  python ext/patches/generate_log_patch.py mpfree
  python ext/patches/generate_log_patch.py multi_critical


**In particular,** if you use an old version of macOS, the clang compiler may fail to compile multipers if
the `aligned-new` compiler optimization is enabled; in that case, pass the following flags during configure/build.

.. code-block:: bash

  CXXFLAGS="-fno-aligned-new" CFLAGS="-fno-aligned-new" cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
  cmake --build build --parallel

External libraries
******************

External libraries are tracked as git submodules under ``ext/``.
Initialize them before building:

.. code-block:: bash

  git submodule update --init --recursive

Some external tools may still be useful as standalone binaries in your conda environment,
for example `mpfree` and `function_delaunay` fallback paths.

In the same conda environment as above, execute

.. code-block:: bash
   
  # more dependencies are needed
  micromamba install llvm-openmp cgal cgal-cpp gmp mpfr eigen cmake ninja -c conda-forge

  # mpfree (standalone binary, optional fallback)
  cd ext/mpfree
  cmake -S . -B build -G Ninja
  cmake --build build --parallel
  cp build/mpfree $CONDA_PREFIX/bin/
  cd .. 
  
  # function_delaunay (standalone binary, optional fallback)
  cd function_delaunay
  sed -i "8i find_package(Eigen3 3.3 REQUIRED NO_MODULE)\nlink_libraries(Eigen3::Eigen)\n" CMakeLists.txt
  cmake -S . -B build -G Ninja
  cmake --build build --parallel
  cp build/main $CONDA_PREFIX/bin/function_delaunay
  cd ..


Licensing of full-feature builds
********************************

When building `multipers` with the compiled external interfaces enabled
(`AIDA`, `Persistence-Algebra`, `function_delaunay`, `mpfree`, `multi_critical`, `multi_chunk`),
the resulting distribution is provided under **GPL-3.0-or-later**.

See `THIRD_PARTY_NOTICES.md` at the repository root for the list of third-party components,
upstream references, and pinned revisions.
