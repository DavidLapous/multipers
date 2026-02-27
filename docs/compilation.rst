Compilation from sources
========================

`multipers` compilation
***********************

To build from the source files, some more dependencies are needed.
The following commands should provide a working environment

.. code-block:: bash

  conda install python=3.12  gudhi=3.10 numpy=2 cython=3 libboost-devel tbb tbb-devel pytest pot scikit-learn matplotlib joblib tqdm scipy cmake -c conda-forge
  pip install pykeops filtration-domination --upgrade

If you don't have a proper build system you can install one with conda, e.g.,

.. code-block:: bash

   conda install cxx-compiler -c conda-forge

Then clone the repository and pip install it.

.. code-block:: bash

  git clone https://github.com/DavidLapous/multipers
  cd multipers
  cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
  cmake --build build -j4 # this may take some time
  cp build/compile_commands.json ./compile_commands.json
  pip install --no-build-isolation .

For incremental development after the first build, a fast loop is:

.. code-block:: bash

  python -m build -n --wheel --no-isolation
  pip install --force-reinstall dist/*.whl

You can also use the direct path below (it remains incremental with the persistent
``build/{wheel_tag}`` directory configured in ``pyproject.toml``):

.. code-block:: bash

  pip install --no-build-isolation .

  # tests
  for f in tests/test_*.py; do pytest "$f" || break; done


**Note:** You can tune compilation flags with CMake variables or environment flags (`CXXFLAGS`, `CFLAGS`).

**In particular,** if you use macOS, the clang compiler may fail to compile multipers if
the `aligned-new` compiler optimization is enabled; in that case, pass the following flags during configure/build.

.. code-block:: bash

  CXXFLAGS="-fno-aligned-new" CFLAGS="-fno-aligned-new" cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
  cmake --build build -j4

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
  conda install llvm-openmp cgal cgal-cpp gmp mpfr eigen cmake -c conda-forge

  # mpfree (standalone binary, optional fallback)
  cd ext/mpfree
  cmake .
  make
  cp mpfree $CONDA_PREFIX/bin/
  cd .. 
  
  # function_delaunay (standalone binary, optional fallback)
  cd function_delaunay
  sed -i "8i find_package(Eigen3 3.3 REQUIRED NO_MODULE)\nlink_libraries(Eigen3::Eigen)\n" CMakeLists.txt
  cmake .
  make
  cp main $CONDA_PREFIX/bin/function_delaunay
  cd ..


Licensing of full-feature builds
********************************

When building `multipers` with the compiled external interfaces enabled
(`AIDA`, `Persistence-Algebra`, `function_delaunay`, `mpfree`, `multi_critical`, `multi_chunk`),
the resulting distribution is provided under **GPL-3.0-or-later**.

See `THIRD_PARTY_NOTICES.md` at the repository root for the list of third-party components,
upstream references, and pinned revisions.
