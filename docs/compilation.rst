Compilation from sources
========================

`multipers` compilation
***********************

To build from the source files, some more dependencies are needed.
The following commands should provide a working environment

.. code-block:: bash

  conda install python=3.12  gudhi=3.10 numpy=2 cython=3 libboost-devel tbb tbb-devel pytest pot scikit-learn matplotlib joblib tqdm scipy  -c conda-forge
  pip install pykeops filtration-domination --upgrade

If you don't have a proper build system you can install one with conda, e.g.,

.. code-block:: bash

   conda install cxx-compiler -c conda-forge

Then clone the repository and pip install it.

.. code-block:: bash

  git clone https://github.com/DavidLapous/multipers
  cd multipers
  python setup.py build_ext -j4 --inplace # this may take some time
  pip install --no-build-isolation .
  # tests
  for f in tests/test_*.py; do pytest "$f" || break; done


**Note:** You can also tune the compilation flags in the `setup.py` file. 

**In particular,** if you use macOS, the clang compiler may fail to compile multipers if 
the `aligned-new` compiler optimization is enabled; in that case, please disable it in the `setup.py` file by uncommenting the following line.

.. code-block:: python

  # "-fno-aligned-new", # Uncomment this if you have trouble compiling on macos.

External libraries
******************

Some external library may be needed to be compiled from sources, e.g., `mpfree`, `function_delaunay`.
The easiest way to deal with those is to compile them in a proper conda environment, and put it in this conda environment.

In the same conda environment as above, execute

.. code-block:: bash
   
  # more dependencies are needed
  conda install llvm-openmp cgal cgal-cpp gmp mpfr eigen cmake -c conda-forge

  # mpfree
  git clone https://bitbucket.org/mkerber/mpfree/
  cd mpfree
  cmake .
  make
  cp mpfree $CONDA_PREFIX/bin/
  cd .. 
  
  # function_delaunay
  git clone https://bitbucket.org/mkerber/function_delaunay/
  cd function_delaunay
  sed -i "8i find_package(Eigen3 3.3 REQUIRED NO_MODULE)\nlink_libraries(Eigen3::Eigen)\n" CMakeLists.txt
  cmake .
  make
  cp main $CONDA_PREFIX/bin/function_delaunay
  cd ..

