Compilation from sources
========================

`multipers` compilation
***********************

To build from the source files, some more dependencies are needed.
The following commands should provide a working environment

.. code-block:: bash

  conda install python=3.11 cxx-compiler boost tbb tbb-devel numpy matplotlib gudhi scikit-learn cython sympy tqdm cycler typing shapely -c conda-forge

Then clone the repository and pip install it.

.. code-block:: bash

  git clone https://github.com/DavidLapous/multipers
  cd multipers
  python setup.py build_ext -j4 --inplace # optional, but faster this way
  pip install . ## faster with --no-build-isolation
  pytest multipers # optional, does test some functions


**Note:** You can also tune the compilation flags in the `setup.py` file.

External libraries
******************
Some external library may be needed to be compiled from sources, e.g., `mpfree`, `function_delaunay`.
The easiest way to deal with those ones is to compile them in a proper conda environment, and put it in this conda environment.

In the same conda environment as above, execute

.. code-block:: bash
   
   # more dependencies are needed
   conda install llvm-openmp cgal cgal-cpp gmp mpfr -c conda-forge

   # mpfree
   git clone https://bitbucket.org/mkerber/mpfree/
   cd mpfree
   cmake .
   make
   cp mpfree $CONDA_PREFIX/bin/
   cd .. 
   rm -rf mpfree
   
   #function_delaunay
   git clone https://bitbucket.org/mkerber/function_delaunay/
   cd function_delaunay
   cmake .
   make
   cp main $CONDA_PREFIX/bin/function_delaunay
   cd ..
   rm -rf function_delaunay

