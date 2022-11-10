# Multipersistence modules approximation

**/!\ Warning** This is a clone of [the gitlab repository](https://gitlab.inria.fr/dloiseau/multipers).

## Description
This repository is a python library, with a `C++` backend, for multipersistence modules approximation. 
It provides a set of functions to compute an approximation of any multiparameter persistence module (from 1-critical filtrations if using python), aswell as some representations, e.g., Fibered Barcode, Multiparameter Persistence Images, Multiparameter Persistence Landscapes, etc.

### Authors
[David Loiseaux](http://www-sop.inria.fr/members/David.Loiseaux/) and [Mathieu Carrière](https://www-sop.inria.fr/members/Mathieu.Carriere/).

### Contributor
[Hannah Schreiber](https://github.com/hschreiber)

### References
[Arxiv link](https://arxiv.org/abs/2206.02026).

Most multipers.py functions taken from [Mathieu Carrière](https://github.com/MathieuCarriere/multipers)'s Github; which adds the [dionysus library](https://github.com/mrzv/dionysus) as a dependency.

## Installation, compilation
### Dependencies
The `C++` part only uses the standard library of `C++`.<br>
The `Python` part relies on several standard packages, that can be found in `src/requirements.txt`; the [Gudhi](https://gudhi.inria.fr) library is needed, and can be easily installed using pip or conda, following [the python documentation of Gudhi](https://gudhi.inria.fr/python/latest/installation.html#packages).

For strong edges collapses, the pip package of [filtration-domination](https://github.com/aj-alonso/filtration-domination) is needed. It **needs** the rust toolchain, that can be installed with  `conda install rust` or by following the [Rust documentation](https://www.rust-lang.org/tools/install).

### Compilation
#### Using pip
Clone (or download) this repository and compile-install this library using the following commands in a terminal in the `src` folder

>	pip install --user .

#### Or manually :
Similarly, in the `src` folder, the following command will compile the python code.

> 	python setup.py build_ext	<br>

You should end up with a file `mma.cpython-*.so` (where `*` depends on your setup) in a `build` folder. This file can be directly imported to python.

<!-- ### C++ documentation
Compile the Doxygen documentation using the following commands

> 	cd /path-to-cloned-directory/	<br>
> 	doxygen doc/Doxyfile

and open the `doc/html/index.html` using your favourite web-browser. -->

## Usage
Import the compiled library and Gudhi in Python:
```
import mma
import gudhi as gd
```
For an introduction, follow the tutorial notebook `How to use`.