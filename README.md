# Multipersistence modules approximation
This is cloned from [Gitlab](https://gitlab.inria.fr/dloiseau/multipers).
## Description
Topological Data Analysis (TDA) Python library, with a `C++` backend, for multipersistence modules approximation.
It provides a set of functions to compute an approximation and fibered barcodes of a persistence module of any dimension n from a Gudhi simplex tree (python interface) and its n-dimensional filtration, and plot module approximations on a compact rectangle.

### Authors
[David Loiseaux](http://www-sop.inria.fr/members/David.Loiseaux/) and [Mathieu Carrière](https://mathieucarriere.github.io/website/).


### References
TODO

Most multipers.py functions taken from [Mathieu Carrière](https://github.com/MathieuCarriere/multipers)'s Github; which adds the [dionysus library](https://github.com/mrzv/dionysus) as a dependency.

## Installation, compilation
### Compilation
Clone this repository and execute the following commands in a terminal to compile the `C++` code, and move the compiled file in the root folder

> $	cd /path-to-cloned-directory/   
> $	cd custom_vineyards/    
> $	python setup.py build_ext   
> $	find build/ -name "*.so" -exec mv {} ../ \;

### Gudhi
You will also need the python backend of [Gudhi](https://gudhi.inria.fr) which can be installed following [the python documentation of Gudhi](https://gudhi.inria.fr/python/latest/installation.html#packages).

### C++ documentation
Compile the Doxygen documentation using the following commands

> $	cd /path-to-cloned-directory/   
> $	doxygen doc/Doxyfile

and open the `doc/html/index.html` using your favourite web-browser.

## Usage
Import the compiled library and Gudhi in Python:
```
from custom_vineyards import *
import gudhi as gd
```
### Main functions
- `approx(simplextree, filtration, precision, box)` computes an interval decomposable approximation of the `n`-persistence module defined by the following `simplextree` and `multifiltration` below, with approximation parameters :
    -  `simplextree` is a Gudhi simplextree, 
    -  `filtration` is the list of filtration values of the simplices, which gets completed as a lower-star filtration if it is not complete,
    -  `precision` is the distance between two lines,
    -  `box` is the support of this computation : each bar along a line which intersects this box will be returned.
    -  `dimension` (optional) if positive integer, returns only the barcodes of this dimension,
    -  `threshold` (optional) if set to true, will intersects the bars along the box (for plot purposes)
    -  `multithread` (optional) if set to true, will compute higher dimensions in parallel. (WIP)
- `vine_alt(simplextree, filtration, precision, box)` takes the same inputs, and compute the fibered barcodes along a set of regularly distributed lines
- `plot_approx_2d(simplextree, filtration, precision, box)` plots an approximation of the bimodule on the box. Same input as the previous function, but dimension has to be specified.
- `plot_vine_2d(simplextree, filtration, precision, box)` plots the matched fibered barcode.

Follow the tutorial notebook for more functions, and examples.
