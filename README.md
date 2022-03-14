# Multipersistence modules approximation

**/!\ Warning** This is a clone of [the gitlab repository](https://gitlab.inria.fr/dloiseau/multipers).
## Description
This repository is a python library, with a `C++` backend, for multipersistence modules approximation. 
It provides a set of functions to compute an approximation and fibered barcodes of a persistence module of any dimension n from a Gudhi simplex tree (python interface) and its n-dimensional filtration, and plot module approximations on a compact rectangle.

### Authors
[David Loiseaux](http://www-sop.inria.fr/members/David.Loiseaux/) and [Mathieu Carrière](https://mathieucarriere.github.io/website/).

### Contributor
[Hannah Schreiber](https://github.com/hschreiber)

### References
TODO

Most multipers.py functions taken from [Mathieu Carrière](https://github.com/MathieuCarriere/multipers)'s Github; which adds the [dionysus library](https://github.com/mrzv/dionysus) as a dependency.

## Installation, compilation
### Dependencies
The `C++` part only uses the standard library of `C++`. Aimed for `c++17`. <br>
The `Python` part relies on several standard packages : `cython`, `sys`, `numpy`, `libcpp`,  `matplotlib`, and `shapely`; the [Gudhi](https://gudhi.inria.fr) library is also needed, and can be easily installed using pip or conda, following [the python documentation of Gudhi](https://gudhi.inria.fr/python/latest/installation.html#packages).

### Compilation

#### Using pip
Clone this repository and compile-install this library using the following commands

>	cd /path-to-cloned-directory/	<br>
>	pip install --user custom_vineyards/


#### Get the compiled file (that can manually be imported from python)
Clone this repository and execute the following commands in a terminal to compile the `C++` code, and move the compiled file in the root folder

> 	cd /path-to-cloned-directory/	<br>
> 	cd custom_vineyards/	<br>
> 	python setup.py build_ext	<br>
> 	find build/ -name "*.so" -exec mv {} ../ \;

You should end up with a file `mma.cpython-*.so` (where `*` depends on your setup) in the root folder of the project.

### C++ documentation
Compile the Doxygen documentation using the following commands

> 	cd /path-to-cloned-directory/	<br>
> 	doxygen doc/Doxyfile

and open the `doc/html/index.html` using your favourite web-browser.

## Usage
Import the compiled library and Gudhi in Python:
```
from mma import *
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
	- `verbose = False` : If set to `True`, the `C++` code will give some information about the computation,
    -  `multithread` (optional) if set to true, will compute higher dimensions in parallel. (WIP)
- `vine_alt(simplextree, filtration, precision, box)` takes the same inputs, and compute the fibered barcodes along a set of regularly distributed lines
- `plot_approx_2d(simplextree, filtration, precision, box)` plots an approximation of the bimodule on the box. Same input as the previous function, but dimension has to be specified. It can take more optional inputs :
	- `return_corners=False` : if set to `True`, makes this algorithm also return the generator & relations used to plot the approximation bimodule,
	- `separated = False` : if set to `True`, makes a different plot for each summand,
	- `min_interleaving = 0` : the summands that are `min_interleaving`-interleaved with the `0` bimodule are not plotted,
	- `alpha = 1`		: set the alpha value of the summand's color,
	- `keep_order=False` : If set to true, will keep the summands (and their plot color) ordered at a small computational overhead,
	- `shapely = True` : if `True` and `alpha` is lower than 1, this will call the shapely library to output a correct plot (Recommended).
	- `save = False` : if nonempty, saves the figure as the string contained in `save`,
	- `dpi = 50` : sets the dpi of the saved figure.
- `plot_vine_2d(simplextree, filtration, precision, box)` plots the matched fibered barcode.


Follow the tutorial notebooks `How to use custom_vineyards` and `examples_of_approximations` for more functions, and detailed examples.
