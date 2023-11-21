# Multipers
Scikit-style multiparameter persistent homology python library. 
This librairy aims to provide easy to use and performant strategies for applied multiparameter topology.
<br> Meant to be integrated in [the Gudhi library](https://gudhi.inria.fr/).

## Installation
### Dependencies
Using conda
```sh
conda create -n python311
conda activate python311
conda install python=3.11 numpy matplotlib gudhi scikit-learn scipy sympy tqdm shapely -c conda-forge
pip install filtration-domination pykeops
```
Other libraries may be required for some specific functions, e.g., pytorch for to compute the rank invariant.

### Installation
#### Using pip (Recommended)
Precompiled versions, for linux and macos, are available [on PyPI](https://pypi.org/project/multipers/). Just use 
```sh
pip install multipers
```
#### Building from source
Clone the repo, and in a terminal, in the root folder
```sh
conda install python=3.11 cxx-compiler boost tbb tbb-devel numpy matplotlib gudhi scikit-learn cython sympy tqdm cycler typing shapely -c conda-forge
pip install .
```
It has been tested with python 3.11 on Linux (gcc12) and Macos (clang14-clang16). 
If the build fails (on macos) see a fix at the end of the readme. 
Don't hesitate to fill an issue if this doesn't work out of the box; it can help future users ;-).

## How to use
### Tutorial notebooks
We provide a few notebooks, which explains, in different scenarios, how to use our library. <br>
**Take a look at them !** They are in the tutorial folder.<br>
If something is not clear, or you have trouble using a function, don't hesitate to fill an issue; I'll try to update this.


## References and similar projects
Filled box refers to implemented or interfaced code.
 - [x] [Multiparameter Module Approximation](https://arxiv.org/abs/2206.02026) provides the multiparameter simplicial structure, aswell as technics of approximating modules, via interval decompostion modules. It is also very useful for visualization.
 - [x] [Stable Vectorization of Multiparameter Persistent Homology using Signed Barcodes as Measures](https://arxiv.org/abs/2306.03801) provides fast representations of multiparameter persistence modules, by using their signed barcodes decompositions, and encoding it into signed measures. Implemented decompositions : Euler surfaces, Hilbert function, rank invariant (i.e. rectangles). It also provides representation technics for Machine Learning, i.e., Sliced Wasserstein kernels, Vectorizations.
 - [x] [A Framework for Fast and Stable Representations of Multiparameter Persistent Homology Decompositions](https://arxiv.org/abs/2306.11170) Provides a vectorization framework for interval decomposable modules, for Machine Learning. Currently implemented as an extension of MMA.
 - [x] [Filtration-Domination in Bifiltered Graphs](https://doi.org/10.1137/1.9781611977561.ch3) Allows for 2-parameter edge collapses for 1-critical clique complexes. **Very useful** to speed up, e.g., Rips-Codensity bifiltrations. 
 - [x] [Computing Minimal Presentations and Bigraded Betti Numbers of 2-Parameter Persistent Homology](https://arxiv.org/abs/1902.05708) Minimal presentation of multiparameter persistence modules, using [mpfree](https://bitbucket.org/mkerber/mpfree/src/master/). Hilbert Decomposition Signed Measures can be computed using the mpfree backend.
 - [ ] [Projected distances for multi-parameter persistence modules](https://arxiv.org/abs/2206.08818) Provides a strategy to estimate the convolution distance between multiparameter persistence module using projected barcodes. Implementation is a WIP.
 - [ ] [Delaunay Bifiltrations of Functions on Point Clouds](https://arxiv.org/abs/2310.15902) Provides an alternative to function rips bifiltrations, using Delaunay complexes.
 - [ ] [Efficient Two-Parameter Persistence Computation via Cohomology](https://doi.org/10.4230/LIPIcs.SoCG.2023.15) Minimal presentations for 2-parameter persistence clique complexes.
 - [Rivet](https://github.com/rivetTDA/rivet) Interactive two parameter persistence. From multipers to rivet is implemented, WIP for the rest.


## Authors
[David Loiseaux](https://www-sop.inria.fr/members/David.Loiseaux/index.html), 
[Luis Scoccola](https://luisscoccola.com/) 
(Möbius inversion in python, degree-rips using [persistable](https://github.com/LuisScoccola/persistable) and [RIVET](https://github.com/rivetTDA/rivet/)), 
[Mathieu Carrière](https://www-sop.inria.fr/members/Mathieu.Carriere/) (Sliced Wasserstein).

## Contributions
Hannah Schreiber

Feel free to contribute, report a bug on a pipeline, or ask for documentation by opening an issue.<br>
**Any contribution is welcome**.




## For mac users 
Due to the clang compiler, one may have to disable a compilator optimization to compile `multipers`: in the `setup.py` file, add the 
```bash
-fno-aligned-new
```
line in the `extra_compile_args` list. You should have should end up with something like the following.
```python
extensions = [Extension(f"multipers.{module}",
	sources=[f"multipers/{module}.pyx"],
	language='c++',
	extra_compile_args=[
		"-Ofast",
		"-std=c++20",
		"-fno-aligned-new",
		'-ltbb',
		"-Wall",
	],
	extra_link_args=['-ltbb'],
	define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
) for module in cython_modules]
```
#### Alternatives
One may try to use the `clang` compiler provided by conda or brew. If you have a simpler alternative, please let me know ;)
