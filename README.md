# multipers : Multiparameter Persistence for Machine Learning
[![PyPI](https://img.shields.io/pypi/v/multipers?color=green)](https://pypi.org/project/multipers)
[![Downloads](https://static.pepy.tech/badge/multipers)](https://pepy.tech/project/multipers)
<br>
Scikit-style multiparameter persistent homology python library. 
This library aims to provide easy to use and performant strategies for applied multiparameter topology.
A non-exhaustive list of features can be found in the **Features** section.
<br> Documentation is available [here](https://www-sop.inria.fr/members/David.Loiseaux/doc/multipers/index.html).
<br> Meant to be integrated in [the Gudhi library](https://gudhi.inria.fr/).


## Quickstart

This library is available [on PyPI](https://pypi.org/project/multipers/) for Linux and macOS, via
```sh
pip install multipers
```

A documentation and building instructions are available [here](https://www-sop.inria.fr/members/David.Loiseaux/doc/multipers/index.html).
<br>
Some more jupyter notebooks are available in the `tutorial` folder, but may be outdated.

## Features, and linked projects
This library features a bunch of different functions and helpers. See below for a non-exhaustive list.
<br>Filled box refers to implemented or interfaced code.
 - [x] [Multiparameter Module Approximation](https://arxiv.org/abs/2206.02026) provides the multiparameter simplicial structure, aswell as technics of approximating modules, via interval decompostion modules. It is also very useful for visualization.
 - [x] [Stable Vectorization of Multiparameter Persistent Homology using Signed Barcodes as Measures](https://proceedings.neurips.cc/paper_files/paper/2023/hash/d75c474bc01735929a1fab5d0de3b189-Abstract-Conference.html) provides fast representations of multiparameter persistence modules, by using their signed barcodes decompositions, and encoding it into signed measures. Implemented decompositions : Euler surfaces, Hilbert function, rank invariant (i.e. rectangles). It also provides representation technics for Machine Learning, i.e., Sliced Wasserstein kernels, Vectorizations.
 - [x] [A Framework for Fast and Stable Representations of Multiparameter Persistent Homology Decompositions](https://proceedings.neurips.cc/paper_files/paper/2023/hash/702b67152ec4435795f681865b67999c-Abstract-Conference.html) Provides a vectorization framework for interval decomposable modules, for Machine Learning. Currently implemented as an extension of MMA.
 - [x] [Multiparameter Persistence Landscapes](https://jmlr.org/papers/v21/19-054.html) A vectoriazation technic for multiparameter persistence modules.
 - [x] [Filtration-Domination in Bifiltered Graphs](https://doi.org/10.1137/1.9781611977561.ch3) Allows for 2-parameter edge collapses for 1-critical clique complexes. **Very useful** to speed up, e.g., Rips-Codensity bifiltrations.
 - [x] [Chunk Reduction for Multi-Parameter Persistent Homology](https://doi.org/10.4230/LIPIcs.SoCG.2019.37) Multi-filtration preprocessing algorithm.
 - [x] [Computing Minimal Presentations and Bigraded Betti Numbers of 2-Parameter Persistent Homology](https://arxiv.org/abs/1902.05708) Minimal presentation of multiparameter persistence modules, using [mpfree](https://bitbucket.org/mkerber/mpfree/src/master/). Hilbert Decomposition Signed Measures, and MMA decompositions can be computed using the mpfree backend.
 - [x] [Delaunay Bifiltrations of Functions on Point Clouds](https://arxiv.org/abs/2310.15902) Provides an alternative to function rips bifiltrations, using Delaunay complexes. Very good alternative to Rips-Density like bi-filtrations.
 - [x] [Rivet](https://github.com/rivetTDA/rivet) Interactive two parameter persistence
 - [ ] [Backend only] [Projected distances for multi-parameter persistence modules](https://arxiv.org/abs/2206.08818) Provides a strategy to estimate the convolution distance between multiparameter persistence module using projected barcodes. Implementation is a WIP.
 - [ ] [Partial, and experimental] [Efficient Two-Parameter Persistence Computation via Cohomology](https://doi.org/10.4230/LIPIcs.SoCG.2023.15) Minimal presentations for 2-parameter persistence algorithm.

If I missed something, or you want to add something, feel free to open an issue.

## Authors
[David Loiseaux](https://www-sop.inria.fr/members/David.Loiseaux/index.html),<br>
[Hannah Schreiber](https://github.com/hschreiber) (Persistence backend code),<br>
[Luis Scoccola](https://luisscoccola.com/) 
(Möbius inversion in python, degree-rips using [persistable](https://github.com/LuisScoccola/persistable) and [RIVET](https://github.com/rivetTDA/rivet/)),<br>
[Mathieu Carrière](https://www-sop.inria.fr/members/Mathieu.Carriere/) (Sliced Wasserstein)<br>

## Contributions

Feel free to contribute, report a bug on a pipeline, or ask for documentation by opening an issue.<br>


