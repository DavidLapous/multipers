# multipers : Multiparameter Persistence for Machine Learning
[![DOI](https://joss.theoj.org/papers/10.21105/joss.06773/status.svg)](https://doi.org/10.21105/joss.06773) [![Documentation](https://img.shields.io/badge/Documentation-website-blue)](https://davidlapous.github.io/multipers) [![Build, test](https://github.com/DavidLapous/multipers/actions/workflows/python_PR.yml/badge.svg)](https://github.com/DavidLapous/multipers/actions/workflows/python_PR.yml)
<br>
Scikit-style PyTorch-autodiff multiparameter persistent homology python library. 
This library aims to provide easy to use and performant strategies for applied multiparameter topology.
<br> Meant to be integrated in the [Gudhi](https://gudhi.inria.fr/) library.

## Compiled packages
| Source | Version | Downloads | Platforms | 
| --- | --- | --- | --- | 
| [![Conda Recipe](https://img.shields.io/badge/conda-recipe-green.svg)](https://github.com/conda-forge/multipers-feedstock)| [![Conda Version](https://img.shields.io/conda/vn/conda-forge/multipers.svg)](https://anaconda.org/conda-forge/multipers) |  [![Conda Downloads](https://img.shields.io/conda/dn/conda-forge/multipers.svg)](https://anaconda.org/conda-forge/multipers) |[![Conda Platforms](https://img.shields.io/conda/pn/conda-forge/multipers.svg)](https://anaconda.org/conda-forge/multipers) | 
| [![pip Recipe](https://img.shields.io/badge/pip-package-green.svg)](https:///pypi.org/project/multipers) | [![PyPI](https://img.shields.io/pypi/v/multipers?color=green)](https://pypi.org/project/multipers) | [![ pip downloads](https://static.pepy.tech/badge/multipers)](https://pepy.tech/project/multipers) | | 



## Quick start
This library allows computing several representations from "geometrical datasets", e.g., point clouds, images, graphs, that have multiple scales.
We provide some *nice* pictures in the [documentation](https://davidlapous.github.io/multipers/index.html). 
A non-exhaustive list of features can be found in the **Features** section.

This library is available on pip and conda-forge for (reasonably up to date) Linux, macOS and Windows, via
```sh
pip install multipers
```
or 
```sh
conda install multipers -c conda-forge
```
Pre-releases are available via
```sh
pip install --pre multipers
```
These release usually contain small bugfixes or unstable new features.

Windows support is experimental, and some core dependencies are not available on Windows.
We hence recommend Windows user to use [WSL](https://learn.microsoft.com/en-us/windows/wsl/).
<br>
A documentation and building instructions are available
[here](https://davidlapous.github.io/multipers/compilation.html).


## Features, and linked projects
This library features a bunch of different functions and helpers. See below for a non-exhaustive list.
<br>Filled box refers to implemented or interfaced code.
 - [x] [[Multiparameter Module Approximation]](https://arxiv.org/abs/2206.02026) provides the multiparameter simplicial structure, as well as technics for approximating modules, via interval-decomposable modules. It is also very useful for visualization.
 - [x] [[Stable Vectorization of Multiparameter Persistent Homology using Signed Barcodes as Measures, NeurIPS2023]](https://proceedings.neurips.cc/paper_files/paper/2023/hash/d75c474bc01735929a1fab5d0de3b189-Abstract-Conference.html) provides fast representations of multiparameter persistence modules, by using their signed barcodes decompositions encoded into signed measures. Implemented decompositions : Euler surfaces, Hilbert function, rank invariant (i.e. rectangles). It also provides representation technics for Machine Learning, i.e., Sliced Wasserstein kernels, and Vectorizations.
 - [x] [[A Framework for Fast and Stable Representations of Multiparameter Persistent Homology Decompositions, NeurIPS2023]](https://proceedings.neurips.cc/paper_files/paper/2023/hash/702b67152ec4435795f681865b67999c-Abstract-Conference.html) Provides a vectorization framework for interval decomposable modules, for Machine Learning. Currently implemented as an extension of MMA.
 - [x] [[Differentiability and Optimization of Multiparameter Persistent Homology, ICML2024]](https://proceedings.mlr.press/v235/scoccola24a.html) An approach to compute a (clarke) gradient for any reasonable multiparameter persistent invariant. Currently, any `multipers` computation is auto-differentiable using this strategy, provided that the input are pytorch gradient capable tensor.
 - [x] [[Multiparameter Persistence Landscapes, JMLR]](https://jmlr.org/papers/v21/19-054.html) A vectorization technic for multiparameter persistence modules.
 - [x] [[Filtration-Domination in Bifiltered Graphs, ALENEX2023]](https://doi.org/10.1137/1.9781611977561.ch3) Allows for 2-parameter edge collapses for 1-critical clique complexes. Very useful to speed up, e.g., Rips-Codensity bifiltrations.
 - [x] [[Chunk Reduction for Multi-Parameter Persistent Homology, SOCG2019]](https://doi.org/10.4230/LIPIcs.SoCG.2019.37) Multi-filtration preprocessing algorithm for homology computations.
 - [x] [[Computing Minimal Presentations and Bigraded Betti Numbers of 2-Parameter Persistent Homology, JAAG]](https://doi.org/10.1137/20M1388425) Minimal presentation of multiparameter persistence modules, using [mpfree](https://bitbucket.org/mkerber/mpfree/src/master/). Hilbert, Rank Decomposition Signed Measures, and MMA decompositions can be computed using the mpfree backend.
 - [x] [[Delaunay Bifiltrations of Functions on Point Clouds, SODA2024]](https://epubs.siam.org/doi/10.1137/1.9781611977912.173) Provides an alternative to function rips bifiltrations, using Delaunay complexes. Very good alternative to Rips-Density like bifiltrations.
 - [x] [[Delaunay Core Bifiltration]](https://arxiv.org/abs/2405.01214) Bifiltration for point clouds, taking into account the density. Similar to Rips-Density. 
 - [x] [[Rivet]](https://github.com/rivetTDA/rivet) Interactive two parameter persistence
 - [x] [[Kernel Operations on the GPU, with Autodiff, without Memory Overflows, JMLR]](http://jmlr.org/papers/v22/20-275.html) Although not linked, at first glance, to persistence in any way, this library allows computing blazingly fast signed measures convolutions (and more!) with custom kernels. 
 - [ ] [Backend only] [[Projected distances for multi-parameter persistence modules]](https://arxiv.org/abs/2206.08818) Provides a strategy to estimate the convolution distance between multiparameter persistence module using projected barcodes. Implementation is a WIP.
 - [ ] [Partial, and experimental] [[Efficient Two-Parameter Persistence Computation via Cohomology, SoCG2023]](https://doi.org/10.4230/LIPIcs.SoCG.2023.15) Minimal presentations for 2-parameter persistence algorithm.

If I missed something, or you want to add something, feel free to open an issue.

## Authors
[David Loiseaux](https://davidlapous.github.io/),<br>
[Hannah Schreiber](https://github.com/hschreiber) (Persistence backend code),<br>
[Luis Scoccola](https://luisscoccola.com/) 
(Möbius inversion in python, degree-rips using [persistable](https://github.com/LuisScoccola/persistable) and [RIVET](https://github.com/rivetTDA/rivet/)),<br>
[Mathieu Carrière](https://www-sop.inria.fr/members/Mathieu.Carriere/) (Sliced Wasserstein),<br>
[Odin Hoff Gardå](https://odinhg.github.io/) (Delaunay Core bifiltration).<br>

## Citation
Please cite this library when using it in scientific publications;
you can use the following journal bibtex entry
```bib
@article{multipers,
  title = {Multipers: {{Multiparameter Persistence}} for {{Machine Learning}}},
  shorttitle = {Multipers},
  author = {Loiseaux, David and Schreiber, Hannah},
  year = {2024},
  month = nov,
  journal = {Journal of Open Source Software},
  volume = {9},
  number = {103},
  pages = {6773},
  issn = {2475-9066},
  doi = {10.21105/joss.06773},
  langid = {english},
}
```
## Contributions
Feel free to contribute, report a bug on a pipeline, or ask for documentation by opening an issue.<br>
In particular, if you have a nice example or application that is not taken care in the documentation (see the `./docs/notebooks/` folder), please contact me to add it there.

