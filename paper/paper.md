---
title: '`multipers` : Multiparameter Persistence for Machine Learning'

tags:
  - machine learning
  - topological data analysis

authors:
 - name: David Loiseaux
   affiliation: 1
 - name: Hannah Schreiber
   affiliation: 1

affiliations:
 - name: Centre Inria d'Université Côte d'Azur, France
   index: 1

date: 15/04/2024
bibliography: paper.bib
---

# Summary

`multipers` is a Python library for Topological Data Analysis, focused on **Multi**parameter **Pers**istence computation and visualizations for Machine Learning.
It features several efficient computational and visualization tools, with integrated, easy to use, auto-differentiable Machine Learning pipelines, that can be seamlessly interfaced with 
`scikit-learn` [@scikit_learn] and `PyTorch` [@pytorch].
This library is meant to be usable for non-experts in Topological or Geometrical Machine Learning.
Performance-critical functions are implemented in `C++` or in `Cython`, are parallelizable with `TBB`, and have `Python` bindings and interface.
<!--Additionally, it follows `Gudhi`'s [@gudhi] structure, for future integration.-->
It can handle a very diverse range of datasets that can be framed into a (finite) multi-filtered simplicial or cell complex, including, e.g., point clouds, graphs, time series, images, etc.

![\label{1} 
**(Left)** Topological 2-filtration grid. 
The color corresponds to the density estimation of the sampling measure of the point cloud. 
More formally, a point $x\in \mathbb R^2$ belongs to the grid cell with coordinates $(r,d)$
iff $d(x,\mathrm{point\ cloud}) \le r$ and $\mathrm{density}(x) \ge d$. 
The green background shape corresponds to the lifetime of the annulus in this 2-parameter grid.
**(Right)** A visualization of the lifetimes of geometric structures given by `multipers`; 
here each colored shape corresponds to a cycle appearing in the bi-filtration on the left,
and the shape represents its lifetime. The biggest green shape on the right is the same as the one on the left.](images/annulus.png)

**Some motivation.** In the example of Figure \ref{1}, a point cloud is given from sampling a probability measure whose mass is, for the most part, located on an annulus,
with some diffuse background noise.
The goal here is to recover this information 
in a topological descriptor. For this, the point cloud can be analyzed at some geometric scale $r>0$ and density scale $d$ by centering balls of radius $r$ around each point whose density is above $d$, and looking at the topology induced by the union of balls. 
However, notice that neither a fixed geometric scale nor density scale alone can retrieve (canonically) meaningful information due to the diffuse noise in the background;
which is the main limitation of the prevalent approach.
Nevertheless, by considering *all* possible combinations of geometric or density scales, also called a bi-filtration, it becomes straightforward with `multipers` to retrieve some of the underlying geometrical structures without relying on any arbitrary scale choice. 



Furthermore, `multipers` seamlessly integrates several `Rust` and `C++` libraries such as `Gudhi` [@gudhi], `filtration-domination` [@filtration_domination], `mpfree` [@mpfree], and `function-delaunay` [@function_delaunay],
and leverages on state-of-the-art
Machine Learning libraries for fast computations, such as `scikit-learn` [@scikit_learn], `Python Optimal Transport` [@pot], `PyKeops` [@pykeops], or `PyTorch` [@pytorch]. 
This makes `multipers` a very efficient and fully-featured library, showcasing a wide variety of mathematically-grounded multiparameter topological invariants, including,
e.g., Multiparameter Module Approximation [@mma], Euler, Hilbert, and Rectangle Signed Barcodes [@signed_barcode] [@signed_betti], Multiparameter Persistent Landscapes [@mpl]; each of them computable from several multi-filtrations, e.g.,
Rips-Density-like filtrations, Cubical, Degree-Rips, Function-Delaunay, or any $k$-critical multi-filtration. 
These topological descriptors can then directly be used in auto-differentiable Machine Learning pipelines, using the differentiability framework developed in [@sm_diff],
through several methods, such as, e.g., 
Decomposable Module Representations [@mma_vect], Sliced Wasserstein Kernels or Convolutions from Signed Measures [@sb_as_sm].
<!--Furthermore, by leveraging on several external libraries,-->
As a result, `multipers` is capable of handling, within a single minute of computation, datasets of $\sim 50k$ points with only 5 lines of Python code. See Figures \ref{2}, \ref{3}.


![\label{2} Typical interpretation of a "Geometric \& Density" bi-filtration with `multipers`. 
**(Left)** Point cloud with color induced by density estimation (same as Figure \ref{1}).
**(Right)** A visualization of the topological structure lifetimes computed from a Delaunay-Codensity bi-filtration;
here the three cycles can be retrieved using their radii (x-axis) and their co-densities (y-axis). 
The first cycle is the densiest, and smallest, and thus corresponds to the one that appears in the bottom(high-density)-left(small-radius)
of the bi-filtration. The second is less dense (thus above the first one) and bigger (thus more on the right). The same goes for the last one.](images/3cycles.png)

![\label{3} Different Signed Barcodes from the same dataset as Figure \ref{1}.
**(Left)** Euler Decomposition Signed Barcode, and the Euler Surface in the background.
**(Middle)** Hilbert Decomposition Signed Barcode, with its Hilbert Function surface.
**(Right)** Rank invariant Signed Barcode, with the Hilbert Function as a background.](images/SignedBarcodes.png)

The core functions of the Python library are automatically tested on Linux and macOS, using `pytest` alongside GitHub Actions.


# Related work and statement of need
There exists several libraries for computation or pre-processing of very specific tasks related to multiparameter persistence. However, to the best of our knowledge, none of them are able to tackle the challenges that `multipers` is dealing with, i.e.,
**(1)** computing and unifying the computations of multiparameter persistent structures, in a non-expert friendly approach, and 
**(2)** provide ready-to-use general tools to use these descriptors for Machine Learning pipelines and projects.

[**Eulearning.**](https://github.com/vadimlebovici/eulearning) This library features different approaches for computing and using the Euler Characteristic of a multiparameter filtration [@eulearning].
Although relying on distinct methods, `multipers` can also be used to compute Machine Learning descriptors from the Euler Characteristic, i.e., the Euler Decomposition Signed Barcode, or Euler Surfaces. Moreover, `multipers` computations are faster (especially on point cloud datasets), easier to use, and available on a wider range of multi-filtrations. 
<!--, as this library does not integrate with external libraries.-->

<!-- [**Rivet.**](https://github.com/rivetTDA/rivet): Rivet is a library for interactively computing 1-parameter persistence slices, and useful for visualizations. Its theory is developped in [@rivet]. -->
<!-- This library is also the core backend of most of the following libraries relying only on one parameter slices. -->
<!-- However, this library doesn't leverage on recently developped significant optimizations ([@filtration_domination], [@mpfree]), takes only very specific text-file inputs, doesn't support general multi-critical filtrations, and can only compute 1-parameter slices.  -->
<!-- Furthermore, although rivet can be used to compute some very specific multiparameter invariants, such as MPL, it needs other libraries to do so. -->

[**Multiparameter Persistent Landscape.**](https://github.com/OliverVipond/Multiparameter_Persistence_Landscapes) Implemented on top of [Rivet](https://github.com/rivetTDA/rivet) [@rivet], this library computes a multiparameter persistent descriptor by computing 1-parameter persistence landscape vectorizations of slices in multi-filtrations [@mpl], called Multiparameter Persistent Landscape (MPL). This library also features some multiparameter persistence visualizations. However, it is limited to `Rivet` capabilities and landscapes computations, which on one hand does not leverage on recently developped optimizations, e.g., [@filtration_domination], or [@mpfree], and on the other hand can only work with very specific text file inputs.

<!--**Multiparameter Persistent Kernel.** Similar to landscapes, this invariant can be computed on diagonal persistence slices given by, e.g., `Rivet`.-->

[**GRIL.**](https://github.com/TDA-Jyamiti/GRIL) This library provides code to compute a specific, generalized version of the Multiparameter Persistent Landscapes [@gril], relying on 1-paramter persistence zigzag computations. This library however is limited to this invariant, can only deal with 2-parameter persistence, and is not as much integrated as `multipers` with other multiparameter persistence and Machine Learning libraries.

[**Elder Rule Staircode.**](https://github.com/Chen-Cai-OSU/ER-staircode) This library features a descriptor for 2-parameter, degree-0 homology, rips-densitity-like filtrations [@erstaircode]. Once again, this library is very specific and not linked with other libraries.

[**Persistable.**](https://github.com/LuisScoccola/persistable) is a GUI interactive library for clustering, using degree-0 multiparameter persistence [@code_persistable] [@persistable]. 
Although aiming at distinct goals and using very different approaches, `multipers` can also be used for clustering, by computing (differentiable) descriptors that can be used afterward with standard clustering methods, e.g., K-means. 

We contribute to this variety of task-specific libraries by providing a **general purpose** library, `multipers`, with novel and efficient topological invariant computations, integrated state-of-the art Machine Learning topological pipelines, and interfaces to standard Machine Learning and Deep Learning libraries.


# Acknowledgements
 David Loiseaux was supported by ANR grant 3IA Côte d’Azur
(ANR-19-P3IA-0002).
The authors would like to thank 
 Mathieu Carrière, and Luis Scoccola
 for their help on Sliced Wasserstein, and Möbius inversion code.

# References

