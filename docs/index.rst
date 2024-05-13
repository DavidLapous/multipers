`multipers` : Multiparameter Persistence for Machine Learning
===============================================================


This library focuses on providing easy to use and performant tools for
applied **multi**\parameter **pers**\istence.
It is also meant to be integrated in the `Gudhi Library <https://gudhi.inria.fr>`_, 
so you can expect to see some similar implementations, for instance
the `SimplexTreeMulti <notebooks/simplex_tree_by_hand.html>`_.

Quickstart 
************
`multipers` is available on `PyPI <https://pypi.org/project/multipers/>`_, install it using

.. code-block:: bash

  pip install multipers

some dependencies are needed. The following ones should be enough.

.. code-block:: bash

  conda create -n python311
  conda activate python311
  conda install python=3.11 numpy matplotlib gudhi scikit-learn scipy tqdm shapely -c conda-forge
  pip install filtration-domination pykeops --upgrade

Not working ? Try installing it `from source <compilation.html>`_.

You are now ready to compute invariants on multifiltered topological spaces !

Citation
********

Please cite `multipers` if using it in a research paper. You can use the key 

.. code-block:: bibtex

  @misc{multipers,
      author={Loiseaux, David and Schreiber, Hannah},
      title={`multipers` : Multiparameter Persistence for Machine Learning},
      year={2022},
      publisher={GitHub},
      journal={GitHub repository},
      howpublished={\url{https://github.com/DavidLapous/multipers}}
  }

For theoretical references,
 - Module Approximation :cite:p:`mma`
 - Module Decomposition Representations :cite:p:`mma_vect`
 - Signed Barcodes as Signed Measures :cite:p:`sb_as_sm`

.. toctree::
  :caption: Introduction

  notebooks/multipers_intro
  notebooks/simplex_tree_by_hand
  compilation
  bibliography

.. toctree::
  :caption: Example zoo:
  
  notebooks/time_series_classification
  notebooks/graph_classification
  notebooks/molecular_embedding
  notebooks/degree_rips_interface

.. toctree::
  :caption: Differentiation

  notebooks/rips_density_autodiff
  notebooks/graph_autodiff

.. toctree::
  :caption: Modules and Functions tree
  :maxdepth: 2

  source/modules






Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`




