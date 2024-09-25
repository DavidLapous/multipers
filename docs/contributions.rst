Contributions
#############

Feel free to contribute, report a bug, or ask for documentation
by `opening an issue <https://github.com/DavidLapous/multipers/issues>`_ or
sending me an email.

Example zoo
***********

There are not enough examples of utilization in the example zoo;
if you can, and have a new example in mind please fill an issue and open a pull request
or contact me via email to add it.

Building the documentation
==========================

These examples are build with :code:`sphinx`, with the :code:`sphinxcontrib-bibtex` and :code:`myst_nb` extensions 
to compile the notebooks that are located in :code:`docs/notebooks`, and that are in the list given in 
the block 
of the :code:`docs/index.rst` file


.. code-block:: rst

  .. toctree::
    :caption: Example zoo:


Then, the following commands build the documentation of multipers, in the :code:`_build/html` folder.


.. code-block:: sh

  sphinx-apidoc -f -o source ..
  make html
  # or 
  sh build_doc.sh

Feature additions
*****************

The :code:`multipers` library is meant to be a general multiparameter persistence library for machine learning;
if you find an invariant that is not implemented in multipers, and want to help me 
integrate it, your contribution is welcome!
This library doesn't have a stable API, so the preferable 
way to add features to this library is to contact me by email.


Bug reports
***********

If you find a bug somewhere in the library, or find that some feature are not
properly documented, please open an issue on GitHub.
Regarding errors, minimal examples are preferred. 
Also, please mention if you're willing to fix the issue with a pull request.

Discussions
***********

Other matters can be discussed in the 
`discussion tab of GitHub <https://github.com/DavidLapous/multipers/discussions>`_.


