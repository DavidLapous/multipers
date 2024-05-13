# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "multipers"
copyright = "2024, David Loiseaux, Mathieu Carrière, Hannah Schreiber"
author = "David Loiseaux, Hannah Schreiber"
release = "2.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.duration",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "myst_nb",
    "sphinxcontrib.bibtex",
]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_favicon = 'icon.svg'

# -- Notebooks
nb_execution_timeout = 10
nb_execution_mode = "off"


# -- biblio
bibtex_bibfiles = ["paper.bib"]
bibtex_encoding = "latin"
