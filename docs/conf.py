# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import multipers as mp

project = "multipers"
copyright = "2024, David Loiseaux, Mathieu Carrière, Hannah Schreiber"
author = "David Loiseaux, Hannah Schreiber"
version = mp.__version__

html_context = {
    "display_github": True,  # Integrate GitHub
    "github_user": "DavidLapous",  # Username
    "github_repo": "multipers",  # Repo name
    "github_version": "main",  # Version
    "conf_py_path": "/docs/",  # Path in the checkout to the docs root
}

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
html_favicon = "icon.svg"
autodoc_typehints = "description"


# -- Notebooks
nb_execution_timeout = 10
nb_execution_mode = "off"
myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "html_image",
]
myst_url_schemes = ("http", "https", "mailto")

# -- biblio
bibtex_bibfiles = ["paper.bib"]
bibtex_encoding = "latin"
