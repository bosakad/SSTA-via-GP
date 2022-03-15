# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('..'))
import src.gateSizing

# -- Project information -----------------------------------------------------

project = 'Gate Sizing'
copyright = '2022, Adam Bosak'
author = 'Adam Bosak'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
	'sphinx.ext.autodoc',
	'sphinx.ext.napoleon',
	'sphinx.ext.viewcode',

]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

suppress_warnings = ['app.add_node',
                     'app.add_directive',
                     'app.add_role',
                     'app.add_generic_role',
                     'app.add_source_parser',
                     'download.not_readable',
                     'image.not_readable',
                     'ref.term',
                     'ref.ref',
                     'ref.numref',
                     'ref.keyword',
                     'ref.option',
                     'ref.citation',
                     'ref.footnote',
                     'ref.doc',
                     'misc.highlighting_failure',
                     'toc.secnum',
                     'epub.unknown_project_files'
			]

autodoc_mock_imports = ["randomVariableHist_Numpy", "histogramGenerator", "test_infiniteLadder", "parser", "node", "cvxpyVariable", "mosekVariable"]
