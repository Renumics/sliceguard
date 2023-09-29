# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
# import glob

sys.path.insert(0, os.path.abspath(os.path.join('..', '..', '.')))

# if 'VIRTUAL_ENV' in os.environ:
    # site_packages_glob = os.sep.join([
        # os.environ['VIRTUAL_ENV'],
        # 'lib', 'python3.10', 'site-packages', 'projectname-*py3.10.egg'])
    # site_packages = glob.glob(site_packages_glob)[-1]
    # sys.path.insert(0, site_packages)

project = 'sliceguard'
copyright = '2023, Daniel Klitzke, Tarek Wilkening'
author = 'Daniel Klitzke, Tarek Wilkening'
release = '0.0.18'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.autodoc', 'sphinx.ext.napoleon', 'myst_parser']

templates_path = ['_templates']
exclude_patterns = []
suppress_warnings = ['myst.header']


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']
