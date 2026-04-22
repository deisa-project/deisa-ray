import os
import sys

sys.path.insert(0, os.path.abspath("../../src"))

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "deisa-ray"
copyright = "2025, Andres Bermeo Marinelli"
author = "Andres Bermeo Marinelli"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",  # enable numpy and google docstring support
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
]
autosummary_generate = True
napoleon_numpy_docstring = True

templates_path = ["_templates"]
exclude_patterns = [
    # sphinx-apidoc creates a wrapper page for the PEP 420 namespace root.
    # The public API entry point for this project is deisa.ray.
    "deisa.rst",
    # Older apidoc runs may have left the generated top-level toctree behind.
    "modules.rst",
    # Ignore stale API files from older apidoc runs that treated src/deisa/ray
    # as a top-level ray package and accidentally imported upstream Ray.
    "ray*.rst",
]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_static_path = ["_static"] if os.path.isdir(os.path.join(os.path.dirname(__file__), "_static")) else []
