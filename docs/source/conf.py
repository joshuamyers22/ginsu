# Configuration file for the Sphinx documentation builder.
import os
import sys

sys.path.insert(0, os.path.abspath("../../"))

# -- Project information

project = "Ginsu"
copyright = "2022 DataDome contributors; 2026 Ginsu contributors"
author = "Ginsu contributors"

# -- General configuration

extensions = [
    "sphinx.ext.duration",
    "sphinx.ext.doctest",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
]

templates_path = ["_templates"]

# -- Options for EPUB output
epub_show_urls = "footnote"
