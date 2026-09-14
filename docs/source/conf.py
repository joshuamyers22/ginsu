import os
import sys
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version

sys.path.insert(0, os.path.abspath("../../"))

# -- Project information

project = "Ginsu"
copyright = "2022 DataDome contributors; 2026 Ginsu contributors"
author = "Ginsu contributors"

try:
    release = package_version("ginsu")
except PackageNotFoundError:
    release = "development"
version = release

# -- General configuration

extensions = [
    "sphinx.ext.duration",
    "sphinx.ext.doctest",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
]

templates_path = ["_templates"]

# -- HTML output -----------------------------------------------------------

html_theme = "sphinx_rtd_theme"
html_title = f"Ginsu {release} documentation"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_theme_options = {
    "collapse_navigation": False,
    "navigation_depth": 3,
    "titles_only": False,
}

# -- Options for EPUB output
epub_show_urls = "footnote"
