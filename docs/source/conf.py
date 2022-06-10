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

sys.path.insert(0, os.path.abspath("../src/menelaus"))


# -- Project information -----------------------------------------------------

project = "menelaus"
copyright = "©2022 The MITRE Corporation. ALL RIGHTS RESERVED"
author = "Leigh Nicholl, Thomas Schill, India Lindsay, Anmol Srivastava, Kodie P McNamara, Austin Downing"
release = "0.1.0"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinxcontrib.bibtex",
]

autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "private-members": False,
    "special-members": "__init__",
}

bibtex_bibfiles = ["refs.bib"]
bibtex_reference_style = "author_year"

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "custom_theme"
html_theme_path = ["."]
html_sidebars = {
    "**": ["fulltoc.html", "relations.html", "sourcelink.html", "searchbox.html"]
}
html_theme_options = {
    "navigation_depth": 5,
    "collapse_navigation": False,
    # "collapse": False,
}

html_css_files = [
    "custom.css",
]

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]


# Get sphinx-apidocs to run on readthedocs pipeline
# see https://github.com/readthedocs/readthedocs.org/issues/1139
def run_apidoc(_):
    from sphinx.ext.apidoc import main
    import os
    import sys

    os.chdir("..")
    print("cwd", os.getcwd(), "\n")
    print("contents", os.listdir())
    src_dir = os.path.join("../src/menelaus")
    template_dir = os.path.join("source", "templates")
    main(["-M", "--templatedir", template_dir, "-f", "-o", "source", src_dir])


def setup(app):
    app.connect("builder-inited", run_apidoc)
