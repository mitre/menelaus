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
from inspect import getsourcefile

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
    "nbsphinx",
    "sphinx.ext.mathjax",
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
exclude_patterns = ["_build", "**.ipynb_checkpoints"]


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

html_js_files = [
    "require.min.js",  # Add to your _static
    "custom.js",
]

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]


# -- Options for HTML output for notebooks -------------------------------------------------

# Always execute notebooks prior to conversion
nbsphinx_execute = "always"


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


# Get path to directory containing this file, conf.py.
DOCS_DIRECTORY = os.path.dirname(os.path.abspath(getsourcefile(lambda: 0)))

# Resolve the pandoc dependency for the runner.
# There may be better options now, since pandoc is available to GitHub actions,
# but I'm not sure how to make it talk with nbsphinx.
# https://stackoverflow.com/a/71585691
def ensure_pandoc_installed(_):
    import pypandoc

    # Download pandoc if necessary. If pandoc is already installed and on
    # the PATH, the installed version will be used. Otherwise, we will
    # download a copy of pandoc into docs/bin/ and add that to our PATH.
    pandoc_dir = os.path.join(DOCS_DIRECTORY, "bin")
    # Add dir containing pandoc binary to the PATH environment variable
    if pandoc_dir not in os.environ["PATH"].split(os.pathsep):
        os.environ["PATH"] += os.pathsep + pandoc_dir
    pypandoc.ensure_pandoc_installed(
        targetfolder=pandoc_dir,
        delete_installer=True,
    )


def setup(app):
    app.connect("builder-inited", run_apidoc)
    app.connect("builder-inited", ensure_pandoc_installed)
