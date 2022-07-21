"""This module for now contains a single function that finds the 
root of the git directory, so we can operate out of a known directory on the
various runners (local, github, readthedocs).

If you are tempted to add more functions to this location, and they aren't
related, they should probably live in a separate script, and potentially in
another module than utils entirely, depending on volume and purpose.
"""
import os


def find_git_root(search_dirs=(".git",)):
    """Find the root directory for the git repo, so that we don't have to
    fool with strange filepaths.
    """
    test_dir = os.getcwd()
    prev_dir, test_dir = None, os.path.abspath(test_dir)
    while prev_dir != test_dir:
        if any(os.path.isdir(os.path.join(test_dir, d)) for d in search_dirs):
            return test_dir
        prev_dir, test_dir = test_dir, os.path.abspath(
            os.path.join(test_dir, os.pardir)
        )
    return None
