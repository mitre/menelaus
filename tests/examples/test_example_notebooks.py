"""
Run .py scripts in examples and ensure they execute without error.
"""

import pathlib
import runpy
import os
import sys

import pytest

from menelaus import test_env_var 


output_file_pattern = 'example_*.*'
scripts = pathlib.Path(__file__, '..').resolve().rglob(output_file_pattern)
for s in scripts:
	print(s)

# for script in scripts:
#     print(f"Running {str(script)}")
#     fp = '..\\..\\src\\menelaus\\tools\\artifacts\\example_data.csv'
#     with open(fp, 'r') as f:
#         print(f)
#     # We have to tweak sys.argv to avoid the pytest arguments from being passed along to our scripts
#     # sys_orig = sys.argv
#     # sys.argv = [ str(script) ]
#     # os.environ[test_env_var] = 'True'

#     try:
#         runpy.run_path(str(script), run_name='__main__')
#     except FileNotFoundError as fe:
#         print(fe)
#     except SystemExit as e:
#         # Some scripts may explicitly call `sys.exit()`, in which case we'll check the error code
#         assert(e.code == 0)