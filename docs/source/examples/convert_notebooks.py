"""
This utility script converts all of the notebooks in the /docs/source/examples
directory into python scripts in the /examples/ directory, to avoid having to
modify the same code in both locations.
"""

import os
from menelaus.datasets.make_example_data import find_git_root

root_dir = find_git_root()
walk_dir = os.path.join(root_dir, "docs", "source", "examples")
out_dir = os.path.join(root_dir, "examples")

for subdir, dirs, files in os.walk(walk_dir):
    for file in files:
        if os.path.splitext(file)[1] == ".ipynb":
            in_file = os.path.join(subdir, file)
            command = f"jupyter nbconvert --to python {in_file} --output-dir {out_dir}"
            # print(f"nbconvert {in_file} \n\t to dir {out_dir}")
            os.system(command)
            # print("\n")
