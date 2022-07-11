import os
from menelaus.utils._locate import find_git_root


def test_find_root_dir():
    assert find_git_root("garbage") is None
    assert ".git" in os.listdir(find_git_root())
