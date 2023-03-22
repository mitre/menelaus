import importlib.metadata

# version from setup.cfg
__version__ = importlib.metadata.version('menelaus')

# name of environment variable we use to check for test harness mode
test_env_var = "MENELAUS_TEST_MODE"
