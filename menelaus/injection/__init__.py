"""This module contains functions with which to artificially inject drift into a dataset."""

from menelaus.injection.feature_manipulation import (
    feature_swap,
    feature_hide_and_sample,
)
from menelaus.injection.class_manipulation import (
    class_swap,
    class_join,
    class_dirichlet_shift,
    class_probability_shift,
)
from menelaus.injection.noise import brownian_noise, random_walk
