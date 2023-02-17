"""
This module contains classes with which to artificially inject drift into a dataset.
"""

from menelaus.injection.feature_manipulation import (
    FeatureSwapInjector,
    FeatureCoverInjector,
)
from menelaus.injection.injector import Injector
from menelaus.injection.label_manipulation import (
    LabelSwapInjector,
    LabelJoinInjector,
    LabelProbabilityInjector,
    LabelDirichletInjector,
)
from menelaus.injection.noise import BrownianNoiseInjector
