import pytest
from stat_test import GenericDetector, CHIDetector, KSDetector, CVMDetector, FETDetector
import numpy as np

def test_generic_detector_initialization():
    # Test initialization of GenericDetector
    representation = MockRepresentation()
    divergence = MockDivergence()
    crit_function = MockCritFunction()
    detector = GenericDetector(representation, divergence, crit_function)
    assert detector is not None
    
