import pytest
from menelaus.data_drift.stat_test import GenericDetector, CHIDetector, KSDetector, CVMDetector, FETDetector
import numpy as np
import pandas as pd

class MockRepresentation:
    def fit(self, X):
        # Simple mock behavior, maybe just return the input for simplicity
        return X


def MockDivergence(rep_ref, rep_test):
        # A simple mock that returns a fixed value or a simple calculation
        return sum(rep_ref) - sum(rep_test)


def MockCritFunction(divergence, alpha = 0.05):
        # Mock behavior that returns True or False based on a simple condition
        return divergence > alpha

class TestGenericDetector:
    def test_initialization(self):
        detector = GenericDetector(representation=MockRepresentation(), 
                      divergence=MockDivergence, 
                      crit_function=MockCritFunction)
        assert detector is not None

    def test_set_reference(self):
        detector = GenericDetector(representation=MockRepresentation(), 
                      divergence=MockDivergence, 
                      crit_function=MockCritFunction)
        reference_data = np.array([1, 2, 3])
        detector.set_reference(reference_data)
        assert np.array_equal(detector.reference, reference_data)

    def test_update_no_drift(self):
        detector = GenericDetector(representation=MockRepresentation(), 
                      divergence=MockDivergence, 
                      crit_function=MockCritFunction)
        reference_data = np.array([1, 2, 3])
        new_data = np.array([1, 2, 3])
        detector.set_reference(reference_data)
        detector.update(new_data)
        assert detector.drift_state is None  # No drift expected

    def test_update_with_drift(self):
        detector = GenericDetector(representation=MockRepresentation(), 
                      divergence=MockDivergence, 
                      crit_function=MockCritFunction)
        reference_data = np.array([10, 20, 30])
        new_data = np.array([1, 2, 3])
        detector.set_reference(reference_data)
        detector.update(new_data)
        assert detector.drift_state is not None  #drift expected

    def test_update_reset():
        """Check GenericDetector.update behavior after drift alarm"""
        detector = GenericDetector(representation=MockRepresentation(), 
                      divergence=MockDivergence, 
                      crit_function=MockCritFunction)
        detector.set_reference([1,2,3])
        detector.drift_state = 'drift'
        detector.reset()
        detector.update(X=[1,2,3])
        assert detector.drift_state is None 
    
class TestCHIDetector:
    def test_initialization(self):
        # Assuming CHIDetector has a default initialization without parameters
        detector = CHIDetector()
        assert detector is not None

    def test_set_reference(self):
        detector = CHIDetector()
        reference_data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        with pytest.raises(ValueError):
            detector.set_reference(reference_data)

    def test_update_no_drift(self):
        detector = CHIDetector()
        reference_data = np.array([1, 2, 3])
        new_data = np.array([1, 2, 3])
        detector.set_reference(reference_data)
        detector.update(new_data)
        assert detector.drift_state is None  # No drift expected

        
    def test_update_with_drift(self):
        detector = CHIDetector()
        reference_data = np.array(['OM','OO','OK'])
        new_data = np.array(['OK','OK','OK','OK','OK','OK','OK','OK'])
        detector.set_reference(reference_data)
        detector.update(new_data)
        assert detector.drift_state is not None

class TestKSDetector:
    def test_initialization(self):
        # Assuming KSDetector has a default initialization without parameters
        detector = KSDetector()
        assert detector is not None

    def test_set_reference(self):
        detector = KSDetector()
        reference_data = np.array(['OM','OO','OK'])
        with pytest.raises(ValueError):
            detector.set_reference(reference_data)

    def test_update_no_drift(self):
        detector = KSDetector()
        reference_data = np.array([1, 2, 3])
        new_data = np.array([1, 2, 3])
        detector.set_reference(reference_data)
        detector.update(new_data)
        assert detector.drift_state is None  # No drift expected

        
    def test_update_with_drift(self):
        detector = KSDetector()
        reference_data = np.array([1, 2, 3, 4, 5, 6, 7, 8])
        new_data = np.array([200,300,400])
        detector.set_reference(reference_data)
        detector.update(new_data)
        assert detector.drift_state is not None
        
class TestCVMDetector:
    def test_initialization(self):
        # Assuming CVMDetector has a default initialization without parameters
        detector = CVMDetector()
        assert detector is not None

    def test_set_reference(self):
        detector = CVMDetector()
        reference_data = np.array(['OM','OO','OK'])
        with pytest.raises(ValueError):
            detector.set_reference(reference_data)

    def test_update_no_drift(self):
        detector = CVMDetector()
        reference_data = np.array([1, 2, 3])
        new_data = np.array([1, 2, 3])
        detector.set_reference(reference_data)
        detector.update(new_data)
        assert detector.drift_state is None  # No drift expected

        
    def test_update_with_drift(self):
        detector = CVMDetector()
        reference_data = np.array([1, 2, 3, 4, 5, 6, 7, 8])
        new_data = np.array([200,300,400])
        detector.set_reference(reference_data)
        detector.update(new_data)
        assert detector.drift_state is not None

class TestFETDetector:
    def test_initialization(self):
        # Assuming FETDetector has a default initialization without parameters
        detector = FETDetector()
        assert detector is not None

    def test_set_reference(self):
        detector = FETDetector()
        reference_data = np.array(['OM','OO','OK'])
        with pytest.raises(ValueError):
            detector.set_reference(reference_data)

    def test_update_no_drift(self):
        detector = FETDetector()
        reference_data = np.array([1,1,1,1,1,0,0,0,0,0])
        new_data = np.array([1,0])
        detector.set_reference(reference_data)
        detector.update(new_data,alt = 'less')
        assert detector.drift_state is None  # No drift expected

        
    def test_update_with_drift(self):
        detector = FETDetector()
        reference_data = np.array([1,1,1,1,1,0,0,0,0,0])
        new_data = np.array([1,0,0,0,0,0,0,0,0,0,0,0,0])
        detector.set_reference(reference_data)
        detector.update(new_data,alt = 'less')
        assert detector.drift_state is not None
