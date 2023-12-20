import pytest
from menelaus.data_drift.stat_test import GenericDetector, CHIDetector, KSDetector, CVMDetector, FETDetector
import numpy as np
import pandas as pd

class MockRepresentation:
    """ The Mock series Serve as a testing class to test the GenericDetector is behaving normally """ 
    def fit(self, X):
        """ Simple mock behavior, maybe just return the input for simplicity """
        return X


def MockDivergence(rep_ref, rep_test):
        """ A simple mock that returns a fixed value or a simple calculation """
        return sum(rep_ref) - sum(rep_test)


def MockCritFunction(divergence, alpha = 0.05):
        """ Mock behavior that returns True or False based on a simple condition """
        return divergence > alpha

class TestGenericDetector:
    def test_initialization(self):
        """Test correct default initialization for GenericDetector"""
        detector = GenericDetector(representation=MockRepresentation(), 
                      divergence=MockDivergence, 
                      crit_function=MockCritFunction)
        assert detector is not None

    def test_set_reference(self):
        """Assert GenericDetector.set_reference works as intended"""
        detector = GenericDetector(representation=MockRepresentation(), 
                      divergence=MockDivergence, 
                      crit_function=MockCritFunction)
        reference_data = np.array([1, 2, 3])
        detector.set_reference(reference_data)
        assert np.array_equal(detector.reference, reference_data)

    def test_update_no_drift(self):
        """Check GenericDetector.update behavior with no drift alarm"""
        detector = GenericDetector(representation=MockRepresentation(), 
                      divergence=MockDivergence, 
                      crit_function=MockCritFunction)
        reference_data = np.array([1, 2, 3])
        new_data = np.array([1, 2, 3])
        detector.set_reference(reference_data)
        detector.update(new_data)
        assert detector.drift_state is None

    def test_update_with_drift(self):
        """Ensure GenericDetector can update with drift actions triggered"""
        detector = GenericDetector(representation=MockRepresentation(), 
                      divergence=MockDivergence, 
                      crit_function=MockCritFunction)
        reference_data = np.array([10, 20, 30])
        new_data = np.array([1, 2, 3])
        detector.set_reference(reference_data)
        detector.update(new_data)
        assert detector.drift_state is not None

    def test_update_reset(self):
        """Check GenericDetector.update behavior after drift alarm"""
        detector = GenericDetector(representation=MockRepresentation(), 
                      divergence=MockDivergence, 
                      crit_function=MockCritFunction)
        detector.set_reference([1,2,3])
        detector.drift_state = "drift"
        detector.update(X=[1,2,3])
        assert detector.drift_state is None 
    
class TestCHIDetector:
    def test_initialization(self):
        """ Test CHIDetector has a default initialization without parameters"""
        detector = CHIDetector()
        assert detector is not None

    def test_set_reference(self):
        """Assert CHIDetector.set_reference works as intended by filterning out Non-complied data"""
        detector = CHIDetector()
        reference_data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        with pytest.raises(ValueError):
            detector.set_reference(reference_data)

    def test_update_no_drift(self):
        """Check if CHIDetector will return no drift accurately"""
        detector = CHIDetector()
        reference_data = np.array([1, 2, 3])
        new_data = np.array([1, 2, 3])
        detector.set_reference(reference_data)
        detector.update(new_data)
        assert detector.drift_state is None

        
    def test_update_with_drift(self):
        """Check if CHIDetector can detect drift accurately"""
        detector = CHIDetector()
        reference_data = np.array(['OM','OO','OK'])
        new_data = np.array(['OK','OK','OK','OK','OK','OK','OK','OK'])
        detector.set_reference(reference_data)
        detector.update(new_data)
        assert detector.drift_state is not None

class TestKSDetector:
    def test_initialization(self):
        """ Test KSDetector has a default initialization without parameters"""
        detector = KSDetector()
        assert detector is not None

    def test_set_reference(self):
        """Assert KSDetector.set_reference works as intended by filterning out Non-complied data"""
        detector = KSDetector()
        reference_data = np.array(['OM','OO','OK'])
        with pytest.raises(ValueError):
            detector.set_reference(reference_data)

    def test_update_no_drift(self):
        """Check if KSDetector will return no drift accurately"""
        detector = KSDetector()
        reference_data = np.array([1, 2, 3])
        new_data = np.array([1, 2, 3])
        detector.set_reference(reference_data)
        detector.update(new_data)
        assert detector.drift_state is None 

        
    def test_update_with_drift(self):
        """Check if KSDetector can detect drift accurately"""
        detector = KSDetector()
        reference_data = np.array([1, 2, 3, 4, 5, 6, 7, 8])
        new_data = np.array([200,300,400])
        detector.set_reference(reference_data)
        detector.update(new_data)
        assert detector.drift_state is not None
        
class TestCVMDetector:
    def test_initialization(self):
        """ Test CVMDetector has a default initialization without parameters"""
        detector = CVMDetector()
        assert detector is not None

    def test_set_reference(self):
        """Assert CVMDetector.set_reference works as intended by filterning out Non-complied data"""
        detector = CVMDetector()
        reference_data = np.array(['OM','OO','OK'])
        with pytest.raises(ValueError):
            detector.set_reference(reference_data)

    def test_update_no_drift(self):
        """Check if CVMDetector will return no drift accurately"""
        detector = CVMDetector()
        reference_data = np.array([1, 2, 3])
        new_data = np.array([1, 2, 3])
        detector.set_reference(reference_data)
        detector.update(new_data)
        assert detector.drift_state is None 

        
    def test_update_with_drift(self):
        """Check if CVMDetector can detect drift accurately"""
        detector = CVMDetector()
        reference_data = np.array([1, 2, 3, 4, 5, 6, 7, 8])
        new_data = np.array([200,300,400])
        detector.set_reference(reference_data)
        detector.update(new_data)
        assert detector.drift_state is not None

class TestFETDetector:
    def test_initialization(self):
        """ Test FETDetector has a default initialization without parameters"""
        detector = FETDetector()
        assert detector is not None

    def test_set_reference(self):
        """Assert FETDetector.set_reference works as intended by filterning out Non-complied data"""
        detector = FETDetector()
        reference_data = np.array(['OM','OO','OK'])
        with pytest.raises(ValueError):
            detector.set_reference(reference_data)

    def test_update_no_drift(self):
        """Check if FETDetector will return no drift accurately"""
        detector = FETDetector()
        reference_data = np.array([1,1,1,1,1,0,0,0,0,0])
        new_data = np.array([1,0])
        detector.set_reference(reference_data)
        detector.update(new_data,alt = 'less')
        assert detector.drift_state is None 

        
    def test_update_with_drift(self):
        """Check if FETDetector can detect drift accurately"""
        detector = FETDetector()
        reference_data = np.array([1,1,1,1,1,0,0,0,0,0])
        new_data = np.array([1,0,0,0,0,0,0,0,0,0,0,0,0])
        detector.set_reference(reference_data)
        detector.update(new_data,alt = 'less')
        assert detector.drift_state is not None
