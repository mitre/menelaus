import pytest
import numpy as np
import pandas as pd
from menelaus.data_drift import PSI

def test_psi_init():
    """Test correct default initialization for PSI"""
    det = PSI()
    assert det.eps == 1e-4
    assert det.threshold == 0.1
    assert det.batches_since_reset == 0
    assert det.drift_state is None
    
def test_psi_set_reference():
    """Assert PSI.set_reference works as intended"""
    det = PSI()
    ref = np.random.randint(0, 5, (100,1))
    det.set_reference(ref)
    assert (det.reference).ndim == 1
    
def test_psi_update_1():
    """Ensure PSI can update with small random batches"""
    det = PSI()
    det.set_reference(np.random.randint(0, 5, (100, 1)))
    det.update(X=np.random.randint(0, 5, (100, 1)))

def test_psi_update_2():
    """Ensure PSI can update with drift actions triggered"""
    det = PSI()
    np.random.seed(123)
    det.set_reference(np.random.randint(0, 5, (100, 1)))
    det.update(X=np.random.randint(10, 40, (100, 1)))
    assert det.drift_state is not None

def test_psi_update_3():
    """Check PSI.update behavior after drift alarm"""
    det = PSI()
    det.set_reference(np.random.randint(0, 5, (25, 1)))
    det._drift_state = "drift"
    det.update(X=np.random.randint(0, 5, (25, 1)))
    assert det.drift_state is None

def test_psi_reset():
    """Check psi.reset works as intended"""
    det = PSI()
    det.batches_since_reset = 1
    det.drift_state = "drift"
    det.reset()
    assert det.batches_since_reset == 0
    assert det.drift_state is None

    
def test_psi_compute_PSI():
    """Check psi._compute_threshold works correctly"""
    det = PSI()
    ref = np.random.randint(0,100,100)
    test = np.random.randint(0,100,100)
    det.set_reference(X = ref)
    det.update(X = test)
    threshold = det.PSI_value
    assert threshold >= 0 and threshold <= 1
