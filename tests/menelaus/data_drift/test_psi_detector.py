import pytest
import numpy as np
from menelaus.data_drift.psi_detector import PSI_Detector

def test_psi_init():
    """Test correct default initialization for PSI"""
    det = PSI_Detector()
    assert det.eps == 1e-4
    assert det.threshold == 0.1
    assert det.batches_since_reset == 0
    assert det.drift_state is None
    
def test_psi_set_reference():
    """Assert PSI.set_reference works as intended"""
    det = PSI_Detector()
    ref = np.random.randint(0, 5, (3, 3))
    det.set_reference(ref)
    assert np.array_equal(ref, det.reference_batch)
    
def test_psi_update_1():
    """Ensure PSI can update with small random batches"""
    det = PSI_Detector()
    det.set_reference(np.random.randint(0, 5, (10, 10)))
    det.update(X=np.random.randint(0, 5, (10, 10)))

def test_psi_update_2():
    """Ensure PSI can update with drift actions triggered"""
    det = PSI_Detector()
    # XXX - AS added this method of forcing drift in psi, which
    #       is otherwise hard to induce drift in, for small data
    #       examples. More stable alternatives may exist
    np.random.seed(123)
    det.set_reference(np.random.randint(0, 5, (10, 10)))
    det.update(X=np.random.randint(10, 40, (10, 10)))
    assert det.drift_state is not None

def test_psi_update_3():
    """Check PSI.update behavior after drift alarm"""
    det = PSI_Detector()
    det.set_reference(np.random.randint(0, 5, (5, 5)))
    det._drift_state = "drift"
    det.update(X=np.random.randint(0, 5, (5, 5)))
    assert det.drift_state is None

def test_psi_update_4():
    """Check failure when batch shapes don't match"""
    det = PSI_Detector()
    det.set_reference(np.random.randint(0, 5, (5, 6)))
    with pytest.raises(ValueError):
        det.update(np.random.randint(0, 5, (5, 5)))

def test_psi_reset():
    """Check psi.reset works as intended"""
    det = PSI_Detector()
    det.batches_since_reset = 1
    det.drift_state = "drift"
    det.reset()
    assert det.batches_since_reset == 0
    assert det.drift_state is None

    
def test_psi_compute_PSI():
    """Check psi._compute_threshold works correctly"""
    det = PSI_Detector()
    # XXX - Hardcoded known example added by AS, in the future a
    #       dynamic way to test this function may be used
    np.random.seed(123)
    threshold = det._PSI(
        v_ref=np.random.randint(0, 2, 5),
        v_test=np.random.randint(0, 2, 5),
    )
    assert threshold >= 0 and threshold <= 1
