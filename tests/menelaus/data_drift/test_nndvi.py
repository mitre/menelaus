import numpy as np
import pytest

from menelaus.data_drift import NNDVI


def test_nndvi_init():
    ''' Test correct default initialization for NNDVI '''
    det = NNDVI()
    assert det.k_nn == 30
    assert det.sampling_times == 500
    assert det.alpha == 0.01
    assert det.batches_since_reset == 0
    assert det.drift_state is None

def test_nndvi_set_reference():
    ''' Assert NNDVI.set_reference works as intended '''
    det = NNDVI()
    ref = np.random.randint(0,5,(3,3))
    det.set_reference(ref)
    assert np.array_equal(ref, det.reference_batch)

def test_nndvi_update_1():
    ''' Ensure NNDVI can update with small random batches '''
    det = NNDVI(k_nn=3, sampling_times=10)
    det.set_reference(np.random.randint(0,5,(10,10)))
    det.update(X=np.random.randint(0,5,(10,10)))

def test_nndvi_update_2():
    ''' Ensure NNDVI can update with drift actions triggered '''
    det = NNDVI(k_nn=5, sampling_times=10)
    # XXX - AS added this method of forcing drift in NNDVI, which
    #       is otherwise hard to induce drift in, for small data
    #       examples. More stable alternatives may exist
    np.random.seed(123)
    det.set_reference(np.random.randint(0,5,(10,10)))
    det.update(X=np.random.randint(10,40,(10,10)))
    assert det.drift_state is not None

def test_nndvi_update_3():
    ''' Check NNDVI.update behavior after drift alarm '''
    det = NNDVI(k_nn=3)
    det.set_reference(np.random.randint(0,5,(5,5)))
    det.drift_state == 'drift'
    det.update(X=np.random.randint(0,5,(5,5)))
    assert det.drift_state is None

def test_nndvi_update_4():
    ''' Check failure when batch shapes don't match '''
    det = NNDVI()
    det.set_reference(np.random.randint(0,5,(5,6)))
    with pytest.raises(ValueError):
        det.update(np.random.randint(0,5,(5,5)))

def test_nndvi_reset():
    ''' Check NNDVI.reset works as intended '''
    det = NNDVI()
    det.batches_since_reset = 1
    det.drift_state = 'drift'
    det.reset()
    assert det.batches_since_reset == 0
    assert det.drift_state is None

def test_nndvi_compute_threshold_1():
    ''' Check NNDVI._compute_threshold works correctly '''
    det = NNDVI()
    # XXX - Hardcoded known example added by AS, in the future a
    #       dynamic way to test this function may be used
    np.random.seed(123)
    threshold = det._compute_drift_threshold(
        M_nnps=np.random.randint(0,5,(5,5)),
        v_ref=np.random.randint(0,2,5),
        v_test=np.random.randint(0,2,5),
        sampling_times=50,
        alpha=0.01
    )
    assert threshold >= 0.6 and threshold <= 0.7

def test_nndvi_compute_threshold_2():
    ''' Check NNDVI._compute_threshold executes for 0 case '''
    det = NNDVI()
    v = np.random.randint(0,2,5)
    det._compute_drift_threshold(
        M_nnps=np.random.randint(0,5,(5,5)),
        v_ref=v,
        v_test=v,
        sampling_times=25,
        alpha=0.01
    )