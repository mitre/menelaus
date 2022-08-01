from menelaus.ensemble import Ensemble, BatchEnsemble
from menelaus.ensemble import eval_simple_majority
from menelaus.data_drift import KdqTreeBatch


def test_batch_ensemble_1():
    ''''''
    assert False

def test_eval_simple_majority_1():
    '''Ensure simple majority scheme can identify drift'''
    det1 = det2 = KdqTreeBatch()
    det1.drift_state = det2.drift_state = 'drift'
    det3 = KdqTreeBatch()
    det3.drift_state = None
    assert eval_simple_majority([det1, det2, det3]) == 'drift'

def test_eval_simple_majority_2():
    '''Ensure simple majority scheme does not false alarm'''
    det1 = det2 = KdqTreeBatch()
    det1.drift_state = det2.drift_state = None
    det3 = KdqTreeBatch()
    det3.drift_state = 'drift'
    assert eval_simple_majority([det1, det2, det3]) == None