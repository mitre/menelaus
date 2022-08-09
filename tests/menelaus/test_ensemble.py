import pandas as pd

from menelaus.ensemble import Ensemble, BatchEnsemble
from menelaus.ensemble import eval_simple_majority
from menelaus.data_drift import KdqTreeBatch


def test_batch_ensemble_1():
    """Ensure batch ensemble executes with no drift"""
    kdq1 = KdqTreeBatch(bootstrap_samples=1)
    kdq2 = KdqTreeBatch(bootstrap_samples=1)
    kdq3 = KdqTreeBatch(bootstrap_samples=1)
    be = BatchEnsemble(
        detectors={"k1": kdq1, "k2": kdq2, "k3": kdq3}, evaluator="simple-majority"
    )
    df = pd.DataFrame({"a": [0, 0], "b": [0, 0], "c": [0, 0]})
    be.set_reference(df.loc[:0])
    be.update(df.loc[1:])
    assert be.drift_state == None


def test_batch_ensemble_2():
    """Ensure batch ensemble executes when columns specified"""
    kdq1 = KdqTreeBatch(bootstrap_samples=1)
    kdq2 = KdqTreeBatch(bootstrap_samples=1)
    be = BatchEnsemble(
        detectors={"k1": kdq1, "k2": kdq2},
        evaluator="simple-majority",
        columns={"k1": ["a"], "k2": ["b", "c"]},
    )
    df = pd.DataFrame({"a": [0, 10.0], "b": [0, 11.0], "c": [0, 12.0]})
    be.set_reference(df.loc[:0])
    be.update(df.loc[1:])
    assert len(be.detectors["k1"]._input_cols) == 1
    assert len(be.detectors["k2"]._input_cols) == 2


def test_batch_ensemble_reset_1():
    """Ensure reset works in batch ensemble and its detectors"""
    kdq1 = KdqTreeBatch(bootstrap_samples=1)
    kdq2 = KdqTreeBatch(bootstrap_samples=1)
    be = BatchEnsemble(
        detectors={"k1": kdq1, "k2": kdq2},
        evaluator="simple-majority",
        columns={"k1": ["a"], "k2": ["b", "c"]},
    )
    df = pd.DataFrame({"a": [0, 10.0], "b": [0, 11.0], "c": [0, 12.0]})
    be.set_reference(df.loc[:0])
    be.update(df.loc[1:])
    be.drift_state = "drift"
    be.detectors["k1"].drift_state = "drift"
    be.detectors["k2"].drift_state = "drift"
    be.reset()
    assert be.batches_since_reset == 0
    assert be.drift_state == None
    assert be.total_batches == 1
    for det_key in be.detectors:
        assert be.detectors[det_key].batches_since_reset == 0
        assert be.detectors[det_key].drift_state == None
        assert be.detectors[det_key].total_batches == 1


def test_eval_simple_majority_1():
    """Ensure simple majority scheme can identify drift"""
    det1 = det2 = KdqTreeBatch()
    det1.drift_state = det2.drift_state = "drift"
    det3 = KdqTreeBatch()
    det3.drift_state = None
    assert eval_simple_majority([det1, det2, det3]) == "drift"


def test_eval_simple_majority_2():
    """Ensure simple majority scheme does not false alarm"""
    det1 = det2 = KdqTreeBatch()
    det1.drift_state = det2.drift_state = None
    det3 = KdqTreeBatch()
    det3.drift_state = "drift"
    assert eval_simple_majority([det1, det2, det3]) == None
