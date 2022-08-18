import pandas as pd

from menelaus.ensemble import StreamingEnsemble, BatchEnsemble, EVALUATORS
from menelaus.ensemble import eval_simple_majority, eval_confirmed_approval, eval_minimum_approval
from menelaus.data_drift import KdqTreeBatch
from menelaus.concept_drift import STEPD
from menelaus.change_detection import ADWIN


def test_stream_ensemble_1():
    """Ensure stream ensemble executes with no drift"""
    step1 = STEPD(window_size=2)
    step2 = STEPD(window_size=2)
    step3 = STEPD(window_size=2)
    se = StreamingEnsemble(
        detectors={"s1": step1, "s2": step2, "s3": step3},
        evaluator="simple-majority"
    )
    df = pd.DataFrame({"a": [0,0], "b": [0,0], "c": [0,0]})
    se.update(X=df.iloc[[0]], y_true=0, y_pred=0)
    assert se.drift_state == None

def test_stream_ensemble_2():
    """Ensure stream ensemble executes when columns specified"""
    step1 = STEPD(window_size=2)
    step2 = STEPD(window_size=2)
    step3 = STEPD(window_size=2)
    se = StreamingEnsemble(
        detectors={"s1": step1, "s2": step2, "s3": step3},
        evaluator=EVALUATORS["simple-majority"],
        columns={"s1": ["a"], "s2": ["b", "c"]}
    )
    df = pd.DataFrame({"a": [0,0], "b": [0,0], "c": [0,0]})
    se.update(X=df.iloc[[0]], y_true=0, y_pred=0)
    assert se.drift_state == None

def test_stream_ensemble_3():
    """Ensure stream ensemble executes with univariate data"""
    adwin1 = ADWIN()
    adwin2 = ADWIN()
    adwin3 = ADWIN()
    se = StreamingEnsemble(
        detectors={"a1": adwin1, "a2": adwin2, "a3": adwin3},
        evaluator=EVALUATORS["simple-majority"],
        columns={"a1": ["a"], "a2": ["b"], "a3":["c"]}
    )
    df = pd.DataFrame({"a": [0,0], "b": [0,0], "c": [0,0]})
    se.update(X=df.iloc[[0]], y_true=0, y_pred=0)
    assert se.drift_state == None

def test_stream_ensemble_reset_1():
    """Ensure reset works in stream ensemble and its detectors"""
    step1 = STEPD(window_size=2)
    step2 = STEPD(window_size=2)
    be = StreamingEnsemble(
        detectors={"s1": step1, "s2": step2},
        evaluator=EVALUATORS["simple-majority"],
        columns={"s1": ["a"], "s2": ["b", "c"]},
    )
    df = pd.DataFrame({"a": [0, 10.0], "b": [0, 11.0], "c": [0, 12.0]})
    be.update(df.loc[1:], 0, 0)
    be.drift_state = "drift"
    be.detectors["s1"].drift_state = "drift"
    be.detectors["s2"].drift_state = "drift"
    be.reset()
    assert be.samples_since_reset == 0
    assert be.drift_state == None
    assert be.total_samples == 1
    for det_key in be.detectors:
        assert be.detectors[det_key].samples_since_reset == 0
        assert be.detectors[det_key].drift_state == None
        assert be.detectors[det_key].total_samples == 1


def test_batch_ensemble_1():
    """Ensure batch ensemble executes with no drift"""
    kdq1 = KdqTreeBatch(bootstrap_samples=1)
    kdq2 = KdqTreeBatch(bootstrap_samples=1)
    kdq3 = KdqTreeBatch(bootstrap_samples=1)
    be = BatchEnsemble(
        detectors={"k1": kdq1, "k2": kdq2, "k3": kdq3},
        evaluator=EVALUATORS["simple-majority"]
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
        evaluator=EVALUATORS["simple-majority"],
        # XXX - forcing >1 columns to satisfy KdqTree Batch
        columns={"k1": ["a", "b"], "k2": ["b", "c"]},
    )
    df = pd.DataFrame({"a": [0, 10.0], "b": [0, 11.0], "c": [0, 12.0]})
    be.set_reference(df.loc[:0])
    be.update(df.loc[1:])
    assert len(be.detectors["k1"]._input_cols) == 2
    assert len(be.detectors["k2"]._input_cols) == 2

def test_batch_ensemble_reset_1():
    """Ensure reset works in batch ensemble and its detectors"""
    kdq1 = KdqTreeBatch(bootstrap_samples=1)
    kdq2 = KdqTreeBatch(bootstrap_samples=1)
    be = BatchEnsemble(
        detectors={"k1": kdq1, "k2": kdq2},
        evaluator=EVALUATORS["simple-majority"],
        columns={"k1": ["a", "b"], "k2": ["b", "c"]},
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

def test_eval_min_approval_1():
    """Ensure minimimum approval scheme can identify drift"""
    s1 = s2 = STEPD()
    s1.drift_state = s2.drift_state = "drift"
    s3 = STEPD()
    s3.drift_state = None
    assert eval_minimum_approval(approvals_needed=2)([s1, s2, s3]) == "drift"

def test_eval_min_approval_2():
    """Ensure minimimum approval scheme does not false alarm"""
    s1 = s2 = STEPD()
    s1.drift_state = s2.drift_state = None
    s3 = STEPD()
    s3.drift_state = "drift"
    assert eval_minimum_approval(approvals_needed=2)([s1, s2, s3]) == None

def test_eval_confirmed_approval_1():
    """Ensure confirmed approval scheme can identify drift"""
    s1 = s2 = STEPD()
    s1.drift_state = s2.drift_state = "drift"
    s3 = STEPD()
    s3.drift_state = None
    assert eval_confirmed_approval(approvals_needed=1)([s1, s2, s3]) == "drift"

def test_eval_confirmed_approval_2():
    """Ensure confirmed approval scheme does not false alarm"""
    s1 = s2 = STEPD()
    s1.drift_state = "drift"
    s3 = STEPD()
    s2.drift_state = s3.drift_state = None
    assert eval_confirmed_approval(approvals_needed=2)([s1, s2, s3]) == None
