import pandas as pd

from menelaus.ensemble import StreamingEnsemble, BatchEnsemble
from menelaus.ensemble import (
    SimpleMajorityElection,
    MinimumApprovalElection,
    OrderedApprovalElection,
    ConfirmedElection
)
from menelaus.data_drift import KdqTreeBatch
from menelaus.concept_drift import STEPD
from menelaus.change_detection import ADWIN


# region - test stream ensemble

def test_stream_ensemble_1():
    """Ensure stream ensemble executes with no drift"""
    step1 = STEPD(window_size=2)
    step2 = STEPD(window_size=2)
    step3 = STEPD(window_size=2)
    se = StreamingEnsemble(
        detectors={"s1": step1, "s2": step2, "s3": step3},
        election=SimpleMajorityElection(),
    )
    df = pd.DataFrame({"a": [0, 0], "b": [0, 0], "c": [0, 0]})
    se.update(X=df.iloc[[0]], y_true=0, y_pred=0)
    assert se.drift_state == None

def test_stream_ensemble_2():
    """Ensure stream ensemble executes when columns specified"""
    step1 = STEPD(window_size=2)
    step2 = STEPD(window_size=2)
    step3 = STEPD(window_size=2)

    column_selectors = {
        "s1": lambda X: X[["a"]],
        "s2": lambda X: X[["b", "c"]]
    }

    se = StreamingEnsemble(
        detectors={"s1": step1, "s2": step2, "s3": step3},
        election=SimpleMajorityElection(),
        column_selectors=column_selectors
    )

    df = pd.DataFrame({"a": [0,0], "b": [0,0], "c": [0,0]})
    se.update(X=df.iloc[[0]], y_true=0, y_pred=0)
    assert se.drift_state == None

def test_stream_ensemble_3():
    """Ensure stream ensemble executes with univariate data"""
    adwin1 = ADWIN()
    adwin2 = ADWIN()
    adwin3 = ADWIN()

    column_selectors = {
        "a1": lambda X: X[["a"]],
        "a2": lambda X: X[["b"]],
        "a3": lambda X: X[["c"]]
    }

    se = StreamingEnsemble(
        detectors={"a1": adwin1, "a2": adwin2, "a3": adwin3},
        election=SimpleMajorityElection(),
        column_selectors=column_selectors
    )

    df = pd.DataFrame({"a": [0,0], "b": [0,0], "c": [0,0]})
    se.update(X=df.iloc[[0]], y_true=0, y_pred=0)
    assert se.drift_state == None

def test_stream_ensemble_reset_1():
    """Ensure reset works in stream ensemble and its detectors"""
    step1 = STEPD(window_size=2)
    step2 = STEPD(window_size=2)
    column_selectors = {
        "s1": lambda X: X[["a"]],
        "s2": lambda X: X[["b", "c"]]
    }
    be = StreamingEnsemble(
        detectors={"s1": step1, "s2": step2},
        election=SimpleMajorityElection(),
        column_selectors=column_selectors
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

# endregion

# region - general tests

def test_ensemble_drift_states_1():
    """ Check member drift states are correctly reported by attribute """
    adwin1 = ADWIN()
    adwin2 = ADWIN()
    adwin3 = ADWIN()

    se = StreamingEnsemble(
        detectors={"a1": adwin1, "a2": adwin2, "a3": adwin3},
        election=SimpleMajorityElection(),
        column_selectors={}
    )

    se.detectors['a1'].drift_state = "drift"
    se.detectors['a2'].drift_state = "warning"    
    assert se.drift_states == {"a1": "drift", "a2": "warning", "a3": None}

def test_ensemble_recs_1():
    """ Check member retraining recs are correctly reported by attribute """
    adwin1 = ADWIN()
    adwin2 = ADWIN()
    k1 = KdqTreeBatch()

    se = StreamingEnsemble(
        detectors={"a1": adwin1, "a2": adwin2, "k1": k1},
        election=SimpleMajorityElection(),
        column_selectors={}
    )

    se.detectors['a1']._retraining_recs = "PLACEHOLDER VALUE"
    se.detectors['a2']._retraining_recs = "PLACEHOLDER VALUE"  
    assert not hasattr(k1, "retraining_recs")  
    assert se.retraining_recs == {"a1": "PLACEHOLDER VALUE", "a2": "PLACEHOLDER VALUE"}

# endregion

# region - test batch ensemble 

def test_batch_ensemble_1():
    """Ensure batch ensemble executes with no drift"""
    kdq1 = KdqTreeBatch(bootstrap_samples=1)
    kdq2 = KdqTreeBatch(bootstrap_samples=1)
    kdq3 = KdqTreeBatch(bootstrap_samples=1)
    be = BatchEnsemble(
        detectors={"k1": kdq1, "k2": kdq2, "k3": kdq3},
        election=SimpleMajorityElection(),
    )
    df = pd.DataFrame(
        {
            "a": [0, 10.0, 11.0, 12.0],
            "b": [0, 11.0, 12.0, 13.0],
            "c": [0, 12.0, 13.0, 14.0],
        }
    )
    be.set_reference(df.loc[:1])
    be.update(df.loc[2:])
    assert be.drift_state == None

def test_batch_ensemble_2():
    """Ensure batch ensemble executes when columns specified"""
    kdq1 = KdqTreeBatch(bootstrap_samples=1)
    kdq2 = KdqTreeBatch(bootstrap_samples=1)
    column_selectors = {
        "k1": lambda X: X[["a", "b"]],
        "k2": lambda X: X[["b", "c"]]
    }
    be = BatchEnsemble(
        detectors={"k1": kdq1, "k2": kdq2},
        election=SimpleMajorityElection(),
        # XXX - forcing >1 columns to satisfy KdqTree Batch
        column_selectors=column_selectors,
    )
    df = pd.DataFrame(
        {
            "a": [0, 10.0, 11.0, 12.0],
            "b": [0, 11.0, 12.0, 13.0],
            "c": [0, 12.0, 13.0, 14.0],
        }
    )
    be.set_reference(df.loc[:1])
    be.update(df.loc[2:])
    assert len(be.detectors["k1"]._input_cols) == 2
    assert len(be.detectors["k2"]._input_cols) == 2

def test_batch_ensemble_reset_1():
    """Ensure reset works in batch ensemble and its detectors"""
    kdq1 = KdqTreeBatch(bootstrap_samples=1)
    kdq2 = KdqTreeBatch(bootstrap_samples=1)
    column_selectors = {
        "k1": lambda X: X[["a", "b"]],
        "k2": lambda X: X[["b", "c"]]
    }
    be = BatchEnsemble(
        detectors={"k1": kdq1, "k2": kdq2},
        election=SimpleMajorityElection(),
        column_selectors=column_selectors,
    )
    df = pd.DataFrame(
        {
            "a": [0, 10.0, 11.0, 12.0],
            "b": [0, 11.0, 12.0, 13.0],
            "c": [0, 12.0, 13.0, 14.0],
        }
    )
    be.set_reference(df.loc[:1])
    be.update(df.loc[2:])
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

# endregion

# region - election tests

def test_eval_simple_majority_1():
    """Ensure simple majority scheme can identify drift"""
    det1 = det2 = KdqTreeBatch()
    det1.drift_state = det2.drift_state = "drift"
    det3 = KdqTreeBatch()
    det3.drift_state = None
    election = SimpleMajorityElection()
    assert election([det1, det2, det3]) == "drift"

def test_eval_simple_majority_2():
    """Ensure simple majority scheme does not false alarm"""
    det1 = det2 = KdqTreeBatch()
    det1.drift_state = det2.drift_state = None
    det3 = KdqTreeBatch()
    det3.drift_state = "drift"
    election = SimpleMajorityElection()
    assert election([det1, det2, det3]) == None

def test_eval_min_election_1():
    """Ensure minimimum approval scheme can identify drift"""
    s1 = s2 = STEPD()
    s1.drift_state = s2.drift_state = "drift"
    s3 = STEPD()
    s3.drift_state = None
    election = MinimumApprovalElection(approvals_needed=2)
    assert election([s1, s2, s3]) == "drift"

def test_eval_min_election_2():
    """Ensure minimimum approval scheme does not false alarm"""
    s1 = s2 = STEPD()
    s1.drift_state = s2.drift_state = None
    s3 = STEPD()
    s3.drift_state = "drift"
    election = MinimumApprovalElection(approvals_needed=2)
    assert election([s1, s2, s3]) == None

def test_eval_ordered_election_1():
    """Ensure confirmed approval scheme can identify drift"""
    s1 = s2 = STEPD()
    s1.drift_state = s2.drift_state = "drift"
    s3 = STEPD()
    s3.drift_state = None
    election = OrderedApprovalElection(
        approvals_needed=1,
        confirmations_needed=1
    )
    assert election([s1, s2, s3]) == "drift"

def test_eval_ordered_election_2():
    """Ensure confirmed approval scheme does not false alarm"""
    s1 = s2 = STEPD()
    s1.drift_state = "drift"
    s3 = STEPD()
    s2.drift_state = s3.drift_state = None
    election = OrderedApprovalElection(
        approvals_needed=2,
        confirmations_needed=1
    )
    assert election([s1, s2, s3]) == None

def test_confirmed_election_1():
    """Ensure ConfirmedElection can detect drift"""
    d1 = ADWIN()
    d2 = ADWIN()
    d3 = STEPD()
    d1.drift_state = "drift"
    election = ConfirmedElection(sensitivity=2, wait_time=10)
    election([d1,d2,d3]) # call #1
    d2.drift_state = "drift"
    assert election([d1, d2, d3]) == "drift" # by call #2, drift

def test_confirmed_election_2():
    """Ensure ConfirmedElection can detect warnings"""
    d1 = ADWIN()
    d2 = ADWIN()
    d3 = STEPD()
    d1.drift_state = "drift"
    election = ConfirmedElection(sensitivity=2, wait_time=10)
    election([d1,d2,d3]) # call #1
    d2.drift_state = "warning"
    assert election([d1, d2, d3]) == "warning" # by call #2, warning

def test_confirmed_election_3():
    """Ensure ConfirmedElection does not false alarm"""
    d1 = ADWIN()
    d2 = ADWIN()
    d3 = STEPD()
    d1.drift_state = "drift"
    election = ConfirmedElection(sensitivity=2, wait_time=10)
    election([d1, d2, d3]) # call 1
    assert election([d1, d2, d3]) == None # no more drift, so call 2 : None

def test_confirmed_election_4():
    """Ensure resetting of wait period counters"""
    d1 = ADWIN()
    d1.drift_state = "drift"
    election = ConfirmedElection(sensitivity=1, wait_time=1)
    election([d1])
    assert election.wait_period_counters[0] == 1
    election([d1]) # wait period counter for d1 would be 2, so rest to 0
    assert election.wait_period_counters[0] == 0

# endregion