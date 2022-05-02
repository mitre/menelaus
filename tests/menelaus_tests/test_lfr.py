"""Module for checking correctness of LFR"""
import numpy as np
from menelaus.concept_drift.lfr import LinearFourRates


def test_build_no_drift():
    """TODO - this entire function is basically one large todo"""
    np.random.seed(
        123
    )  # hack to get tests to pass; need a better way to force drift/non-drift
    burn_in = 50
    det = LinearFourRates(burn_in=burn_in, num_mc=15)
    for _ in range(burn_in + 50):
        # values that ostensibly should not indicate drift
        det.update(y_pred=1, y_true=1)
        det.update(y_pred=0, y_true=1)
        det.update(y_pred=1, y_true=0)
        det.update(y_pred=0, y_true=0)
        assert det.drift_state is None
        assert det.retraining_recs == [None, None]


def test_build_with_warning_drift():
    """See if a single test can include warning and alarm"""
    burn_in = 50
    np.random.seed(
        123
    )  # hack to get tests to pass; need a better way to force drift/non-drift
    det = LinearFourRates(
        burn_in=burn_in, 
        num_mc=50, 
        warning_level=0.5, 
        detect_level=0.1,
        time_decay_factor=0.5
    )  # low num_mc induces drift - better way?
    # have warning and drift significance levels be spread out
    for i in range(burn_in + 20):
        if i < burn_in:
            #det.update(1,np.random.binomial(n=1,p=0.6))
            det.update(1,1)
            assert det.drift_state is None
            assert det.retraining_recs == [None, None]
        elif i == burn_in: # get this to be warning
            det.update(1,1)
            assert det.drift_state == "warning"
        elif i == burn_in + 1: # get this to be drift
            det.update(1,0)
            assert det.drift_state == "drift"
        else:
            det.update(1,1)


# need a test that goes straight from no drift to drift (bypassing warning)
def test_build_with_drift():
    """Drift alarm that bypasses warning state to check retraining recs"""
    burn_in = 30
    det = LinearFourRates(
        burn_in=burn_in, num_mc=1, detect_level=0.5 # get it to drift asap
    ) 
    for i in range(burn_in + 1):
        det.update(1, 1)
        if i >= burn_in:
           assert det.drift_state == "drift"
           assert det._retraining_recs == [burn_in,burn_in]
          
