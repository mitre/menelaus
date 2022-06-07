"""Module for checking correctness of LFR"""
import numpy as np
from menelaus.concept_drift.lfr import LinearFourRates


def test_build_no_drift():
    np.random.seed(
        123
    )  # hack to get tests to pass; need a better way to force drift/non-drift
    burn_in = 50
    det = LinearFourRates(burn_in=burn_in, num_mc=15)
    detParallel = LinearFourRates(burn_in=burn_in, num_mc=15, parallelize=True)
    for _ in range(burn_in + 50):
        # values that ostensibly should not indicate drift
        det.update(y_pred=1, y_true=1)
        det.update(y_pred=0, y_true=1)
        det.update(y_pred=1, y_true=0)
        det.update(y_pred=0, y_true=0)
        assert det.drift_state is None
        assert det.retraining_recs == [None, None]

        # values that ostensibly should not indicate drift
        np.random.seed(123)
        detParallel.update(y_pred=1, y_true=1)
        detParallel.update(y_pred=0, y_true=1)
        detParallel.update(y_pred=1, y_true=0)
        detParallel.update(y_pred=0, y_true=0)
        assert detParallel.drift_state is None
        assert detParallel.retraining_recs == [None, None]


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
    detParallel = LinearFourRates(
        burn_in=burn_in, 
        num_mc=50, 
        warning_level=0.5, 
        detect_level=0.1,
        time_decay_factor=0.5,
        parallelize=True
    )
    # have warning and drift significance levels be spread out
    for i in range(burn_in + 20):
        if i < burn_in:
            #det.update(1,np.random.binomial(n=1,p=0.6))
            det.update(1,1)
            assert det.drift_state is None
            assert det.retraining_recs == [None, None]
            np.random.seed(123)
            detParallel.update(1,1)
            assert detParallel.drift_state is None
            assert detParallel.retraining_recs == [None, None]
        elif i == burn_in: # get this to be warning
            det.update(1,1)
            assert det.drift_state == "warning"
            np.random.seed(123)
            detParallel.update(1,1)
            assert detParallel.drift_state == "warning"
        elif i == burn_in + 1: # get this to be drift
            det.update(1,0)
            assert det.drift_state == "drift"
            np.random.seed(123)
            detParallel.update(1,0)
            assert detParallel.drift_state == "drift"
        else:
            det.update(1,1)
            np.random.seed(123)
            detParallel.update(1,1)


# need a test that goes straight from no drift to drift (bypassing warning)
def test_build_with_drift():
    """Drift alarm that bypasses warning state to check retraining recs"""
    burn_in = 30
    det = LinearFourRates(
        burn_in=burn_in, num_mc=1, detect_level=0.5 # get it to drift asap
    )
    detParallel = LinearFourRates(
        burn_in=burn_in, num_mc=1, detect_level=0.5, parallelize=True # get it to drift asap
    ) 
    for i in range(burn_in + 1):
        det.update(1, 1)
        detParallel.update(1, 1)
        if i >= burn_in:
            assert det.drift_state == "drift"
            assert det._retraining_recs == [burn_in,burn_in]
            assert detParallel.drift_state == "drift"
            assert detParallel._retraining_recs == [burn_in,burn_in]


def test_drift_no_rates_tracked():
    """Drift alarm that bypasses warning state to check retraining recs"""
    """Contains drift alarm but drift_state is not updated because no rates are being tracked"""
    burn_in = 30
    det = LinearFourRates(
        burn_in=burn_in, num_mc=1, detect_level=0.5, rates_tracked=[] # get it to drift asap
    )
    for i in range(burn_in + 1):
        det.update(1, 1)
        if i >= burn_in:
            assert det.drift_state is None
            assert det._retraining_recs == [None, None]


def test_drift_rate_dependence():
    """drift alert caused by PPV, but only warnings or no alerts caused by other rates"""
    burn_in = 50
    np.random.seed(
        123
    )  # hack to get tests to pass; need a better way to force drift/non-drift
    detTPR = LinearFourRates(
        burn_in=burn_in, 
        num_mc=50, 
        warning_level=0.5, 
        detect_level=0.1,
        time_decay_factor=0.5,
        rates_tracked=["tpr"]
    )  # low num_mc induces drift - better way?
    detTNR = LinearFourRates(
        burn_in=burn_in, 
        num_mc=50, 
        warning_level=0.5, 
        detect_level=0.1,
        time_decay_factor=0.5,
        rates_tracked=["tnr"]
    )
    detPPV = LinearFourRates(
        burn_in=burn_in, 
        num_mc=50, 
        warning_level=0.5, 
        detect_level=0.1,
        time_decay_factor=0.5,
        rates_tracked=["ppv"]
    )
    detNPV = LinearFourRates(
        burn_in=burn_in, 
        num_mc=50, 
        warning_level=0.5, 
        detect_level=0.1,
        time_decay_factor=0.5,
        rates_tracked=["npv"]
    )
    # have warning and drift significance levels be spread out
    for i in range(burn_in + 20):
        if i < burn_in:
            #det.update(1,np.random.binomial(n=1,p=0.6))
            detTPR.update(1,1)
            assert detTPR.drift_state is None
            assert detTPR.retraining_recs == [None, None]

            np.random.seed(123)
            detTNR.update(1,1)
            assert detTNR.drift_state is None
            assert detTNR.retraining_recs == [None, None]

            np.random.seed(123)
            detPPV.update(1,1)
            assert detPPV.drift_state is None
            assert detPPV.retraining_recs == [None, None]

            np.random.seed(123)
            detNPV.update(1,1)
            assert detNPV.drift_state is None
            assert detNPV.retraining_recs == [None, None]
        elif i == burn_in: # get this to be warning
            detTPR.update(1,1)
            assert detTPR.drift_state == "warning"

            np.random.seed(123)
            detTNR.update(1,1)
            assert detTNR.drift_state is None

            np.random.seed(123)
            detPPV.update(1,1)
            assert detPPV.drift_state == "warning"

            np.random.seed(123)
            detNPV.update(1,1)
            assert detNPV.drift_state is None
        elif i == burn_in + 1: # get this to be drift
            detTPR.update(1,0)
            assert detTPR.drift_state == "warning"

            np.random.seed(123)
            detTNR.update(1,0)
            assert detTNR.drift_state == "warning"

            np.random.seed(123)
            detPPV.update(1,0)
            assert detPPV.drift_state == "drift"

            np.random.seed(123)
            detNPV.update(1,0)
            assert detNPV.drift_state is None
        else:
            detTPR.update(1,1)
            np.random.seed(123)
            detTNR.update(1,1)
            np.random.seed(123)
            detPPV.update(1,1)
            np.random.seed(123)
            detNPV.update(1,1)
          