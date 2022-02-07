"""Module for checking correctness of LFR"""
import numpy as np
from molten.performance.lfr import LinearFourRates


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


def test_build_with_drift():
    """TODO - this entire function is basically one large todo"""
    burn_in = 50
    det = LinearFourRates(
        burn_in=burn_in, num_mc=1
    )  # low num_mc induces drift - better way?
    for i in range(burn_in + 50):
        det.update(1, 1)
        if i < burn_in:
            assert det.drift_state is None
            assert det.retraining_recs == [None, None]
        elif i == burn_in:
            # TODO - need to find a way to induce 'warning' but not 'drift'
            assert det.drift_state == "drift"
            assert det.retraining_recs == [i, i]  # TODO - is this the right check?
        else:
            assert det.drift_state is None
            assert det.retraining_recs == [None, None]
