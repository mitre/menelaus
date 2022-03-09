from molten.concept_drift.stepd import STEPD


def test_warning():
    """Confirm STEPD enters warning region trivially"""
    det = STEPD(window_size=1, alpha_warning=0.6)
    det.update(y_pred=1, y_true=1)
    det.update(y_pred=0, y_true=1)
    assert det.drift_state == "warning"
    det.update(y_pred=1, y_true=1)
    assert det.drift_state is None


def test_drift():
    """Confirm STEPD enters drift region trivially"""
    det = STEPD(window_size=1, alpha_warning=0.7, alpha_drift=0.6)
    det.update(y_pred=1, y_true=1)
    det.update(y_pred=0, y_true=1)
    assert det.drift_state == "drift"
    det.update(y_pred=1, y_true=1)
    assert det.drift_state is None


def test_retraining_recs():
    """
    Confirm behavior of retraining_recs:
    1. needs to grow to the right as the detector remains in 'warning' or 'drift'
    2. needs to reset to current samples as detector leaves warning/drift
    """

    det = STEPD(window_size=5, alpha_warning=0.4, alpha_drift=0.24)

    n_window = 5
    n_errors = 3
    total_samples = 0
    warning_ix = None

    for _ in range(n_window):
        total_samples += 1
        det.update(y_pred=1, y_true=1)

        total_samples += 1
        det.update(y_pred=0, y_true=1)
    for _ in range(n_errors):
        total_samples += 1
        det.update(y_pred=0, y_true=1)

        # the detector should only enter the warning region once and stay there
        if det.drift_state is not None and warning_ix is None:
            warning_ix = total_samples - 1

    # should enter drift region after this sample
    total_samples += 1
    det.update(y_pred=1, y_true=1)
    assert det.retraining_recs == [warning_ix, total_samples - 1]

    # should reset the detector and retraining recs
    total_samples += 1
    det.update(y_pred=1, y_true=1)
    assert det.retraining_recs == [None, None]


def test_accuracies():
    """Confirm reasonable values for STEPD accuracy functions"""
    det = STEPD(window_size=2)
    assert det.recent_accuracy() == 0
    assert det.past_accuracy() == 0
    assert det.overall_accuracy() == 0

    det.update(y_pred=0, y_true=1)
    det.update(y_pred=0, y_true=1)
    det.update(y_pred=1, y_true=1)
    det.update(y_pred=1, y_true=1)

    assert det.recent_accuracy() == 1
    assert det.past_accuracy() == 0
    assert det.overall_accuracy() == 0.5
