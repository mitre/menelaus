"""Methods for testing correctness of EDDM."""
from menelaus.concept_drift.eddm import EDDM


def test_drift():
    """Check that EDDM enters the drift region in a trivial stream."""
    det = EDDM(n_threshold=2)
    n_correct_samples = 2
    n_error_samples = 4
    for _ in range(n_correct_samples):
        det.update(1, 1)

    for _ in range(n_error_samples):
        det.update(y_true=0, y_pred=1)

    # drift and reset
    assert det.drift_state == "drift"
    det.update(y_true=0, y_pred=1)
    assert det.drift_state is None


def test_warning():
    """Check that EDDM enters the warning region in a trivial stream."""
    det = EDDM(n_threshold=2)
    n_correct_samples = 2
    n_error_samples = 3
    for _ in range(n_correct_samples):
        det.update(1, 1)

    # enter warning region
    for _ in range(n_error_samples):
        det.update(y_true=0, y_pred=1)
    assert det.drift_state == "warning"

    # increase distance between errors, leave warning region
    det.update(1, 1)
    det.update(y_true=1, y_pred=0)
    assert det.drift_state is None


def test_retraining_recs_warning():
    """Check that EDDM's retraining_recs take correct values in warning region"""
    det = EDDM()
    n_samples = 3
    for _ in range(n_samples):
        det.update(1, 1)
    det.drift_state = "warning"
    det._increment_retraining_recs()
    assert det.retraining_recs == [n_samples - 1, None]


def test_retraining_recs_drift():
    """Check that DDM's retraining_recs take correct values in drift region"""
    det = EDDM()
    n_samples = 3
    for _ in range(n_samples):
        det.update(1, 1)
    det.drift_state = "drift"
    det._increment_retraining_recs()
    assert det.retraining_recs == [n_samples - 1, n_samples - 1]


def test_retraining_recs_reset():
    """Check that DDM's retraining_recs are reset properly"""
    det = EDDM()
    n_samples = 3
    for _ in range(n_samples):
        det.update(1, 1)
    det.drift_state = "drift"
    det._initialize_retraining_recs()
    assert det.retraining_recs == [None, None]
