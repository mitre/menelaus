"""Methods for testing correctness of DDM."""
from menelaus.concept_drift.ddm import DDM


def test_drift():
    """Check that DDM enters the drift region in a trivial stream."""
    det = DDM(n_threshold=2)
    n_samples = 2
    for _ in range(n_samples):
        det.update(1, 1)

    # drift and reset
    assert det.drift_state == "drift"
    det.update(1, 1)
    assert det.drift_state is None


def test_warning():
    """Check that DDM enters the warning region in a trivial stream."""
    det = DDM(n_threshold=10, warning_scale=1.1, drift_scale=1.5)

    n_correct_samples = 5
    n_error_samples = 6
    for _ in range(n_correct_samples):
        det.update(1, 1)

    # enter the "warning" region
    for _ in range(n_error_samples):
        det.update(1, 0)
    assert det.drift_state == "warning"

    # leave the "warning" region
    for _ in range(1):
        det.update(1, 1)
    assert det.drift_state is None


def test_retraining_recs_warning():
    """Check that EDDM's retraining_recs take correct values in warning region"""
    det = DDM()
    n_samples = 3
    for _ in range(n_samples):
        det.update(1, 1)
    det.drift_state = "warning"
    det._increment_retraining_recs()
    assert det.retraining_recs == [n_samples - 1, None]


def test_retraining_recs_drift():
    """Check that DDM's retraining_recs take correct values in drift region"""
    det = DDM()
    n_samples = 3
    for _ in range(n_samples):
        det.update(1, 1)
    det.drift_state = "drift"
    det._increment_retraining_recs()
    assert det.retraining_recs == [n_samples - 1, n_samples - 1]


def test_retraining_recs_reset():
    """Check that DDM's retraining_recs are reset properly"""
    det = DDM()
    n_samples = 3
    for _ in range(n_samples):
        det.update(1, 1)
    det.drift_state = "drift"
    det._initialize_retraining_recs()
    assert det.retraining_recs == [None, None]
