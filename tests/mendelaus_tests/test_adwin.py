"""Methods for checking simple behaviors of ADWIN."""
import pytest
from mendelaus.concept_drift.adwin import ADWIN


def test_compression():
    """Check that ADWIN.compress_buckets maintains _BucketRows within the
    max_buckets param
    """
    max_buckets = 1
    det = ADWIN(max_buckets=max_buckets, delta=0.000001)
    n_samples = 10
    for element in [1, 2, 3]:
        for _ in range(n_samples):
            det.update(element)
    curr = det._bucket_row_list.head

    overflow = False
    while curr.next_bucket is not None:
        overflow = (curr.bucket_count > max_buckets) or overflow
        curr = curr.next_bucket

    assert overflow is False


def test_validation():
    """Validation on delta parameter should raise ValueError"""
    with pytest.raises(ValueError) as _:
        _ = ADWIN(delta=25)


def test_mean():
    """Check that ADWIN.mean returns sane values"""
    det = ADWIN(new_sample_thresh=2)
    assert det.mean() == 0
    n_samples = 5
    for _ in range(n_samples):
        det.update(1)
    assert det.mean() == 1


def test_variance():
    """Check that ADWIN.variance returns sane values"""
    det = ADWIN(new_sample_thresh=2)
    assert det.variance() == 0
    n_samples = 10
    for _ in range(n_samples):
        det.update(1)
        det.update(0)
    assert det.variance() == 0.25


def test_drift():
    """Check that, for a trivially drifting data stream:
    - drift is identified
    - ADWIN.retraining_recs takes on the proper values before and after drift
    """
    det = ADWIN(new_sample_thresh=2)
    n_samples = 20
    for _ in range(n_samples):
        det.update(0.1)

    drift_found = False
    for _ in range(n_samples):
        det.update(100)
        drift_found = (det.drift_state == "drift") or drift_found
        if drift_found:
            break
    # confirm drift is found with trivially different stream
    assert drift_found is True
    # better refactored as a separate test with determined, desired window
    # but for now, just confirm that it's populated as part of the normal flow
    assert det.retraining_recs is not None

    # do we reset properly?
    det.update(100)
    assert det.drift_state is None
    assert det.retraining_recs == [None, None]


def test_conservative_bound():
    """Confirm that, using conservative_bound param, drift is found in a
    trivially drifting stream.
    """
    det = ADWIN(new_sample_thresh=2, conservative_bound=True)
    n_samples = 20
    for _ in range(n_samples):
        det.update(0.1)

    drift_found = False
    for _ in range(n_samples):
        det.update(100)
        drift_found = (det.drift_state == "drift") or drift_found
        if drift_found:
            break
    # confirm drift is found with trivially different stream
    assert drift_found is True
