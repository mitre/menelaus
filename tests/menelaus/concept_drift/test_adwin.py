"""Methods for checking simple behaviors of ADWIN."""
import pytest
from menelaus.concept_drift.adwin import ADWIN, _BucketRow, _BucketRowList


def test_compression():
    """Check that ADWIN.compress_buckets maintains _BucketRows within the
    max_buckets param
    """
    max_buckets = 1
    det = ADWIN(max_buckets=max_buckets, delta=0.000001)
    n_samples = 10
    for element in [(1, 0), (0, 0), (1, 1)]:
        for _ in range(n_samples):
            det.update(y_true=element[0], y_pred=element[1])
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
        det.update(y_true=1, y_pred=1)
    assert det.mean() == 1


def test_variance():
    """Check that ADWIN.variance returns sane values"""
    det = ADWIN(new_sample_thresh=2)
    assert det.variance() == 0
    n_samples = 10
    for _ in range(n_samples):
        det.update(y_true=1, y_pred=1)
        det.update(y_true=0, y_pred=1)
    assert det.variance() == 0.25


def test_drift():
    """Check that, for a very drifting data stream:
    - drift is identified
    - ADWIN.retraining_recs takes on the proper values before and after drift
    """
    det = ADWIN(new_sample_thresh=2)
    n_samples = 50
    for _ in range(n_samples):
        det.update(y_true=1, y_pred=1)

    drift_found = False
    for _ in range(n_samples):
        det.update(y_true=1, y_pred=0)
        drift_found = (det.drift_state == "drift") or drift_found
        if drift_found:
            break
    # confirm drift is found with very different stream
    assert drift_found is True
    # better refactored as a separate test with determined, desired window
    # but for now, just confirm that it's populated as part of the normal flow
    assert det.retraining_recs is not None

    # do we reset properly?
    det.update(y_true=1, y_pred=0)
    assert det.drift_state is None
    assert det.retraining_recs == [None, None]


def test_conservative_bound():
    """Confirm that, using conservative_bound param, drift is found in a
    very drifting stream.
    """
    det = ADWIN(new_sample_thresh=2, conservative_bound=True)
    n_samples = 50
    for _ in range(n_samples):
        det.update(y_true=1, y_pred=1)

    drift_found = False
    for _ in range(n_samples):
        det.update(y_true=1, y_pred=0)
        drift_found = (det.drift_state == "drift") or drift_found
        if drift_found:
            break
    # confirm drift is found with very different stream
    assert drift_found is True


def test_bucket_row_init():
    max_buckets = 10
    row3 = _BucketRow(max_buckets=max_buckets, prev_bucket=None, next_bucket=None)
    row1 = _BucketRow(max_buckets=max_buckets, prev_bucket=None, next_bucket=row3)
    row2 = _BucketRow(max_buckets=max_buckets, prev_bucket=row1, next_bucket=row3)

    assert row1.next_bucket is row2
    assert row2.next_bucket is row3
    assert row3.next_bucket is None

    assert row1.prev_bucket is None
    assert row2.prev_bucket is row1
    assert row3.prev_bucket is row2

    assert row1.bucket_count == 0
    assert len(row1.bucket_totals) == (max_buckets + 1)
    assert len(row1.bucket_totals) == (max_buckets + 1)


def test_bucket_row_list_empty():
    max_buckets = 10
    b_list = _BucketRowList(max_buckets=max_buckets)
    b_list.remove_tail()
    assert b_list.head is None

    b_list.append_tail()
    assert b_list.head is b_list.tail


def test_bucket_row_list_append_head():
    max_buckets = 10
    b_list = _BucketRowList(max_buckets=max_buckets)
    original_head = b_list.head
    b_list.append_head()
    assert original_head.prev_bucket is b_list.head
