import pytest
from menelaus.concept_drift.adwin_acc import ADWINacc

# All but the first test here are completely redundant with change_detection.test_adwin.
# Because we want to get the docstrings to generate properly, and also be search-able,
# they're aliased, which means these need to run to get coverage.


def test_aliased_input():
    det = ADWINacc()

    det.update(y_true=1, y_pred=0)
    assert det.total_samples == 1

    # the alias shouldn't take any "feature" input
    with pytest.raises(TypeError) as _:
        det.update(X="hello!")


def test_mean():
    """Check that ADWIN.mean returns sane values"""
    det = ADWINacc(new_sample_thresh=2)
    assert det.mean() == 0
    n_samples = 5
    for _ in range(n_samples):
        det.update(1, 1)
    assert det.mean() == 1


def test_variance():
    """Check that ADWIN.variance returns sane values"""
    det = ADWINacc(new_sample_thresh=2)
    assert det.variance() == 0
    n_samples = 10
    for _ in range(n_samples):
        det.update(1, 1)
        det.update(0, 1)
    assert det.variance() == 0.25


def test_drift():
    """Check that, for a very drifting data stream:
    - drift is identified
    - ADWIN.retraining_recs takes on the proper values before and after drift
    """
    det = ADWINacc(new_sample_thresh=2)
    n_samples = 60
    for _ in range(n_samples):
        det.update(1, 1)

    drift_found = False
    for _ in range(n_samples):
        det.update(0, 1)
        drift_found = (det.drift_state == "drift") or drift_found
        if drift_found:
            break
    # confirm drift is found with very different stream
    assert drift_found is True
    # better refactored as a separate test with determined, desired window
    # but for now, just confirm that it's populated as part of the normal flow
    assert det.retraining_recs is not None

    # do we reset properly?
    det.update(1, 1)
    assert det.drift_state is None
    assert det.retraining_recs == [None, None]
