import pytest
from menelaus.concept_drift.adwin_accuracy import ADWINAccuracy


def test_aliased_input():
    det = ADWINAccuracy()

    det.update(y_true=1, y_pred=0)
    assert det.total_samples == 1

    # the alias shouldn't take any "feature" input
    with pytest.raises(TypeError) as _:
        det.update(X="hello!")
