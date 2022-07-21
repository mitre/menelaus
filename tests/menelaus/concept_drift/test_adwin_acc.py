import pytest
from menelaus.concept_drift.adwin_acc import ADWINacc


def test_aliased_input():
    det = ADWINacc()

    det.update(y_true=1, y_pred=0)
    assert det.total_samples == 1

    with pytest.raises(TypeError) as _:
        det.update(X="hello!")
