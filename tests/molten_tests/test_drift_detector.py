"""Module for checking behavior of DriftDetector."""
from molten.drift_detector import DriftDetector
import pytest


def test_validation():
    """Confirm that bad drift_state values raise ValueErrors."""
    det = DriftDetector()
    with pytest.raises(ValueError) as _:
        det.drift_state = "balloon"

    for state in ["drift", "warning", None]:
        det.drift_state = state
        assert det.drift_state == state
