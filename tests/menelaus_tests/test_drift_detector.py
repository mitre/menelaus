"""Module for checking behavior of DriftDetector."""
from menelaus.concept_drift import DDM
import pytest


def test_validation():
    """Confirm that bad drift_state values raise ValueErrors."""
    det = DDM()
    with pytest.raises(ValueError) as _:
        det.drift_state = "balloon"

    for state in ["drift", "warning", None]:
        det.drift_state = state
        assert det.drift_state == state
