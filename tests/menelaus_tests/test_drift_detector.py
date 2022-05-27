"""Module for checking behavior of DriftDetector."""
from menelaus.drift_detector import DriftDetector
import pytest

# override this attribute so that it is instantiable
# hacky, but proper child classes doesn't cover the input_type setter.
DriftDetector.__abstractmethods__ = set()


def test_validation():
    """Confirm that bad drift_state values raise ValueErrors."""
    det = DriftDetector()
    with pytest.raises(ValueError) as _:
        det.drift_state = "balloon"

    for state in ["drift", "warning", None]:
        det.drift_state = state
        assert det.drift_state == state


def test_input_type():
    det = DriftDetector()
    det._input_type = "stream"
    assert det.input_type == "stream"
