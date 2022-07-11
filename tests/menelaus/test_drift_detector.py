"""Module for checking behavior of DriftDetector."""
from menelaus.drift_detector import DriftDetector, StreamingDetector, BatchDetector
import pytest

# override this attribute so that it is instantiable
# hacky, but proper child classes doesn't cover the input_type setter.
DriftDetector.__abstractmethods__ = set()
StreamingDetector.__abstractmethods__ = set()
BatchDetector.__abstractmethods__ = set()


def test_validation_streaming():
    '''Ensure ValueError on invalid drift_state for StreamingDetector'''
    det = StreamingDetector()
    with pytest.raises(ValueError) as _:
        det.drift_state = "balloon"

    for state in ["drift", "warning", None]:
        det.drift_state = state
        assert det.drift_state == state

def test_validation_batch():
    '''Ensure ValueError on invalid drift_state for BatchDetector'''
    det = BatchDetector()
    with pytest.raises(ValueError) as _:
        det.drift_state = "balloon"

    for state in ["drift", "warning", None]:
        det.drift_state = state
        assert det.drift_state == state

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
