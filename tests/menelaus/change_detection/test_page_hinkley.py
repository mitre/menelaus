"""All methods for testing correctness of Page Hinkley implementation."""
import pytest
import numpy as np
from menelaus.change_detection.page_hinkley import PageHinkley


def test_build_no_drift():
    """Function purpose."""
    monitor = PageHinkley()
    for i in range(monitor.burn_in * 2):
        # constant update
        monitor.update(1)
        assert monitor.drift_state is None
        assert monitor.samples_since_reset == i + 1


def test_build_pos_drift():
    """Function purpose"""
    monitor = PageHinkley(burn_in=30)  # positive direction
    stream_size = 0
    for i in range(monitor.burn_in * 2):
        stream_size += 1
        # forced update
        if i < monitor.burn_in:
            monitor.update(1)
            assert monitor.drift_state is None
        elif i == monitor.burn_in:
            monitor.update(100)
            stream_size = 0
            assert monitor.drift_state == "drift"
        elif i == monitor.burn_in + 1:
            monitor.update(1)
            assert monitor.drift_state is None
            assert (
                len(monitor.to_dataframe()) == stream_size
            )  # hack to test all reset() behavior


def test_build_neg_drift():
    """Function purpose"""
    monitor = PageHinkley(burn_in=30, direction="negative")
    stream_size = 0
    for i in range(monitor.burn_in * 2):
        stream_size += 1
        # forced update
        if i < monitor.burn_in:
            monitor.update(1)
            assert monitor.drift_state is None
        elif i == monitor.burn_in:
            monitor.update(-100)
            stream_size = 0
            assert monitor.drift_state == "drift"
        elif i == monitor.burn_in + 1:
            monitor.update(1)
            assert monitor.drift_state is None
            assert (
                len(monitor.to_dataframe()) == stream_size
            )  # hack to test all reset() behavior


def test_univariate_validation():
    det = PageHinkley()
    input = np.array([[1, 2]])
    with pytest.raises(ValueError) as _:
        det.update(input)
