"""All methods for testing correctness of CUSUM implementation."""
import numpy as np
import pytest
from menelaus.change_detection.cusum import CUSUM


def test_no_drift():
    """Test build / sliding with drift-less stream."""
    cusum = CUSUM(threshold=50)
    stream_size = 0
    for _ in range(40):  # test burn-in period (30) + post-burn-in (10)
        cusum.update(
            np.random.normal(size=1)
        )  # shouldn't detect drift, but cant inject same number
        stream_size += 1
        assert cusum.drift_state is None
        assert len(cusum._stream) == stream_size
        assert (
            cusum.samples_since_reset == stream_size
        )  # no reset til at least after burn in


def test_zero_sd():
    """Test build / feed stream with 1 unique value."""
    cusum = CUSUM(threshold=10)
    stream_size = 0
    try:
        for _ in range(31):  # test burn-in period (30) + post-burn-in (1)
            cusum.update(1)  # should trigger SD error on 31st observation
            stream_size += 1
    except ValueError:
        pass


def test_with_drift():
    """Test build / sliding with drift-y stream (bi-directional)."""
    cusum = CUSUM(threshold=10)
    stream_size = 0
    running_samples_since_reset = 0
    # wait for burn-in, induce drift, test, wait for burn-in again, induce drift, test
    for i in range(75):
        assert len(cusum._stream) == stream_size
        stream_size += 1
        # set drift at two locations (31st and 31st + 30 = 62nd indices), otherwise constants
        if i != 30 and i != 61:
            cusum.update(np.random.uniform())
            running_samples_since_reset += 1
            assert cusum.drift_state is None
        elif i == 30:
            cusum.update(100)
            running_samples_since_reset = 0
            assert cusum.drift_state == "drift"
        elif i == 61:
            cusum.threshold = 0  # arbitrarily lower parameter to induce drift
            cusum.update(100)
            running_samples_since_reset = 0
            assert cusum.drift_state == "drift"
        else:
            pass


def test_positive_drift():
    """Test build / sliding with drift-y stream in the positive direction."""
    cusum = CUSUM(direction="positive")
    stream_size = 0
    # wait for burn-in, induce drift, test
    for i in range(75):
        stream_size += 1
        # set positive drift at 31st location , negative drift at 62nd
        if i != 30:
            cusum.update(np.random.uniform())
        elif i == 30:
            cusum.update(100)
            assert cusum.drift_state == "drift"
        elif i == 61:
            cusum.update(
                -100
            )  # drift in the negative direction should not induce alarm
            assert cusum.drift_state is None
        else:
            pass


def test_negative_drift():
    """Test build / sliding with drift-y stream in the negative direction."""
    cusum = CUSUM(direction="negative")
    stream_size = 0
    # wait for burn-in, induce drift, test
    for i in range(75):
        stream_size += 1
        # set negative drift at 31st location , positive drift at 62nd
        if i != 30:
            cusum.update(np.random.uniform())
        elif i == 30:
            cusum.update(-100)
            assert cusum.drift_state == "drift"
        elif i == 61:
            cusum.update(100)  # drift in the positive direction should not induce alarm
            assert cusum.drift_state is None
        else:
            pass


def test_univariate_validation():
    det = CUSUM()
    input = np.array([[1, 2]])
    with pytest.raises(ValueError) as _:
        det.update(input)
