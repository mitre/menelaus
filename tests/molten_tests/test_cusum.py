"""All methods for testing correctness of CUSUM implementation."""
import numpy as np
from molten.other.cusum import CUSUM


def test_no_drift():
    """Test build / sliding with drift-less stream."""
    cusum = CUSUM()
    stream_size = 0
    for _ in range(40):  # test burn-in period (30) + post-burn-in (10)
        cusum.update(1)  # shouldn't detect drift
        stream_size += 1
        assert cusum.drift_state is None
        assert len(cusum._stream) == stream_size
        assert (cusum.samples_since_reset == stream_size)  # no reset til at least after burn in


def test_with_drift():
    """Test build / sliding with drift-y stream."""
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
