"""All methods for testing HDDDM implementation."""
import numpy as np
import pandas as pd
import pytest
from molten.data_drift.hdddm import HDDDM

def test_validation():
    """Test ability to raise error if batches have different number of features"""
    data1 = np.repeat(1, 100)
    reference = pd.DataFrame(data=[data1, data1, data1]).T
    hdddm = HDDDM(reference_batch=reference)

    test = pd.DataFrame(data=[data1, data1, data1, data1]).T
    with pytest.raises(ValueError) as _:
        _ = hdddm.update(test)


def test_no_drift():
    """Tests functionality of update for detect batch = 1 with identical test batch, also tests drift is not detected"""

    # init with first test batch
    data1 = np.repeat(1, 100)
    reference = pd.DataFrame(data=[data1, data1, data1]).T
    hdddm = HDDDM(reference_batch=reference)

    # update with identical test batch
    hdddm.update(reference)

    assert hdddm.drift_state == None


def test_detect_drift():
    """Tests functionality of detecting drift for detect batch = 1 with very different batches"""

    # init with first test batch
    data1 = np.repeat(1, 100)
    reference = pd.DataFrame(data=[data1, data1, data1]).T
    hdddm = HDDDM(reference_batch=reference)

    # update with different test batch
    data2 = np.random.randint(2, 10, 150)
    test = pd.DataFrame(data=[data2, data2, data2]).T
    hdddm.update(test)

    assert hdddm.feature_info != None
    assert hdddm._drift_state == "drift"
    assert hdddm._lambda == 2
    assert len(hdddm.reference) == len(test)
