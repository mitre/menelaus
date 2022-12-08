"""All methods for testing HDDDM implementation."""
import numpy as np
import pandas as pd
import pytest
from menelaus.data_drift.hdddm import HDDDM


def test_validation():
    """Test ability to raise error if batches have different number of features"""
    data1 = np.repeat(1, 100)
    reference = pd.DataFrame(data=[data1, data1, data1]).T
    hdddm = HDDDM()
    hdddm.set_reference(reference)

    test = pd.DataFrame(data=[data1, data1, data1, data1]).T
    with pytest.raises(ValueError) as _:
        _ = hdddm.update(test)
