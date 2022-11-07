"""All methods for testing CDBD implementation."""
import numpy as np
import pandas as pd
import pytest
from menelaus.data_drift.cdbd import CDBD


def test_batch_validation():
    """Test ability to raise error if updating CDBD with a dataset containing more than 1 variable"""
    data1 = np.repeat(1, 100)
    reference = pd.DataFrame(data=[data1]).T
    test = pd.DataFrame(data=[data1, data1, data1]).T

    # initialize with reference batch
    cdbd = CDBD()
    cdbd.set_reference(reference)

    # update with test batch
    with pytest.raises(ValueError) as _:
        _ = cdbd.update(test)


def test_init_validation():
    """Test ability to raise error if initializing CDBD with a dataset containing more than 1 variable"""
    data1 = np.repeat(1, 100)
    reference = pd.DataFrame(data=[data1, data1, data1]).T
    with pytest.raises(ValueError) as _:
        _ = CDBD(detect_batch=3)
        _.set_reference(reference)
