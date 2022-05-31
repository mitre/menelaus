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


def test_drift():
    """Test of modified CDBD test using KL divergence"""

    # np.random.seed(1)
    reference = pd.DataFrame(np.random.uniform(0, 1, 100))
    test = pd.DataFrame(np.random.uniform(3, 4, 100))

    # initialize with reference batch
    cdbd = CDBD()
    cdbd.set_reference(reference)

    # update with different test batch
    cdbd.update(test)

    assert cdbd._drift_state == "drift"
    assert len(cdbd.reference) == len(test)

def test_custom_divmetric():
    """ Tests functionality of user defining custom divergence metric """
    
    # Define divergence metric
    def distance_metric1(reference_histogram, test_histogram):

        # Convert inputs to appropriate datatype 
        ref = np.array(reference_histogram[0])
        test = np.array(test_histogram[0])

        dist = np.sqrt(np.sum(np.square(ref-test)))
        
        return dist

    # Setup data
    reference = pd.DataFrame(data=[np.repeat(1, 100)]).T
    test = pd.DataFrame(data=[np.repeat(4, 100)]).T

    # initialize with reference batch
    cdbd = CDBD(divergence=distance_metric1)
    cdbd.set_reference(reference)

    cdbd.update(test)
    assert cdbd.current_distance != 0
