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


def test_no_drift():
    """Tests functionality of update for detect batch = 1 with identical test batch, also tests drift is not detected"""

    # init with first test batch
    data1 = np.repeat(1, 100)
    reference = pd.DataFrame(data=[data1, data1, data1]).T
    hdddm = HDDDM()
    hdddm.set_reference(reference)

    # update with identical test batch
    hdddm.update(reference)

    assert hdddm.drift_state == None


def test_detect_drift():
    """Tests functionality of detecting drift for detect batch = 1 with very different batches"""

    # init with first test batch
    data1 = np.repeat(1, 100)
    reference = pd.DataFrame(data=[data1, data1, data1]).T
    hdddm = HDDDM()
    hdddm.set_reference(reference)

    # update with different test batch
    data2 = np.random.randint(2, 10, 150)
    test = pd.DataFrame(data=[data2, data2, data2]).T
    hdddm.update(test)

    assert hdddm.feature_epsilons != None
    assert hdddm._drift_state == "drift"
    assert hdddm._lambda == 2
    assert len(hdddm.reference) == len(test)



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
    data1 = np.repeat(1, 100)
    reference = pd.DataFrame(data=[data1, data1, data1]).T
    data2 = np.random.randint(2, 10, 150)
    test = pd.DataFrame(data=[data2, data2, data2]).T

    # Run HDDDM
    hdddm = HDDDM(divergence=distance_metric1)
    hdddm.set_reference(reference)
    hdddm.update(test)

    assert hdddm.current_distance != 0
