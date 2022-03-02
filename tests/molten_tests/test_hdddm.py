"""All methods for testing HDDDM implementation."""
import numpy as np
import pandas as pd
import pytest
from molten.distribution.hdddm import HDDDM

def test_stdev():
    """ Test statistic = 'stdev' parameter vs 'tstat' method"""
    
    # setup data
    data2 = np.random.randint(2, 10, 150)
    reference = pd.DataFrame(data = [data2,data2,data2]).T
    data1 = np.repeat(1,100)
    test = pd.DataFrame(data = [data1,data1,data1]).T

    # test 'stdev'
    hdddm = HDDDM(reference_batch= reference, statistic = 'stdev', significance = 0.05)
    hdddm.update(test)
    beta_stdev = hdddm.beta

    # test 'tstat'
    hdddm = HDDDM(reference_batch= reference)
    hdddm.update(test)
    beta_tstat = hdddm.beta

    # check unequal
    assert beta_stdev != beta_tstat

def test_histograms():
    """ Test histograms are calculated by reference data characteristics"""
    data1 = np.repeat(1,100)
    reference = pd.DataFrame(data = [data1,data1,data1]).T
    hdddm = HDDDM(reference_batch= reference)

    # number of histograms matches number of variables 
    assert len(hdddm.reference_density) == reference.shape[1]

    # bins in histogram are calculated appropriately, using only first half of reference window 
    first_half = reference.iloc[0:int(len(reference)/2),]
    bin_count = int(np.floor(np.sqrt(len(first_half))))
    assert len(hdddm.reference_density[0]) == bin_count


def test_validation():
    """ Test ability to raise error if batches have different number of features """
    data1 = np.repeat(1,100)
    reference = pd.DataFrame(data = [data1,data1,data1]).T
    hdddm = HDDDM(reference_batch= reference)

    test = pd.DataFrame(data = [data1,data1,data1,data1]).T
    with pytest.raises(ValueError) as _:
        _ = hdddm.update(test)


def test_detect_batch_1_init():
    """ Tests functionality of init for detect batch = 1 """
    
    #init with first test batch 
    data1 = np.repeat(1,100)
    reference = pd.DataFrame(data = [data1,data1,data1]).T
    hdddm = HDDDM(reference_batch= reference)

    assert hdddm.total_samples == 1
    assert hdddm.samples_since_reset == 1
    assert hdddm.drift_state == None
    assert hdddm.prev_feature_distances != []
    assert hdddm.reference_n == len(reference)
    assert hdddm.prev_hellinger_distance != None

def test_detect_batch_1_no_drift():
    """ Tests functionality of update for detect batch = 1 with identical test batch, also tests drift is not detected """

    # init with first test batch 
    data1 = np.repeat(1,100)
    reference = pd.DataFrame(data = [data1,data1,data1]).T
    hdddm = HDDDM(reference_batch= reference)

    # update with identical test batch 
    hdddm.update(reference)

    assert hdddm.total_samples == 2
    assert hdddm.samples_since_reset == 2
    assert hdddm.drift_state == None
    assert hdddm.reference_n == len(reference)*2
    assert hdddm.feature_epsilons != []
    assert len(hdddm.epsilon) == 2
    assert hdddm.beta != None 

def test_detect_batch_1_epsilon():
    """ Tests initial epsilon estimate is removed from later calculations after 2 test batches """

    # init with first test batch 
    data1 = np.repeat(1,100)
    reference = pd.DataFrame(data = [data1,data1,data1]).T
    hdddm = HDDDM(reference_batch= reference)

    # update with identical test batch 
    hdddm.update(reference)

    # update with identical test batch 
    hdddm.update(reference)

    assert hdddm.drift_state == None
    assert len(hdddm.epsilon) == 2

def test_detect_batch_1_drift():
    """ Tests functionality of detecting drift for detect batch = 1 with very different batches """

    # init with first test batch 
    data1 = np.repeat(1,100)
    reference = pd.DataFrame(data = [data1,data1,data1]).T
    hdddm = HDDDM(reference_batch= reference)

    # update with different test batch 
    data2 = np.random.randint(2, 10, 150)
    test = pd.DataFrame(data = [data2,data2,data2]).T
    hdddm.update(test)

    assert hdddm.feature_info != None
    assert hdddm._drift_state == 'drift'
    assert hdddm._lambda == 2 
    assert len(hdddm.reference) == len(test) 

    

def test_detect_batch_1_reset():
    """ Tests functionality of reset after drift for detect batch = 1"""

    # init with first test batch 
    data1 = np.repeat(1,100)
    reference = pd.DataFrame(data = [data1,data1,data1]).T
    hdddm = HDDDM(reference_batch= reference)

    # update with different test batch 
    data2 = np.random.randint(2, 10, 150)
    test = pd.DataFrame(data = [data2,data2,data2]).T
    hdddm.update(test)

     # test reset updating with identical batch to test
    hdddm.update(test)

    assert hdddm.total_samples == 4 
    assert hdddm.samples_since_reset == 2
    assert len(hdddm.reference) == len(test)*2
    assert len(hdddm.epsilon) == 2 #epsilon is reset
    assert hdddm.beta != None
    assert hdddm._drift_state == None


def test_detect_batch_2_init():
    """ Tests functionality of init for detect batch = 2 """
    
    #init with first test batch 
    data1 = np.repeat(1,100)
    reference = pd.DataFrame(data = [data1,data1,data1]).T
    hdddm = HDDDM(reference_batch= reference, detect_batch=2)

    assert hdddm.total_samples == 0
    assert hdddm.samples_since_reset == 0
    assert hdddm.drift_state == None
    assert hdddm.reference_n == len(reference)


def test_detect_batch_2_no_drift():
    """ Tests functionality of not detecting drift for detect batch = 2 with identical batches, 
    tests that initial epsilon is removed from later calculations """

    # init with first test batch 
    data1 = np.repeat(1,100)
    reference = pd.DataFrame(data = [data1,data1,data1]).T
    hdddm = HDDDM(reference_batch= reference, detect_batch=2)

    # update with identical test batch
    hdddm.update(reference)

    assert hdddm.drift_state == None
    assert hdddm.reference_n == len(reference) + len(reference)

     # update with identical test batch
    hdddm.update(reference)

    assert hdddm.drift_state == None
    assert hdddm.total_samples == 2
    assert hdddm.feature_epsilons != []
    assert len(hdddm.epsilon) == 2 
    assert hdddm.beta != None

def test_detect_batch_2_drift():
    """ Tests functionality of detecting drift for detect batch = 2 with very different batches """

    # init with first test batch 
    data1 = np.repeat(1,100)
    reference = pd.DataFrame(data = [data1,data1,data1]).T
    hdddm = HDDDM(reference_batch= reference, detect_batch=2)

    # update with identical test batch 1
    hdddm.update(reference)

    # update with different test batch 2
    data2 = np.random.randint(2, 10, 150)
    test = pd.DataFrame(data = [data2,data2,data2]).T
    hdddm.update(test)

    assert hdddm.feature_info != None
    assert hdddm._drift_state == 'drift'
    assert hdddm._lambda == 2 
    assert len(hdddm.reference) == len(test) 

def test_detect_batch_2_reset():
    """ Tests functionality of reset after drift for detect batch = 2 """

    # init with first test batch 
    data1 = np.repeat(1,100)
    reference = pd.DataFrame(data = [data1,data1,data1]).T
    hdddm = HDDDM(reference_batch= reference, detect_batch = 2)
    
    # update with identical test batch 1
    hdddm.update(reference)

    # update with different test batch  2
    data2 = np.random.randint(2, 10, 150)
    test = pd.DataFrame(data = [data2,data2,data2]).T
    hdddm.update(test)
    
    assert hdddm._drift_state == 'drift'

    # test reset updating with identical batch to test batch 2
    hdddm.update(test)

    assert hdddm.total_samples == 3 
    assert hdddm.samples_since_reset == 1
    assert len(hdddm.epsilon) == 0 #epsilon is reset
    assert hdddm.beta != None
    assert hdddm._drift_state == None


def test_detect_batch_3_init():
    """ Tests functionality of init for detect batch = 3 """
    
    #init with first test batch 
    data1 = np.repeat(1,100)
    reference = pd.DataFrame(data = [data1,data1,data1]).T
    hdddm = HDDDM(reference_batch= reference, detect_batch=3)

    assert hdddm.total_samples == 0
    assert hdddm.samples_since_reset == 0
    assert hdddm.drift_state == None
    assert hdddm.reference_n == len(reference)
    
def test_detect_batch_3_no_drift():
    """ Tests functionality of not detecting drift for detect batch = 3 with identical batches, 
    tests functionality of update """

    # init with first test batch 
    data1 = np.repeat(1,100)
    reference = pd.DataFrame(data = [data1,data1,data1]).T
    hdddm = HDDDM(reference_batch= reference, detect_batch=3)

    # update with identical test batch
    hdddm.update(reference)

    assert hdddm.drift_state == None
    assert hdddm.reference_n == len(reference) + len(reference)

     # update with identical test batch
    hdddm.update(reference)
    
    # update with identical test batch
    hdddm.update(reference)

    assert hdddm.drift_state == None
    assert hdddm.total_samples == 3
    assert hdddm.feature_epsilons != []
    assert len(hdddm.epsilon) == 2 
    assert hdddm.beta != None

def test_detect_batch_3_drift():
    """ Tests functionality of detecting drift for detect batch = 3 with very different batches """

    # init with first test batch 
    data1 = np.repeat(1,100)
    reference = pd.DataFrame(data = [data1,data1,data1]).T
    hdddm = HDDDM(reference_batch= reference, detect_batch=3)

    # update with identical test batch 1
    hdddm.update(reference)
    
    # update with identical test batch 2
    hdddm.update(reference)

    # update with different test batch 3
    data2 = np.random.randint(2, 10, 150)
    test = pd.DataFrame(data = [data2,data2,data2]).T
    hdddm.update(test)

    assert hdddm.feature_info != None
    assert hdddm._drift_state == 'drift'
    assert hdddm._lambda == 3 
    assert len(hdddm.reference) == len(test) 

def test_detect_batch_3_reset():
    """ Tests functionality of reset after drift for detect batch = 3"""

    # init with first test batch 
    data1 = np.repeat(1,100)
    reference = pd.DataFrame(data = [data1,data1,data1]).T
    hdddm = HDDDM(reference_batch= reference, detect_batch = 3)
    
    # update with identical test batch 1
    hdddm.update(reference)
    
    # update with identical test batch 2
    hdddm.update(reference)

    # update with different test batch 3
    data2 = np.random.randint(2, 10, 150)
    test = pd.DataFrame(data = [data2,data2,data2]).T
    hdddm.update(test)
    
    assert hdddm._drift_state == 'drift'

    # test reset updating with identical batch to test batch 3
    hdddm.update(test)

    assert hdddm.total_samples == 4 
    assert hdddm.samples_since_reset == 1
    assert len(hdddm.epsilon) == 0 #epsilon is reset
    assert hdddm.beta != None
    assert hdddm._drift_state == None