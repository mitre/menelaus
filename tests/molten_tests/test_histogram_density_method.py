"""All methods for testing HDM implementation."""
import numpy as np
import pandas as pd
from molten.data_drift.histogram_density_method import HistogramDensityMethod as HDM


def test_stdev():
    """Test statistic = 'stdev' parameter vs 'tstat' method"""

    # setup data
    data2 = np.random.randint(2, 10, 150)
    reference = pd.DataFrame(data=[data2, data2, data2]).T
    data1 = np.repeat(1, 100)
    test = pd.DataFrame(data=[data1, data1, data1]).T

    # test 'stdev'
    hdm = HDM(
        reference_batch=reference,
        divergence="H",
        detect_batch=1,
        statistic="stdev",
        significance=0.05,
        subsets=5,
    )
    hdm.update(test)
    beta_stdev = hdm.beta

    # test 'tstat'
    hdm = HDM(
        reference_batch=reference,
        divergence="H",
        detect_batch=1,
        statistic="tstat",
        significance=0.05,
        subsets=5,
    )
    hdm.update(test)
    beta_tstat = hdm.beta

    # check unequal
    assert beta_stdev != beta_tstat


def test_histograms():
    """Test histograms are calculated by reference data characteristics"""
    data1 = np.repeat(1, 100)
    reference = pd.DataFrame(data=[data1, data1, data1]).T
    hdm = HDM(
        reference_batch=reference,
        divergence="H",
        detect_batch=1,
        statistic="tstat",
        significance=0.05,
        subsets=5,
    )

    # number of histograms matches number of variables
    assert len(hdm.reference_density) == reference.shape[1]

    # bins in histogram are calculated appropriately, using only first half of reference window
    first_half = reference.iloc[
        0 : int(len(reference) / 2),
    ]
    bin_count = int(np.floor(np.sqrt(len(first_half))))
    assert len(hdm.reference_density[0]) == bin_count


def test_detect_batch_1_init():
    """Tests functionality of init for detect batch = 1"""

    # init with first test batch
    data1 = np.repeat(1, 100)
    reference = pd.DataFrame(data=[data1, data1, data1]).T
    hdm = HDM(
        reference_batch=reference,
        divergence="H",
        detect_batch=1,
        statistic="tstat",
        significance=0.05,
        subsets=5,
    )

    assert hdm.total_samples == 1
    assert hdm.samples_since_reset == 1
    assert hdm.drift_state == None
    assert hdm.prev_feature_distances != []
    assert hdm.reference_n == len(reference)
    assert hdm.prev_distance != None


def test_detect_batch_1_no_drift():
    """Tests functionality of update for detect batch = 1 with identical test batch, also tests drift is not detected"""

    # init with first test batch
    data1 = np.repeat(1, 100)
    reference = pd.DataFrame(data=[data1, data1, data1]).T
    hdm = HDM(
        reference_batch=reference,
        divergence="H",
        detect_batch=1,
        statistic="tstat",
        significance=0.05,
        subsets=5,
    )

    # update with identical test batch
    hdm.update(reference)

    assert hdm.total_samples == 2
    assert hdm.samples_since_reset == 2
    assert hdm.drift_state == None
    assert hdm.reference_n == len(reference) * 2
    assert hdm.feature_epsilons != []
    assert len(hdm.epsilon) == 2
    assert hdm.beta != None


def test_detect_batch_1_epsilon():
    """Tests initial epsilon estimate is removed from later calculations after 2 test batches"""

    # init with first test batch
    data1 = np.repeat(1, 100)
    reference = pd.DataFrame(data=[data1, data1, data1]).T
    hdm = HDM(
        reference_batch=reference,
        divergence="H",
        detect_batch=1,
        statistic="tstat",
        significance=0.05,
        subsets=5,
    )

    # update with identical test batch
    hdm.update(reference)

    # update with identical test batch
    hdm.update(reference)

    assert hdm.drift_state == None
    assert len(hdm.epsilon) == 2


def test_detect_batch_1_drift():
    """Tests functionality of detecting drift for detect batch = 1 with very different batches"""

    # init with first test batch
    data1 = np.repeat(1, 100)
    reference = pd.DataFrame(data=[data1, data1, data1]).T
    hdm = HDM(
        reference_batch=reference,
        divergence="H",
        detect_batch=1,
        statistic="tstat",
        significance=0.05,
        subsets=5,
    )

    # update with different test batch
    data2 = np.random.randint(2, 10, 150)
    test = pd.DataFrame(data=[data2, data2, data2]).T
    hdm.update(test)

    assert hdm.feature_info != None
    assert hdm._drift_state == "drift"
    assert hdm._lambda == 2
    assert len(hdm.reference) == len(test)


def test_detect_batch_1_reset():
    """Tests functionality of reset after drift for detect batch = 1"""

    # init with first test batch
    data1 = np.repeat(1, 100)
    reference = pd.DataFrame(data=[data1, data1, data1]).T
    hdm = HDM(
        reference_batch=reference,
        divergence="H",
        detect_batch=1,
        statistic="tstat",
        significance=0.05,
        subsets=5,
    )

    # update with different test batch
    data2 = np.random.randint(2, 10, 150)
    test = pd.DataFrame(data=[data2, data2, data2]).T
    hdm.update(test)

    # test reset updating with identical batch to test
    hdm.update(test)

    assert hdm.total_samples == 4
    assert hdm.samples_since_reset == 2
    assert len(hdm.epsilon) == 2  # epsilon is reset
    assert hdm.beta != None
    assert hdm.drift_state is None


def test_detect_batch_2_init():
    """Tests functionality of init for detect batch = 2"""

    # init with first test batch
    data1 = np.repeat(1, 100)
    reference = pd.DataFrame(data=[data1, data1, data1]).T
    hdm = HDM(
        reference_batch=reference,
        divergence="H",
        detect_batch=2,
        statistic="tstat",
        significance=0.05,
        subsets=5,
    )

    assert hdm.total_samples == 0
    assert hdm.samples_since_reset == 0
    assert hdm.drift_state == None
    assert hdm.reference_n == len(reference)


def test_detect_batch_2_no_drift():
    """Tests functionality of not detecting drift for detect batch = 2 with identical batches,
    tests that initial epsilon is removed from later calculations"""

    # init with first test batch
    data1 = np.repeat(1, 100)
    reference = pd.DataFrame(data=[data1, data1, data1]).T
    hdm = HDM(
        reference_batch=reference,
        divergence="H",
        detect_batch=2,
        statistic="tstat",
        significance=0.05,
        subsets=5,
    )

    # update with identical test batch
    hdm.update(reference)

    assert hdm.drift_state == None
    assert hdm.reference_n == len(reference) + len(reference)

    # update with identical test batch
    hdm.update(reference)

    assert hdm.drift_state == None
    assert hdm.total_samples == 2
    assert hdm.feature_epsilons != []
    assert len(hdm.epsilon) == 2
    assert hdm.beta != None


def test_detect_batch_2_drift():
    """Tests functionality of detecting drift for detect batch = 2 with very different batches"""

    # init with first test batch
    data1 = np.repeat(1, 100)
    reference = pd.DataFrame(data=[data1, data1, data1]).T
    hdm = HDM(
        reference_batch=reference,
        divergence="H",
        detect_batch=2,
        statistic="tstat",
        significance=0.05,
        subsets=5,
    )

    # update with identical test batch 1
    hdm.update(reference)

    # update with different test batch 2
    data2 = np.random.randint(2, 10, 150)
    test = pd.DataFrame(data=[data2, data2, data2]).T
    hdm.update(test)

    assert hdm.feature_info != None
    assert hdm._drift_state == "drift"
    assert hdm._lambda == 2
    assert len(hdm.reference) == len(test)


def test_detect_batch_2_reset():
    """Tests functionality of reset after drift for detect batch = 2"""

    # init with first test batch
    data1 = np.repeat(1, 100)
    reference = pd.DataFrame(data=[data1, data1, data1]).T
    hdm = HDM(
        reference_batch=reference,
        divergence="H",
        detect_batch=2,
        statistic="tstat",
        significance=0.05,
        subsets=5,
    )

    # update with identical test batch 1
    hdm.update(reference)

    # update with different test batch  2
    data2 = np.random.randint(2, 10, 150)
    test = pd.DataFrame(data=[data2, data2, data2]).T
    hdm.update(test)

    assert hdm._drift_state == "drift"

    # test reset updating with identical batch to test batch 2
    hdm.update(test)

    assert hdm.total_samples == 3
    assert hdm.samples_since_reset == 1
    assert len(hdm.epsilon) == 0  # epsilon is reset
    assert hdm.beta != None
    assert hdm._drift_state == None


def test_detect_batch_3_init():
    """Tests functionality of init for detect batch = 3"""

    # init with first test batch
    data1 = np.repeat(1, 100)
    reference = pd.DataFrame(data=[data1, data1, data1]).T
    hdm = HDM(
        reference_batch=reference,
        divergence="H",
        detect_batch=3,
        statistic="tstat",
        significance=0.05,
        subsets=5,
    )

    assert hdm.total_samples == 0
    assert hdm.samples_since_reset == 0
    assert hdm.drift_state == None
    assert hdm.reference_n == len(reference)


def test_detect_batch_3_no_drift():
    """Tests functionality of not detecting drift for detect batch = 3 with identical batches,
    tests functionality of update"""

    # init with first test batch
    data1 = np.repeat(1, 100)
    reference = pd.DataFrame(data=[data1, data1, data1]).T
    hdm = HDM(
        reference_batch=reference,
        divergence="H",
        detect_batch=3,
        statistic="tstat",
        significance=0.05,
        subsets=5,
    )

    # update with identical test batch
    hdm.update(reference)

    assert hdm.drift_state == None
    assert hdm.reference_n == len(reference) + len(reference)

    # update with identical test batch
    hdm.update(reference)

    # update with identical test batch
    hdm.update(reference)

    assert hdm.drift_state == None
    assert hdm.total_samples == 3
    assert hdm.feature_epsilons != []
    assert len(hdm.epsilon) == 2
    assert hdm.beta != None


def test_detect_batch_3_drift():
    """Tests functionality of detecting drift for detect batch = 3 with very different batches"""

    # init with first test batch
    data1 = np.repeat(1, 100)
    reference = pd.DataFrame(data=[data1, data1, data1]).T
    hdm = HDM(
        reference_batch=reference,
        divergence="H",
        detect_batch=3,
        statistic="tstat",
        significance=0.05,
        subsets=5,
    )

    # update with identical test batch 1
    hdm.update(reference)

    # update with identical test batch 2
    hdm.update(reference)

    # update with different test batch 3
    data2 = np.random.randint(2, 10, 150)
    test = pd.DataFrame(data=[data2, data2, data2]).T
    hdm.update(test)

    assert hdm.feature_info != None
    assert hdm._drift_state == "drift"
    assert hdm._lambda == 3
    assert len(hdm.reference) == len(test)


def test_detect_batch_3_reset():
    """Tests functionality of reset after drift for detect batch = 3"""

    # init with first test batch
    data1 = np.repeat(1, 100)
    reference = pd.DataFrame(data=[data1, data1, data1]).T
    hdm = HDM(
        reference_batch=reference,
        divergence="H",
        detect_batch=3,
        statistic="tstat",
        significance=0.05,
        subsets=5,
    )

    # update with identical test batch 1
    hdm.update(reference)

    # update with identical test batch 2
    hdm.update(reference)

    # update with different test batch 3
    data2 = np.random.randint(2, 10, 150)
    test = pd.DataFrame(data=[data2, data2, data2]).T
    hdm.update(test)

    assert hdm._drift_state == "drift"

    # test reset updating with identical batch to test batch 3
    hdm.update(test)

    assert hdm.total_samples == 4
    assert hdm.samples_since_reset == 1
    assert len(hdm.epsilon) == 0  # epsilon is reset
    assert hdm.beta != None
    assert hdm._drift_state == None


# TODO add tests for testing KL divergence


def test_KL():
    """Test of modified CDBD test using KL divergence"""

    # np.random.seed(1)
    reference = pd.DataFrame(np.random.uniform(0, 1, 100))
    test = pd.DataFrame(np.random.uniform(3, 4, 100))

    # initialize with reference batch
    hdm = HDM(
        reference_batch=reference,
        divergence="KL",
        detect_batch=1,
        statistic="tstat",
        significance=0.05,
        subsets=5,
    )

    # update with different test batch
    hdm.update(test)

    assert hdm._drift_state == "drift"
    assert len(hdm.reference) == len(test)
