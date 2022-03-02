"""Module to check correctness of kdqTree"""
import copy
import pytest
import numpy as np
import pandas as pd
from molten.distribution.kdq_tree import KdqTree

# TODO: visualization
# TODO: batch mode
NUM_FEATURES = 3


@pytest.fixture(name="kdq_det_stream")
def fixture_kdq_det_stream():
    """
    Returns:
        det: instance of KdqTree detector which has been updated with a stream
        of data that has just triggered drift.
    """
    # instance of KdqTree detector which has been updated with a
    # stream of data which triggers drift detection twice
    np.random.seed(123)
    det = KdqTree(window_size=50, count_ubound=3)

    # could be much faster if kdqTree accepted non-df args; numpy array, itertuples, ...
    input_df = pd.DataFrame(np.random.sample((60, NUM_FEATURES)))

    for i in range(len(input_df)):
        det.update(input_df.iloc[[i]].values)

    input_df = 50 * input_df

    for i in range(len(input_df)):
        det.update(input_df.iloc[[i]].values)
        if det.drift_state is not None:
            break

    return det


@pytest.fixture(name="kdq_det_batch")
def fixture_kdq_det_batch():
    """
    Returns:
        det: instance of KdqTree detector (batch) which has been updated with a
        batch of data that has just triggered drift.
    """
    np.random.seed(123)
    in_df = np.random.sample((10, NUM_FEATURES))
    det = KdqTree(stream=False, count_ubound=1, bootstrap_samples=10)
    det.update(in_df)
    det.update(in_df)
    det.update(50 * in_df)
    return det


def test_reset_stream(kdq_det_stream):
    """Check that drift_state resets one sample after taking the value "drift".

    Args:
        kdq_det_stream : kdqTree object from fixture_kdq_det_stream
    """
    det = copy.copy(kdq_det_stream)

    # given that the detector just drifted, next step ought not to regardless of input
    new_sample = pd.DataFrame(np.random.sample((1, NUM_FEATURES))).values
    det.update(new_sample)
    assert det.drift_state is None


def test_reset_batch(kdq_det_batch):
    det = copy.copy(kdq_det_batch)
    new_sample = pd.DataFrame(np.random.sample((1, NUM_FEATURES))).values
    det.update(new_sample)
    assert det.drift_state is None


def test_validation():
    with pytest.raises(ValueError) as _:
        det = KdqTree(window_size=None, stream=True)
    with pytest.raises(ValueError) as _:
        det = KdqTree(window_size=-5, stream=True)
    # TODO: validation on update()


# def test_tree_visualization(kdq_det_stream):
#     """Placeholder, only to execute the code without erroring out. Evaluating
#     correctness on this would be complicated, if only due to the way it's stored.

#     For now, just confirms that the tree visualization doesn't break midway, and
#     that it returns a single drift index.

#     Args:
#         kdq_det_stream : kdqTree object from fixture_kdq_det_stream
#     """
#     det = KdqTree(window_size=200)
#     assert det.drift_location_visualization() is None
#     out = kdq_det_stream.drift_location_visualization()
#     assert len(out) == 1
