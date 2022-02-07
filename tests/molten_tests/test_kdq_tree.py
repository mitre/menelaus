"""Module to check correctness of kdqTree"""
import copy
import pytest
import numpy as np
import pandas as pd
from molten.distribution.kdq_tree import KdqTree


@pytest.fixture(name="kdq_det")
def fixture_kdq_det():
    """
    Returns:
        det: instance of KdqTree detector which has been updated with a stream
        of data that has just triggered drift.
    """
    # instance of KdqTree detector which has been updated with a
    # stream of data which triggers drift detection twice
    np.random.seed(123)
    det = KdqTree(window_size=50, min_points_in_bin=3)

    # could be much faster if kdqTree accepted non-df args; numpy array, itertuples, ...
    input_df = pd.DataFrame(np.random.sample((60, 3)))

    for i in range(len(input_df)):
        det.update(input_df.iloc[[i]])

    input_df = 50 * input_df

    for i in range(len(input_df)):
        det.update(input_df.iloc[[i]])
        if det.drift_state is not None:
            break

    return det


def test_reset(kdq_det):
    """Check that drift_state resets one sample after taking the value "drift".

    Args:
        kdq_det : kdqTree object from fixture_kdq_det
    """
    det = copy.copy(kdq_det)
    # infer the shape of the passed data
    n_features = det._window_data["reference"].shape[1]

    # given that the detector just drifted, next step ought not to regardless of input
    new_sample = pd.DataFrame(np.random.sample((1, n_features)))
    det.update(new_sample)
    assert det.drift_state is None


def test_tree_visualization(kdq_det):
    """Placeholder, only to execute the code without erroring out. Evaluating
    correctness on this would be complicated, if only due to the way it's stored.

    For now, just confirms that the tree visualization doesn't break midway, and
    that it returns a single drift index.

    Args:
        kdq_det : kdqTree object from fixture_kdq_det
    """
    det = KdqTree(window_size=200)
    assert det.drift_location_visualization() is None
    out = kdq_det.drift_location_visualization()
    assert len(out) == 1
