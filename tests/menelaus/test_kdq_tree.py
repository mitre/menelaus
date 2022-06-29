"""Module to check correctness of kdqTree"""
import copy
import pytest
import numpy as np
import pandas as pd
from menelaus.data_drift.kdq_tree import KdqTree


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
    det = KdqTree(input_type="batch", count_ubound=1, bootstrap_samples=10)
    det.set_reference(in_df)
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
    new_input_col_names = list(range(5, 5 + NUM_FEATURES))
    new_sample = pd.DataFrame(
        np.random.sample((1, NUM_FEATURES)), columns=new_input_col_names
    )
    det.update(new_sample)
    assert det.drift_state is None
    det.reset()
    det.update(new_sample)
    assert det.input_cols.equals(new_sample.columns)


def test_reset_batch(kdq_det_batch):
    det = copy.copy(kdq_det_batch)
    new_sample = pd.DataFrame(np.random.sample((1, NUM_FEATURES))).values
    det.update(new_sample)
    assert det.drift_state is None


def test_set_reference(kdq_det_batch):
    det = copy.copy(kdq_det_batch)
    new_sample = pd.DataFrame(np.random.sample((1, NUM_FEATURES))).values
    det.set_reference(new_sample)
    det.update(new_sample)
    assert det.drift_state is None

    # make sure that input_cols is set appropriately when initialized
    det = det = KdqTree(input_type="batch", count_ubound=1, bootstrap_samples=10)
    new_sample = pd.DataFrame(np.random.sample((1, NUM_FEATURES)))
    det.set_reference(new_sample)
    assert det.input_cols.equals(new_sample.columns)


def test_init_validation_stream(kdq_det_stream):
    with pytest.raises(ValueError) as _:
        det = KdqTree(window_size=None, input_type="stream")
    with pytest.raises(ValueError) as _:
        det = KdqTree(window_size=-5, input_type="stream")


def test_set_reference_validation_stream(kdq_det_stream):
    with pytest.raises(ValueError) as _:
        det = copy.copy(kdq_det_stream)
        det.set_reference("garbage input")


def test_set_reference_validation_batch(kdq_det_batch):
    det = copy.copy(kdq_det_batch)
    with pytest.raises(ValueError) as _:
        det.set_reference("garbage input")


def test_update_validation_batch(kdq_det_batch):
    det = copy.copy(kdq_det_batch)
    with pytest.raises(ValueError) as _:
        det.update("garbage input")

    # this should not raise a ValueError in the current implementation
    # since the kdq_det_batch fixture is initialized with a numpy array
    new_name_df = pd.DataFrame(
        np.random.sample((10, NUM_FEATURES)), columns=list(range(5, 5 + NUM_FEATURES))
    )
    det.update(new_name_df)

    # this *should* raise a ValueError in the current implementation
    with pytest.raises(ValueError) as _:
        det.set_reference(new_name_df)
        bad_name_df = pd.DataFrame(
            np.random.sample((10, NUM_FEATURES)),
            columns=list(range(10, 10 + NUM_FEATURES)),
        )
        det.update(bad_name_df)


def test_viz_dataframe(kdq_det_batch):
    plot_df = kdq_det_batch.to_plotly_dataframe()
    assert plot_df.shape[0] > 0
    # another one where checking the contents would be quite elaborate, but we can at least
    # ensure the structure is correct
    assert set(plot_df.columns) == set(
        ["name", "idx", "parent_idx", "cell_count", "depth", "count_diff", "kss"]
    )
    new_input_col_names = list(range(5, 5 + NUM_FEATURES))
    plot_df = kdq_det_batch.to_plotly_dataframe(input_cols=new_input_col_names)
    assert set(new_input_col_names) == set(
        set(plot_df["name"].str.split(" ").str[0][1:].astype("int").unique())
    )
