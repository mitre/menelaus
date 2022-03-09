"""Methods for testing <logical flow> of PCA-CD."""
import pandas as pd
import numpy as np
from molten.data_drift.pca_cd import PCACD


def test_build_no_drift():
    window_size = 100
    det = PCACD(window_size=window_size)
    reference_dataframe = test_dataframe = np.ones((100, 3))
    # introduce some change so STD is not 0
    for i in range(window_size):
        arr = np.roll(np.array([1, 1, 0.9]), i)
        reference_dataframe[i] = test_dataframe[i] = arr
    reference_dataframe = pd.DataFrame(reference_dataframe)
    test_dataframe = pd.DataFrame(test_dataframe)
    # fill reference window
    reference_size = 0
    for i in range(window_size):
        # update by constant - no drift
        det.update(reference_dataframe.iloc[[i]])
        reference_size += 1
        assert det.drift_state is None
        assert len(det._reference_window) == reference_size  # e.g., compare to expected
        assert len(det._test_window) == 0
        assert det._build_reference_and_test is True
    # fill test window
    test_size = 0
    for i in range(window_size):
        # update by constant - no drift
        det.update(test_dataframe.iloc[[i]])
        test_size += 1
        assert det.drift_state is None
        assert (
            len(det._reference_window) == reference_size
        )  # alt: check actual data fixed
        assert len(det._test_window) == test_size
        if i < window_size - 1:
            assert det._build_reference_and_test is True
    assert det._build_reference_and_test is False
    # sliding
    for i in range(window_size):
        # update by constant - no drift
        det.update(test_dataframe.iloc[[i]])
        assert (
            len(det._test_window) == window_size
        )  # without drift, should be shifting by 1
        assert det.drift_state is None


# def test_build_with_drift():
#     window_size = 100
#     det = PCACD(window_size=window_size)
#     # fill reference window
#     for i in range(window_size):
#         det.update(pd.DataFrame([np.roll(np.array([1, 1, 0.9]), i)]))
#     # fill test window
#     test_size = 0
#     for i in range(window_size):
#         # notice the distribution settings here
#         det.update(pd.DataFrame([np.random.uniform(low=0, high=1, size=3)]))
#         test_size += 1
#         assert det.drift_state is None
#         assert len(det._test_window) == test_size
#         if i < window_size - 1:
#             assert det._build_reference_and_test is True
#     assert det._build_reference_and_test is False
#     # detection
#     drift_found = False
#     for i in range(window_size):
#         # notice the distribution settings here
#         det.update(pd.DataFrame([np.random.uniform(low=0, high=1e3, size=3)]))
#         drift_found = (det.drift_state == "drift") or drift_found
#         # trying to test that when drift found, windows updated correctly
#         # assert len(det._test_window) == window_size
#     assert drift_found
