import numpy as np

from menelaus.injection import FeatureSwapInjector, FeatureCoverInjector, FeatureShiftInjector


def test_feature_swap_1():
    ''' Check correct feature swap behavior '''
    data = np.array([[0,0], [1,8], [2,9], [3,3], [4,4]])
    expected_data = np.array([[0,0], [8,1], [9,2], [3,3], [4,4]])
    i = FeatureSwapInjector()
    copy = i(data, from_index=1, to_index=3, col_1=0, col_2=1)
    assert np.array_equal(copy, expected_data)

def test_feature_cover_1():
    ''' Check correct feature cover behavior '''
    data = np.array([[0,2], [0,2], [0,2], [1,3], [1,3], [1,3]])
    i = FeatureCoverInjector()
    copy = i(data, col=0, sample_size=2, random_state=0)
    assert copy.shape == (2,1)
    assert np.array_equal(copy, [[2], [3]])

def test_feature_shift_1():
    ''' Check correct feature cover behavior '''
    data = np.array([[0.0], [0.0], [3.0]])
    i = FeatureShiftInjector()
    copy = i(data=data, from_index=0, to_index=3, col=0, shift_factor=0.5, alpha=0.001)
    # delta = 0.5(0.001 + mean) = 0.5(1.001) = 0.5005
    exp = np.array([[0.5005], [0.5005], [3.5005]])
    assert np.allclose(copy, exp)