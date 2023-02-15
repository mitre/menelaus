import numpy as np

from menelaus.injection import FeatureSwapInjector, FeatureCoverInjector


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
    assert copy.shape == (4,1)
    assert np.array_equal(np.where(copy[:,0]==2)[0], [0,1])
    assert np.array_equal(np.where(copy[:,0]==3)[0], [2,3])
