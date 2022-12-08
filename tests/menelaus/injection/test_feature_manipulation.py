import pytest
import pandas as pd
import numpy as np

from menelaus.injection import feature_swap, feature_hide_and_sample


def test_feature_swap_1():
    ''' Check correct swap in pandas.DataFrame '''
    data = pd.DataFrame({'a':[0,1,2,3,4], 'b':[0,8,9,3,4]})
    expected_data = pd.DataFrame({'a':[0,8,9,3,4], 'b':[0,1,2,3,4]})
    data = feature_swap(data, 'a', 'b', 1, 3)
    assert data.equals(expected_data)

def test_feature_swap_2():
    ''' Check correct swap in numpy.ndarray '''
    data = np.array([[0,0], [1,8], [2,9], [3,3], [4,4]])
    expected_data = np.array([[0,0], [8,1], [9,2], [3,3], [4,4]])
    data = feature_swap(data, 0, 1, 1, 3)
    assert np.array_equal(data, expected_data)

def test_feature_swap_3():
    ''' Check function fails when data is neither numpy nor pandas '''
    data = [1,2,3,4]
    with pytest.raises(ValueError):
        feature_swap(data, 'col1', 'col2', 0, 2)

def test_feature_hide_1():
    ''' Check correct operation in pandas.DataFrame '''
    data = pd.DataFrame({'a': [0, 0, 0, 1, 1, 1], 'b': [0, 1, 2, 3, 4, 5]})
    expected_data = pd.DataFrame({'b': [2,1,5,3]}) 
    # expect result, since random state is default 0 in function call
    data = feature_hide_and_sample(data, 'a', 2)
    assert data.equals(expected_data)

def test_feature_hide_2():
    ''' Check correct operation in numpy.ndarray '''
    data = np.array([[0,0], [0,1], [0,2], [1,3], [1,4], [1,5]])
    expected_data = np.array([2,1,5,3]) 
    # expect result, since random state is default 0 in function call
    data = feature_hide_and_sample(data, 0, 2)
    data = data.reshape(-1)
    assert np.array_equal(data, expected_data)

def test_feature_hide_3():
    ''' Check function fails when data is neither numpy nor pandas '''
    data = [1,2,3,4]
    with pytest.raises(ValueError):
        feature_hide_and_sample(data, 'col', 2)