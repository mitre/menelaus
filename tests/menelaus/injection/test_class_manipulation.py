import pytest
import pandas as pd
import numpy as np

from menelaus.injection import class_join, class_swap


def test_join_1():
    ''' Check correct join in pd.DataFrame '''
    data = pd.DataFrame({'a': [0,0,1,1], 'b': [5,5,5,5]})
    expected_data = pd.DataFrame({'a': [0,2,2,1], 'b':[5,5,5,5]})
    data = class_join(data, 'a', 0, 1, 2, 1, 3)
    assert data.equals(expected_data)

def test_join_2():
    ''' Check correct join in numpy.ndarray '''
    data = np.array([[0,5], [0,5], [1,5], [1,5]])
    expected_data = np.array([[0,5], [2,5], [2,5], [1,5]])
    data = class_join(data, 0, 0, 1, 2, 1, 3)
    assert np.array_equal(data, expected_data)

def test_join_3():
    ''' Check ValueError if data neither pandas.DataFrame nor np.ndarray '''
    data = [[0,0], [1,1], [0,2], [1,3]]
    with pytest.raises(ValueError):
        class_join(
            data, target_col=0, class_1=0, class_2=1,
            new_class=2, from_index=1, to_index=3
        )

def test_swap_1():
    ''' Check correct swap in pd.DataFrame '''
    data = pd.DataFrame({'a': [0,0,1,1], 'b': [5,5,6,6]})
    expected_data = pd.DataFrame({'a': [1,1,0,0], 'b':[5,5,6,6]})
    data = class_swap(data, 'a', 0, 1, 0, 4)
    assert data.equals(expected_data)

def test_swap_2():
    ''' Check correct swap in numpy.ndarray'''
    data = np.array([[0,5], [0,5], [1,6], [1,6]])
    expected_data = np.array([[1,5], [1,5], [0,6], [0,6]])
    data = class_swap(data, 0, 0, 1, 0, 4)
    assert np.array_equal(data, expected_data)

def test_swap_3():
    ''' Check ValueError if data neither pandas.DataFrame nor np.ndarray '''
    data = [[0,0], [1,1], [0,2], [1,3]]
    with pytest.raises(ValueError):
        class_swap(
            data, target_col=0, class_1=0, class_2=1, 
            from_index=1, to_index=3
        )