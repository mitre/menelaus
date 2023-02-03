import pytest
import pandas as pd
import numpy as np


from menelaus.injection import class_join, class_swap, class_probability_shift, class_dirichlet_shift


def test_join_1():
    ''' Check correct join in pd.DataFrame '''
    data = pd.DataFrame({'a': [0,0,1,1], 'b': [5,5,5,5]})
    expected_data = pd.DataFrame({'a': [0,2,2,1], 'b':[5,5,5,5]})
    data = class_join(data, 'a', 1, 3, 0, 1, 2)
    assert data.equals(expected_data)

def test_join_2():
    ''' Check correct join in numpy.ndarray '''
    data = np.array([[0,5], [0,5], [1,5], [1,5]])
    expected_data = np.array([[0,5], [2,5], [2,5], [1,5]])
    data = class_join(data, 0, 1, 3, 0, 1, 2)
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
    data = class_swap(data, 'a', 0, 4, 0, 1)
    assert data.equals(expected_data)

def test_swap_2():
    ''' Check correct swap in numpy.ndarray'''
    data = np.array([[0,5], [0,5], [1,6], [1,6]])
    expected_data = np.array([[1,5], [1,5], [0,6], [0,6]])
    data = class_swap(data, 0, 0, 4, 0, 1)
    assert np.array_equal(data, expected_data)

def test_swap_3():
    ''' Check ValueError if data neither pandas.DataFrame nor np.ndarray '''
    data = [[0,0], [1,1], [0,2], [1,3]]
    with pytest.raises(ValueError):
        class_swap(
            data, target_col=0, class_1=0, class_2=1, 
            from_index=1, to_index=3
        )

def test_probability_shift_1():
    ''' Ensure probability shift causes some drift in numpy.ndarray '''
    np.random.seed(0)
    data = np.array([
        [0,0,0], [1,0,0], [2,0,0], [3,0,0], [4,0,0], [5,0,0],
        [6,0,0], [7,0,0], [8,0,0], [9,0,0]
    ])
    probs = {k:0.1 for k in range(0,10)}
    new_data = class_probability_shift(data, 0, 1, 9, probs)
    assert np.array_equal(data[0], new_data[0])
    assert np.array_equal(data[-1], new_data[-1])
    assert not np.array_equal(data[1:9], new_data[1:9])

def test_probability_shift_2():
    ''' Ensure probability shift causes some drift in pandas.DataFrame '''
    np.random.seed(0)
    data = pd.DataFrame({
        'a': [0,1,2,3,4,5,6,7,8,9],
        'b': [0,0,0,0,0,0,0,0,0,0]
    })
    probs = {k:0.1 for k in range(0,10)}
    new_data = class_probability_shift(data, 0, 1, 9, probs)
    assert data.iloc[0].equals(new_data.iloc[0])
    assert data.iloc[-1].equals(new_data.iloc[-1])
    assert not np.array_equal(data[1:9], new_data[1:9])
    assert list(new_data.columns) == ['a', 'b']

def test_probability_shift_3():
    ''' Check ValueError if data neither pandas.DataFrame nor numpy.ndarray '''
    data = [[0,0], [1,1], [0,2], [1,3]]
    with pytest.raises(ValueError):
        class_probability_shift(
            data, target_col=0, from_index=0, to_index=2, 
            class_probabilities={0: 0.5}
        )

def test_probability_shift_4():
    ''' Check ValueError when probabilities add to >1 '''
    data = np.random.randint(0,5,(10,3))
    with pytest.raises(ValueError):
        class_1 = np.unique(data[:,0])[0]
        class_2 = np.unique(data[:,0])[1]
        probs = {class_1: 1.0, class_2: 0.5}
        class_probability_shift(data, 0, 0, 3, probs)

def test_probability_shift_5():
    ''' Check ValueError when given probabilities for classes not in data '''
    data = np.random.randint(0,5,(10,3))
    with pytest.raises(ValueError):
        probs = {1000: 0.5}
        class_probability_shift(data, 0, 0, 3, probs)

def test_probability_shift_6():
    ''' Check drift injected when not all classes specified '''
    np.random.seed(0)
    data = np.array([
        [0,0,0], [1,0,0], [2,0,0], [3,0,0], [4,0,0], [5,0,0],
        [6,0,0], [7,0,0], [8,0,0], [9,0,0]
    ])
    probs = {k:0.1 for k in range(2,10)}
    new_data = class_probability_shift(data, 0, 1, 9, probs)
    assert np.array_equal(data[0], new_data[0])
    assert np.array_equal(data[-1], new_data[-1])
    assert not np.array_equal(data[1:9], new_data[1:9])

def test_dirichlet_shift_1():
    ''' Ensure Dirichlet shift causes some drift '''
    np.random.seed(0)
    data = np.array([
        [0,0,0], [1,0,0], [2,0,0], [3,0,0], [4,0,0], [5,0,0],
        [6,0,0], [7,0,0], [8,0,0], [9,0,0]
    ])
    alpha = {k:1 for k in range(0,10)}
    new_data = class_dirichlet_shift(data, 0, 1, 9, alpha)
    assert np.array_equal(data[0], new_data[0])
    assert np.array_equal(data[-1], new_data[-1])
    assert not np.array_equal(data[1:9], new_data[1:9])