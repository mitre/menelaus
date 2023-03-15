import pytest
import numpy as np


from menelaus.injection import LabelJoinInjector, LabelSwapInjector, LabelProbabilityInjector, LabelDirichletInjector


def test_join_1():
    ''' Check correct join behavior '''
    i = LabelJoinInjector()
    data = np.array([[0,5], [0,5], [1,5], [1,5]])
    expected_data = np.array([[0,5], [2,5], [2,5], [1,5]])
    copy = i(data, from_index=1, to_index=3, target_col=0, class_1=0, class_2=1, new_class=2)
    assert np.array_equal(copy, expected_data)

def test_swap_1():
    ''' Check correct swap in pd.DataFrame '''
    i = LabelSwapInjector()
    data = np.array([[0,5], [0,5], [1,6], [1,6]])
    expected_data = np.array([[1,5], [1,5], [0,6], [0,6]])
    copy = i(data, from_index=0, to_index=4, target_col=0, class_1=0, class_2=1)
    assert np.array_equal(copy, expected_data)

def test_probability_shift_1():
    ''' Ensure probability shift causes some drift in numpy.ndarray '''
    i = LabelProbabilityInjector()
    data = np.array([
        [0,0,0], [1,0,0], [2,0,0], [3,0,0], [4,0,0], [5,0,0],
        [6,0,0], [7,0,0], [8,0,0], [9,0,0]
    ])
    probs = {k:0.1 for k in range(0,10)}
    np.random.seed(0)
    new_data = i(data, from_index=1, to_index=9, target_col=0, class_probabilities=probs)
    assert np.array_equal(data[0], new_data[0])
    assert np.array_equal(data[-1], new_data[-1])
    assert not np.array_equal(data[1:9], new_data[1:9])

def test_probability_shift_2():
    ''' Check drift injected when not all classes specified '''
    i = LabelProbabilityInjector()
    data = np.array([
        [0,0,0], [1,0,0], [2,0,0], [3,0,0], [4,0,0], [5,0,0],
        [6,0,0], [7,0,0], [8,0,0], [9,0,0]
    ])
    probs = {k:0.075 for k in range(2,10)}
    np.random.seed(0)
    _ = i(data, from_index=0, to_index=10, target_col=0, class_probabilities=probs)
    # individual = class prob / class size, + leftover prob from unspecified classes
    # 1-(8*0.075)=0.4, classes 0, 1 will have prob 0.2 each
    # class sizes are 1, expected distr is simple
    expected_distr = [0.2, 0.2, 0.075, 0.075, 0.075, 0.075, 0.075, 0.075, 0.075, 0.075]
    p_distr = np.round(i._p_distribution, decimals=3)
    assert np.array_equal(p_distr, expected_distr)

def test_probability_shift_4():
    ''' Check ValueError when probabilities add to >1 '''
    i = LabelProbabilityInjector()
    data = np.random.randint(0,5,(10,3))
    with pytest.raises(ValueError):
        class_1 = np.unique(data[:,0])[0]
        class_2 = np.unique(data[:,0])[1]
        probs = {class_1: 1.0, class_2: 0.5}
        i(data, from_index=0, to_index=3, target_col=0, class_probabilities=probs)

def test_probability_shift_5():
    ''' Check ValueError when given probabilities for classes not in data '''
    i = LabelProbabilityInjector()
    data = np.random.randint(0,5,(10,3))
    with pytest.raises(ValueError):
        probs = {1000: 0.5}
        i(data, from_index=0, to_index=3, target_col=0, class_probabilities=probs)

def test_dirichlet_shift_1():
    ''' Ensure Dirichlet shift causes some drift '''
    i = LabelDirichletInjector()
    data = np.array([
        [0,0,0], [1,0,0], [2,0,0], [3,0,0], [4,0,0], [5,0,0],
        [6,0,0], [7,0,0], [8,0,0], [9,0,0]
    ])
    alpha = {k:1 for k in range(0,10)}
    np.random.seed(0)
    new_data = i(data, from_index=1, to_index=9, target_col=0, alpha=alpha)
    assert np.array_equal(data[0], new_data[0])
    assert np.array_equal(data[-1], new_data[-1])
    assert not np.array_equal(data[1:9], new_data[1:9])