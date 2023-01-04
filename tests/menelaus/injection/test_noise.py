""" Module to test noise injection functions """

import pytest
import pandas as pd
import numpy as np

from menelaus.injection import brownian_noise, random_walk


def test_brownian_noise_1():
    ''' Check general behavior for pandas DataFrame'''
    old_values = [0,1,2,3,4,5,6]
    data = pd.DataFrame({'a': old_values})
    data = brownian_noise(data, 'a', 0, 1, 5)
    assert not data.equals(pd.Series(old_values))

def test_brownian_noise_2():
    ''' Check general behavior for numpy array '''
    old_values = np.array([[0],[1],[2],[3],[4],[5],[6]])
    data = np.copy(old_values)
    data = brownian_noise(data, 0, 0, 1, 5)
    assert not np.array_equal(data, old_values)

def test_bronwian_noise_3():
    ''' Check ValueError if data neither pandas DataFrame nor numpy array '''
    data = [[1,2], [0,1]]
    with pytest.raises(ValueError):
        brownian_noise(data, 0, 0, 0, 2)

def test_random_walk():
    ''' Check random walk generally behaves within bounds '''
    vec = random_walk(10, 0)
    for i in range(1,10):
        assert vec[i] <= vec[i-1] + 1/np.sqrt(10)
        assert vec[i] >= vec[i-1] - 1/np.sqrt(10)

