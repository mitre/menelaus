""" Module to test noise injection functions """

import pandas as pd
import numpy as np

from menelaus.injection import BrownianNoiseInjector


def test_brownian_noise_1():
    ''' Check RW noise injected correctly '''
    i = BrownianNoiseInjector()
    exp_data = np.array([[0],[1],[2],[3],[4],[5],[6]])
    copy = i(exp_data, from_index=1, to_index=5, col=0, x0=0, random_state=0)
    
    # copy of random walk function
    np.random.seed(0)
    w = np.ones(5-1) * 0
    for i in range(1,5-1):
        yi = np.random.choice([1,-1])
        w[i] = w[i-1] + (yi / np.sqrt(5-1))
    
    exp_data[1:5, 0] = exp_data[1:5, 0] + w

    # XXX - repeated code for testing random walk, no good alternatives
    assert np.array_equal(exp_data, copy)

def test_random_walk():
    ''' Check random walk values '''
    vec = BrownianNoiseInjector._random_walk(steps=5, x0=0, random_state=0)

    # copy of random walk function
    np.random.seed(0)
    w = np.ones(5) * 0
    for i in range(1, 5):
        yi = np.random.choice([1, -1])
        w[i] = w[i - 1] + (yi / np.sqrt(5))
    
    # XXX - repeating code for test coverage, although this may help if
    #       implementation of random walk changes over time
    assert np.array_equal(w, vec)

