import numpy as np
import pandas as pd
import pytest

from menelaus.injection import Injector, LabelSwapInjector


# override this attribute so that it is instantiable
# hacky, but lets us reach 100% coverage
Injector.__abstractmethods__ = set()


def test_preprocess_1():
    ''' Check behavior when ndarray input, ndarray desired output, column processing needed '''
    i = LabelSwapInjector()
    data = np.random.randint(0,5,(4,4))
    copy, cols = i._preprocess(data, 0, 1)
    assert np.array_equal(data, copy)
    assert cols == (0,1)

def test_preprocess_2():
    ''' Check behavior when ndarray input, dataframe desired output, column processing needed '''
    i = LabelSwapInjector()
    data = np.ones((2,2))
    copy, cols = i._preprocess(data, 0, 1, return_df=True)
    expected_copy = pd.DataFrame({0: [1.0,1.0], 1: [1.0,1.0]})
    assert expected_copy.equals(copy)
    assert cols == (0,1)

def test_preprocess_3():
    ''' Check behavior when pandas input, ndarray desired output, column processing needed '''
    i = LabelSwapInjector()
    data = pd.DataFrame({'a':[0,0], 'b':[1,2]})
    copy, cols = i._preprocess(data, 'a', 'b')
    expected_copy = np.array([[0,1], [0,2]])
    assert np.array_equal(copy, expected_copy)
    assert cols == (0,1)

def test_preprocess_4():
    ''' Check behavior when pandas input, pandas desired output, column processing needed '''
    i = LabelSwapInjector()
    data = pd.DataFrame({'a':[0,0], 'b':[1,2]})
    copy, cols = i._preprocess(data, 'a', 'b', return_df=True)
    assert copy.equals(pd.DataFrame({'a':[0,0], 'b':[1,2]}))
    assert cols == ('a', 'b')

def test_preprocess_5():
    ''' Check preprocessing failure on non-DF/ndarray data '''
    i = LabelSwapInjector()
    with pytest.raises(ValueError):
        i._preprocess([[1,2], [2,3]], 'col_0')

def test_call_1():
    ''' Check Injector ABC has no call implementation '''
    i = Injector()
    with pytest.raises(NotImplementedError):
        i()

def test_postprocess_1():
    ''' Check behavior when initial data pandas, input data numpy'''
    i = LabelSwapInjector()
    i._columns = ['a', 'b']
    inp = np.ones((2,2))
    exp = pd.DataFrame({'a':[1.0,1.0], 'b':[1.0,1.0]})
    assert i._postprocess(inp).equals(exp)

def test_postprocess_2():
    ''' Check behavior when initial data ndarray, input data dataframe '''
    i = LabelSwapInjector()
    i._columns = None
    inp = pd.DataFrame({'a':[1.0,1.0], 'b':[1.0,1.0]})
    assert np.array_equal(i._postprocess(inp), np.ones((2,2)))

def test_postprocess_3():
    ''' Check behavior when initial data pandas, input data pandas '''
    i = LabelSwapInjector()
    i._columns = ['a', 'b']
    inp = pd.DataFrame({'a':[1.0,1.0], 'b':[1.0,1.0]})
    assert i._postprocess(inp).equals(inp)

def test_postprocess_4():
    ''' Check behavior when initial data numpy, input data numpy '''
    i = LabelSwapInjector()
    i._columns = None
    inp = np.ones((2,2))
    assert np.array_equal(inp, i._postprocess(inp))