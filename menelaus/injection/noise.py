import numpy as np
import pandas as pd

def brownian_noise(data, feature, x0, from_index, to_index):
    """
    """
    steps = to_index - from_index
    if not isinstance(data, pd.DataFrame) and not isinstance(data, np.ndarray):
        raise ValueError(f"Data of type {type(data)} not supported")    
    elif isinstance(data, pd.DataFrame):  
        ret = data.copy()
        ret_col = ret.loc[from_index:to_index, feature].copy()
        ret_col += random_walk(steps, x0)
        ret.loc[from_index:to_index, feature] = ret_col
    else:               
        ret = np.copy(data)
        ret[from_index:to_index, feature] += random_walk(steps, x0)
    return ret

def random_walk(steps, x0):
    """
    """
    w = np.ones(steps)*x0
    for i in range(1,steps):
        yi = np.random.choice([1,-1])
        w[i] = w[i-1] + (yi / np.sqrt(steps))
    return w