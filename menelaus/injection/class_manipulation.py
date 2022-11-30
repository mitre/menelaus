import numpy as np
import pandas as pd


def class_swap(data, target_col, class_1, class_2, from_index, to_index):
    """
    Swaps two classes in a target column of a given dataset with each other.
    One of the basic drift injection methods suggested in [CITE!].
    Accepts ``pandas.DataFrame`` with column names or ``numpy.ndarray``
    with column indices.

    Args:
        data (np.array): data to inject with drift
        target_col (int or str): column index/label of targets column
        class_1 (int): value of first label in class swap
        class_2 (int): value of second label in class swap
        from_index: row index at which to start class swap
        to_index: row index at which to end (non-inclusive) class swap

    Returns:
        np.array or pd.DataFrame: copy of data, with two classes swapped
            in given target column, over given indices
    """
    if not isinstance(data, pd.DataFrame) and not isinstance(data, np.ndarray):
        raise ValueError(f"Data of type {type(data)} not supported")    
    elif isinstance(data, pd.DataFrame):  
        ret = data.copy()
        class_1_idx = ret[target_col] == class_1
        class_2_idx = ret[target_col] == class_2
        ret.loc[class_1_idx, target_col] = class_2
        ret.loc[class_2_idx, target_col] = class_1
    else:               
        ret = np.copy(data)
        class_1_idx = np.where(ret[target_col] == class_1)
        class_2_idx = np.where(ret[target_col] == class_2)
        ret[class_1_idx, target_col] = class_2
        ret[class_2_idx, target_col] = class_1
    return ret

def class_join():
    raise NotImplemented