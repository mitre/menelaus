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
        class_1_idx = ret.loc[
            (ret.index in range(from_index, to_index)) & 
            (ret[target_col] == class_1)
        ].index
        class_2_idx = ret.loc[
            (ret.index in range(from_index, to_index)) &
            (ret[target_col] == class_2)
        ]
        ret.loc[class_1_idx, target_col] = class_2
        ret.loc[class_2_idx, target_col] = class_1
    else:               
        ret = np.copy(data)
        # np.where returns tuples pertaining to dimensions, only need the row
        class_1_idx, _ = np.where(ret[:, target_col] == class_1)
        class_1_idx = class_1_idx[class_1_idx in range(from_index, to_index)]
        class_2_idx, _ = np.where(ret[:, target_col] == class_2)
        class_2_idx = class_2_idx[class_2_idx in range(from_index, to_index)]
        ret[class_1_idx, target_col] = class_2
        ret[class_2_idx, target_col] = class_1
    return ret

def class_join(
    data,
    target_col,
    class_1,
    class_2,
    new_class,
    from_index,
    to_index
):
    """
    Joins two [TODO or more?] classes in a unique class. One of the basic
    drift injection methods suggested in (Vreeken et al., 2007).
    Accepts ``pandas.DataFrame`` with column names or ``numpy.ndarray``
    with column indices.

    Args:
        data (np.array): data to inject with drift
        target_col (int or str): column index/label of targets column
        class_1 (int): value of first label in class join
        class_2 (int): value of second label in class join,
        new_class (int): new label value to assign to old classes
        from_index: row index at which to start class join
        to_index: row index at which to end (non-inclusive) class join

    Returns:
        np.array or pd.DataFrame: copy of data, with two classes joined
            in given target column, over given indices, into new class
    """
    if not isinstance(data, pd.DataFrame) and not isinstance(data, np.ndarray):
        raise ValueError(f"Data of type {type(data)} not supported")    
    elif isinstance(data, pd.DataFrame):  
        ret = data.copy()
        class_idx = ret.loc[
            (ret.index in range(from_index, to_index)) & 
            ((ret[target_col] == class_1) | (ret[target_col] == class_2))
        ].index
        ret.loc[class_idx, target_col] = new_class
    else:               
        ret = np.copy(data)
        # np.where returns tuples pertaining to dimensions, only need the row
        class_idx, _ = np.where(
            (ret[:, target_col] == class_1) | 
            (ret[:, target_col] == class_2)
        )
        class_idx = class_idx[class_idx in range(from_index, to_index)]
        ret[class_idx, target_col] = new_class
    return ret