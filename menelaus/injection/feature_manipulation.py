import numpy as np
import pandas as pd


def feature_swap(data, col_1, col_2, from_index, to_index):
    """
    Swaps two features/columns of a given dataset with each other.
    One of the basic drift injection methods suggested in [CITE!].

    Args:
        data (np.array): data to inject with drift
        col_1 (int): column index of first column
        col_2 (int): column index of second column
        from_index: row index at which to start column swap
        to_index: row index at which to end (non-inclusive) column swap

    Returns:
        np.array or pd.DataFrame: copy of data, with two columns swapped
            over given indices
    """
    if not isinstance(data, pd.DataFrame) and not isinstance(data, np.ndarray):
        raise ValueError(f"Data of type {type(data)} not supported")    
    elif isinstance(data, pd.DataFrame):  
        ret = data.copy()
        ret_col_1 = ret.loc[from_index:to_index, col_1].copy()
        ret.loc[from_index:to_index, col_1] = ret.loc[from_index:to_index, col_2]
        ret.loc[from_index:to_index, col_2] = ret_col_1
    else:               
        ret = np.copy(data)
        ret[from_index:to_index, [col_1,col_2]] = ret[from_index:to_index, [col_2,col_1]]
    return ret

def reorder_by_feature():
    raise NotImplemented