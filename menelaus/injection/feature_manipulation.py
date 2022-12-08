import numpy as np
import pandas as pd


def feature_swap(data, col_1, col_2, from_index, to_index):
    """
    Swaps two features/columns of a given dataset with each other.
    Accepts ``pandas.DataFrame`` with column names or ``numpy.ndarray``
    with column indices.

    Ref. :cite:t:`TODO`

    Args:
        data (np.ndarray or pd.DataFrame): data to inject with drift
        col_1 (int or str): column index/label of first column
        col_2 (int or str): column index/label of second column
        from_index (int): row index at which to start column swap
        to_index (int): row index at which to end (non-inclusive) column swap

    Returns:
        np.ndarray or pd.DataFrame: copy of data, with two columns swapped
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

def feature_hide_and_sample(data, col, sample_size, random_state=0):
    """
    Hides a feature, then treats it as a shared concept by which to group the data.
    Afterwards samples are uniformly drawn from each group. Accepts
    ``pandas.DataFrame`` with column names or ``numpy.ndarray`` with column indices. 

    Ref. :cite:t:`TODO`

    Args:
        data (np.ndarray or pd.DataFrame): data to inject with drift
        col (int or str): index/label of column to hide and re-sample (note this
            should be a categorical feature that can be treated as a concept)
        sample_size (int): data points to be drawn from each group in new concept
        random_state (int): optional random seed. Default 0

    Returns:
        np.ndarray or pd.DataFrame: copy of data, grouped by indicated column,
            with each group sampled and column removed
    """
    if not isinstance(data, pd.DataFrame) and not isinstance(data, np.ndarray):
        raise ValueError(f"Data of type {type(data)} not supported")    
    elif isinstance(data, pd.DataFrame):  
        df = data.copy()
    else:               
        df = np.copy(data)
        df = pd.DataFrame(df)
    ret = df.groupby(col).sample(n=sample_size, random_state=random_state)
    ret = ret.drop(columns=[col]).reset_index(drop=True)
    ret = ret if isinstance(data, pd.DataFrame) else ret.to_numpy()
    return ret