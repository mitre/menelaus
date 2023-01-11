import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split


# region - simple class manipulation functions 

def class_swap(data, target_col, class_1, class_2, from_index, to_index):
    """
    Swaps two classes in a target column of a given dataset with each other.
    Accepts ``pandas.DataFrame`` with column names or ``numpy.ndarray``
    with column indices.

    Ref. :cite:t:`souza2020challenges`

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
            (ret.index < to_index)
            & (ret.index >= from_index)
            & (ret[target_col] == class_1)
        ].index
        class_2_idx = ret.loc[
            (ret.index < to_index)
            & (ret.index >= from_index)
            & (ret[target_col] == class_2)
        ].index
        ret.loc[class_1_idx, target_col] = class_2
        ret.loc[class_2_idx, target_col] = class_1
    else:
        ret = np.copy(data)
        # np.where returns tuples pertaining to dimensions, only need the row
        class_1_idx = np.where(ret[:, target_col] == class_1)[0]
        class_1_idx = class_1_idx[
            (class_1_idx < to_index) & (class_1_idx >= from_index)
        ]
        class_2_idx = np.where(ret[:, target_col] == class_2)[0]
        class_2_idx = class_2_idx[
            (class_2_idx < to_index) & (class_2_idx >= from_index)
        ]
        ret[class_1_idx, target_col] = class_2
        ret[class_2_idx, target_col] = class_1
    return ret


def class_join(data, target_col, class_1, class_2, new_class, from_index, to_index):
    """
    Joins two [TODO or more?] classes in a unique class. Accepts
    ``pandas.DataFrame`` with column names or ``numpy.ndarray``
    with column indices.

    Ref. :cite:t:`souza2020challenges`

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
            (ret.index < to_index)
            & (ret.index >= from_index)
            & ((ret[target_col] == class_1) | (ret[target_col] == class_2))
        ].index
        ret.loc[class_idx, target_col] = new_class
    else:
        ret = np.copy(data)
        # np.where returns tuples pertaining to dimensions, only need the row
        class_idx = np.where(
            (ret[:, target_col] == class_1) | (ret[:, target_col] == class_2)
        )[0]
        class_idx = class_idx[(class_idx < to_index) & (class_idx >= from_index)]
        ret[class_idx, target_col] = new_class
    return ret

# endregion


# region - LTF-inspired class manipulation functions

def tweak_one_shift(X, y, shifted_class, shift_p, val_size, test_size):
    """
    TODO - Currently only supports ``np.array`` data - fix?

    Description. 
    
    but first split into 3 and sample with replacement for all
    assign p to one class, uniform to rest

    Args:
        X (np.array): input feature data
        y (np.array): corresponding targets to inject with drift
        shifted_class (int): TBD
        shift_p (float):  TBD 
        val_size (float): proportion of data to allocate for validation
        test_size (float): proportion of data to allocate for testing
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_size)
    
    for x, targets in (X,y):
        x = x+0
        targets = targets+0 

    return X_train, X_val, X_test, y_train, y_val, y_test
    # pf


def minority_class_shift():
    """
    split into train/test and sample from each with replacement
    to create smaller subsets??
    20/30/40/50% of classes are set to 0.001
    while ratios of other classes are uniform
    """
    pass


def dirichlet_shift():
    """ 
    """
    pass

# endregion