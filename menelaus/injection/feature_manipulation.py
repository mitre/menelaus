import numpy as np


def feature_swap(data, col_1, col_2):
    """
    Swaps two features/columns of a given dataset with each other.
    One of the basic drift injection methods suggested in [CITE!].
    *Support for ``pandas`` TBD*.

    Args:
        data (np.array): data to inject with drift
        col_1 (int): column index of first column
        col_2 (int): column index of second column

    Returns:
        np.array: copy of data, with two columns swapped
    """
    ret = np.copy(data)
    ret[:, [col_1, col_2]] = ret[:, [col_2, col_1]]
    return ret

def reorder_by_feature():
    raise NotImplemented