import numpy as np
import pandas as pd


def brownian_noise(data, feature, x0, from_index, to_index):
    """
    Adds Brownian noise to a column of a given dataset. Accepts
    ``pandas.DataFrame`` with column names or ``numpy.ndarray``
    with column indices. Calls ``random_walk`` for full computation.

    Ref. :cite:t:`fields2019mitigating`

    Args:
        data (np.ndarray or pd.DataFrame): data to inject with drift
        feature (int or str): index/label of column to add noise to
        x0 (int): starting point for random walk noise
        from_index (int): row index at which to start injection
        to_index (int): row index at which to end (non-inclusive) injection

    Returns:
        np.ndarray or pd.DataFrame: copy of data, with two columns swapped
            over given indices
    """
    steps = to_index - from_index
    if not isinstance(data, pd.DataFrame) and not isinstance(data, np.ndarray):
        raise ValueError(f"Data of type {type(data)} not supported")
    elif isinstance(data, pd.DataFrame):
        ret = data.copy()
        # unlike other operations, .loc[0:r] is inclusive, so -1
        ret_col = ret.loc[from_index : to_index - 1, feature].copy()
        ret_col = ret_col + random_walk(steps, x0)
        ret.loc[from_index:to_index, feature] = ret_col
    else:
        ret = np.copy(data)
        ret[from_index:to_index, feature] = ret[
            from_index:to_index, feature
        ] + random_walk(steps, x0)
    return ret


def random_walk(steps, x0):
    """
    Performs Weiner process given a number of steps and initial starting point.

    Ref. :cite:t:`fields2019mitigating`

    Args:
        steps (int): number of steps, corresponds to size of output vector
        x0 (int): initial starting value for output vector values
    """
    w = np.ones(steps) * x0
    for i in range(1, steps):
        yi = np.random.choice([1, -1])
        w[i] = w[i - 1] + (yi / np.sqrt(steps))
    return w
