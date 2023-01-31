import numpy as np
import pandas as pd


# region - simple class manipulation functions 

def class_swap(data, target_col, from_index, to_index, class_1, class_2):
    """
    Swaps two classes in a target column of a given dataset with each other.
    Accepts ``pandas.DataFrame`` with column names or ``numpy.ndarray``
    with column indices.

    Ref. :cite:t:`souza2020challenges`

    Args:
        data (np.array): data to inject with drift
        target_col (int or str): column index/label of targets column
        from_index: row index at which to start class swap
        to_index: row index at which to end (non-inclusive) class swap
        class_1 (int): value of first label in class swap
        class_2 (int): value of second label in class swap

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


def class_join(data, target_col, from_index, to_index, class_1, class_2, new_class):
    """
    Joins two [TODO or more?] classes in a unique class. Accepts
    ``pandas.DataFrame`` with column names or ``numpy.ndarray``
    with column indices.

    Ref. :cite:t:`souza2020challenges`

    Args:
        data (np.array): data to inject with drift
        target_col (int or str): column index/label of targets column
        from_index: row index at which to start class join
        to_index: row index at which to end (non-inclusive) class join
        class_1 (int): value of first label in class join
        class_2 (int): value of second label in class join,
        new_class (int): new label value to assign to old classes

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

def class_probability_shift(data, target_col, from_index, to_index, class_probabilities):
    """
    Resamples the data over a specified window, with altered probability
    for specified classes (and uniform probability for remaining classes).
    Accepts ``pandas.DataFrame`` with column names or ``numpy.ndarray``
    with column indices.

    Note: this function can perform tweak-one and minority shift.

    Ref. :cite:t:`LTFmethods`

    Args:
        data (np.array): data to inject with drift
        target_col (int or str): column index/label of targets column
        from_index (int): row index at which to start class swap
        to_index (int): row index at which to end (non-inclusive) class swap
        class_probabilities (dict): classes as keys, and their desired
            resampling chance as values. Un-specified classes are given
            a uniform resampling chance with respect to all other
            un-specified classes

    Returns:
        np.array or pd.DataFrame: copy of data, resampled with shifted
            class probability for 1 or more desired classes
    """
    # if pandas.DataFrame, save columns and convert to numpy
    columns = None
    if isinstance(data, pd.DataFrame):
        ret = data.to_numpy()
        columns = data.columns
    elif isinstance(data, np.array):
        ret = np.copy(data)
    else:
        raise ValueError(f"Data of type {type(data)} not supported")

    # determine all unique classes and classes not specified in args
    all_classes = np.unique(ret[:, target_col])
    undefined_classes = [k for k in all_classes if k not in class_probabilities]

    # specified class probabilities must sum to 1 or less
    if sum(class_probabilities.values) > 1.0:
        raise ValueError(f"Probabilities in {class_probabilities} exceed 1")

    # args should not specify previously unseen classes
    if set(all_classes) != set(list(class_probabilities.keys()) + undefined_classes):
        raise ValueError(f"Argument {class_probabilities} has classes not found in data {all_classes}")

    # undefined classes are resampled uniformly
    p_uniform = (1-sum(class_probabilities.values)) / len(undefined_classes)
    for uc in undefined_classes:
        class_probabilities[uc] = p_uniform

    # distribution for each data point, and reordering of each point by class
    p_distribution = np.array([])
    sample_grouped = np.array([])

    # locate each class in window
    for cls in all_classes:
        cls_idx = np.where(data[:, target_col] == cls)[0]
        cls_idx = cls_idx[
            (cls_idx < to_index) &
            (cls_idx >= from_index)
        ]

        # each member of class has p_class / class_size chance
        p_individual = class_probabilities[cls] / cls_idx.shape[0]
        
        # append to grouped array and corresponding distribution
        sample_grouped = np.concatenate((sample_grouped, data[cls_idx]))
        p_distribution = np.concatenate(p_distribution, np.ones(cls_idx.shape[0]) * p_individual)

    # shuffled sample over window, with replacement, with weights
    sample = np.random.choice(sample_grouped, to_index-from_index, True, p_distribution)
    ret[from_index:to_index] = sample

    # back to DF if needed
    if isinstance(data, pd.DataFrame):
        ret = pd.DataFrame(ret)
        ret.columns = columns
    
    return ret


def class_dirichlet_shift(data, target_col, from_index, to_index, alpha):
    """ 
    Resamples the data over a specified window, per a generated Dirichlet
    distribution (with specified alpha) over all labels.
    Accepts ``pandas.DataFrame`` with column names or ``numpy.ndarray``
    with column indices.

    Note: If all labels are not given weights, unexpected behavior may
    cause all un-specified classes to be given uniform resampling chance.

    Ref. :cite:t:`LTFmethods`

    Args:
        data (np.array): data to inject with drift
        target_col (int or str): column index/label of targets column
        from_index: row index at which to start class swap
        to_index: row index at which to end (non-inclusive) class swap
        alpha (dict): used to derive alpha parameter for Dirichlet
            distribution. Keys are ALL labels, values are the desired
            average weight (typically ``int``) per label when resampling. 
            For example, weights of [4,1] correspond to an expected 80/20
            percent split between first and second classes.

    Returns:
        np.array or pd.DataFrame: copy of data, resampled per Dirichlet
            distribution over classes with specified alpha
    """    
    alpha_classes = list(alpha.keys())
    alpha_values = [alpha[k] for k in alpha]

    # generate dirichlet distribution by class
    # XXX - minor concern that order of these list-types not always guaranteed
    dirichlet_distribution = np.random.dirichlet(alpha_values)
    dirichlet_probabilities = {
        alpha_classes[i]:dirichlet_distribution[i] 
        for i in range(len(alpha_classes))
    }

    # use class_probability_shift with fully-specified distribution
    return class_probability_shift(data, target_col, from_index, to_index, dirichlet_probabilities)

# endregion