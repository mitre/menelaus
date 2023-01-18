import numpy as np
import pandas as pd


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

def class_probability_shift(data, target_col, classes, p_classes, from_index, to_index):
    """
    """
    ret = np.copy(data)
    classes = np.unique(ret[:, target_col])

    # distribution for each data point, and reordering of each point by class
    p_distribution = np.array([])
    sample_grouped = np.array([])

    # locate all classes to shift
    shifted_classes_idx = np.where(data[:, target_col].isin(classes))[0]

    # locate each class in window
    for cls in classes:
        cls_idx = np.where(data[:, target_col] == cls)[0]
        cls_idx = cls_idx[
            (cls_idx < to_index) &
            (cls_idx >= from_index)
        ]

        if cls in classes:
            # pts from shifted classes drawn with separate probability
            p_individual = p_classes / shifted_classes_idx.shape[0]
        else:
            # pts from remaining classes drawn with uniform probability
            p_individual = ((1 - p_classes) / (classes.shape[0] - 1)) / cls_idx.shape[0]

        # append to grouped array and corresponding distribution
        sample_grouped = np.concatenate((sample_grouped, data[cls_idx]))
        p_distribution = np.concatenate(p_distribution, np.ones(cls_idx.shape[0]) * p_individual)

    # shuffled sample over window, with replacement, with weights
    sample = np.random.choice(sample_grouped, to_index-from_index, True, p_distribution)
    ret[from_index:to_index] = sample
    return ret


def class_dirichlet_shift(data, target_col, alpha, from_index, to_index):
    """ 
    numpy has dirichlet shift
    3.	Dirichlet shift:  (Doesn't appear to be implemented in os code)
    b.	Sample from each with replacement to create smaller subsets
    c.	"we draw p(y) from a Dirichlet distribution with concentration parameter α. With uniform p(y), Dirichlet shift is bigger for smaller α."

    alpha = (,,) is relative proportion for each class
    make alpha a dict
    sort by value
    each number / total of numbers is prob for class
    """
    ret = np.copy(data)

    # generate dirichlet distribution by class
    alpha_list = [alpha[k] for k in alpha]
    dirichlet_distribution = np.random.dirichlet(alpha_list)

    # distribution for each data point, and reordering of each point by class
    p_distribution = np.array([])
    sample_grouped = np.array([])

    # locate each class in window
    for i, cls in enumerate(alpha.keys()):
        cls_idx = np.where(data[:, target_col] == cls)[0]
        cls_idx = cls_idx[
            (cls_idx < to_index) &
            (cls_idx >= from_index)
        ]

        # pts for each class drawn with corresponding probability
        p_individual = dirichlet_distribution[i] / cls_idx.shape[0]
        
        # append to grouped array and corresponding distribution
        sample_grouped = np.concatenate((sample_grouped, data[cls_idx]))
        p_distribution = np.concatenate(p_distribution, np.ones(cls_idx.shape[0]) * p_individual)

    # shuffled sample over window, with replacement, with weights
    sample = np.random.choice(sample_grouped, to_index-from_index, True, p_distribution)
    ret[from_index:to_index] = sample
    return ret

# endregion