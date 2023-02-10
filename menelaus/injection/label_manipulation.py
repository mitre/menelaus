import numpy as np
import pandas as pd

from menelaus.injection.injector import Injector





# region - Simple Label Manipulation

class LabelSwapInjector(Injector):
    """
    """
    def __call__(self, data, target_col, from_index, to_index, class_1, class_2):
        ret = self._preprocess(data)
        class_1_idx = np.where(ret[:, target_col] == class_1)[0]
        class_1_idx = class_1_idx[(class_1_idx < to_index) & (class_1_idx >= from_index)]
        class_2_idx = np.where(ret[:, target_col] == class_2)[0]
        class_2_idx = class_2_idx[(class_2_idx < to_index) & (class_2_idx >= from_index)]
        ret[class_1_idx, target_col] = class_2
        ret[class_2_idx, target_col] = class_1
        ret = self._postprocess(ret)
        return ret


class LabelJoinInjector(Injector):
    """
    """
    def __call__(self, data, target_col, from_index, to_index, class_1, class_2, new_class):
        ret = self._preprocess(data)
        class_idx = np.where((ret[:, target_col] == class_1) | (ret[:, target_col] == class_2))[0]
        class_idx = class_idx[(class_idx < to_index) & (class_idx >= from_index)]
        ret[class_idx, target_col] = new_class
        ret = self._postprocess(ret)
        return ret

# endregion

# region - LTF-Inspired Label Manipulation

class LabelProbabilityInjector(Injector):
    """
    """
    def __call__(self, data, target_col, from_index, to_index, class_probabilities):
        ret = self._preprocess(data)
        
        # determine all unique classes and classes not specified in args
        all_classes = np.unique(ret[:, target_col])
        undefined_classes = [k for k in all_classes if k not in class_probabilities]

        # specified class probabilities must sum to 1 or less
        if sum(class_probabilities.values()) > 1.0:
            raise ValueError(f"Probabilities in {class_probabilities} exceed 1")

        # args should not specify previously unseen classes
        if set(all_classes) != set(list(class_probabilities.keys()) + undefined_classes):
            raise ValueError(
                f"Argument {class_probabilities} has classes not found in data {all_classes}"
            )

        # undefined classes are resampled uniformly
        for uc in undefined_classes:
            class_probabilities[uc] = (1 - sum(class_probabilities.values())) / len(
                undefined_classes
            )

        # distribution for each data point, and reordering of each point by class
        p_distribution = []
        sample_idxs_grouped = []

        # locate each class in window
        for cls in all_classes:
            cls_idx = np.where(data[:, target_col] == cls)[0]
            cls_idx = cls_idx[(cls_idx < to_index) & (cls_idx >= from_index)]

            # each member has p_class / class_size chance, represented as bool to avoid div/0
            p_individual = (
                cls_idx.shape[0] and class_probabilities[cls] / cls_idx.shape[0]
            ) or 0

            # append to grouped array and corresponding distribution
            sample_idxs_grouped.extend(cls_idx)
            p_distribution.extend(np.ones(cls_idx.shape[0]) * p_individual)

        # if classes skipped, ensure probability distribution adds to 1
        p_leftover = (1 - sum(p_distribution)) / len(p_distribution)
        p_distribution = [p + p_leftover for p in p_distribution]

        # shuffled sample over window, with replacement, with weights
        sample_idxs = np.random.choice(
            sample_idxs_grouped, to_index - from_index, True, p_distribution
        )
        ret[from_index:to_index] = data[sample_idxs]
        ret = self._postprocess(ret)
        return ret


class LabelDirichletInjector(Injector):
    """
    """
    def __call__(self, data, target_col, from_index, to_index, alpha):
        alpha_classes = list(alpha.keys())
        alpha_values = [alpha[k] for k in alpha]

        # generate dirichlet distribution by class
        # XXX - minor concern that order of these list-types not always guaranteed
        dirichlet_distribution = np.random.dirichlet(alpha_values)
        dirichlet_probabilities = {alpha_classes[i]: dirichlet_distribution[i] for i in range(len(alpha_classes))}

        # use class_probability_shift with fully-specified distribution
        label_prob_injector = LabelProbabilityInjector()
        return label_prob_injector(data, target_col, from_index, to_index, dirichlet_probabilities)

# endregion


def class_probability_shift(
    data, target_col, from_index, to_index, class_probabilities
):
    """
    Resamples the data over a specified window, with altered probability
    for specified classes (and uniform probability for remaining classes).
    Accepts ``pandas.DataFrame`` with column names or ``numpy.ndarray``
    with column indices.

    Note:
    * this function can perform tweak-one and minority shift
    * When a class is not present in the window specified,
    but specified in ``class_probabilities``, the probability
    value is uniformly divided into the remaining classes in
    the window.

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
   
    pass


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
    pass
