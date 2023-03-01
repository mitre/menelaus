import numpy as np

from menelaus.injection.injector import Injector


class FeatureShiftInjector(Injector):
    """
    Shifts a column in a dataset by increasing it according to a formula:
    * ``column = column + (shift_factor * (alpha + mean_column))``

    The alpha is a small value used to inject drift even if the mean is 0.
    """

    def __call__(self, data, from_index, to_index, col, shift_factor, alpha=0.001):
        """
        Args:
            data (np.ndarray or pd.DataFrame): data to inject with drift
            from_index (int): row index at which to start shift
            to_index (int): row index at which to end (non-inclusive) shift
            col (int or str): column index/name of column to shift
            shift_factor (float): percentage of mean by which to shift data
            alpha (float): small initial value to add to shift amount, in case mean is 0. Default 0.001

        Returns:
            np.ndarray or pd.DataFrame: copy of data, with two columns swapped
                over given indices
        """
        # handle type
        ret, (col,) = self._preprocess(data, col)

        # add shift
        self._section_mean = np.mean(ret[from_index:to_index, col])
        self._delta = (alpha + self._section_mean) * shift_factor
        ret[from_index:to_index, col] = np.add(
            ret[from_index:to_index, col], self._delta
        )

        # handle type and return
        ret = self._postprocess(ret)
        return ret


class FeatureSwapInjector(Injector):
    """
    Swaps two features/columns of a given dataset with each other.
    Accepts ``pandas.DataFrame`` with column names or ``numpy.ndarray``
    with column indices.

    Ref. :cite:t:`souza2020challenges`
    """

    def __call__(self, data, from_index, to_index, col_1, col_2):
        """
        Args:
            data (np.ndarray or pd.DataFrame): data to inject with drift
            from_index (int): row index at which to start column swap
            to_index (int): row index at which to end (non-inclusive) column swap
            col_1 (int or str): column index/name of first column
            col_2 (int or str): column index/name of second column

        Returns:
            np.ndarray or pd.DataFrame: copy of data, with two columns swapped
                over given indices
        """
        # handle type
        ret, (col_1, col_2) = self._preprocess(data, col_1, col_2)

        # swap columns
        ret[from_index:to_index, [col_1, col_2]] = ret[
            from_index:to_index, [col_2, col_1]
        ]

        # handle type and return
        ret = self._postprocess(ret)
        return ret


class FeatureCoverInjector(Injector):
    """
    Hides a feature, then treats it as a shared concept by which to group the data.
    Afterwards samples are uniformly drawn from each group. Accepts
    ``pandas.DataFrame`` with column names or ``numpy.ndarray`` with column indices.

    Note:
    * This function cannot hide a feature over a window, as the appearance of the
        column outside the window may raise errors.

    Ref. :cite:t:`souza2020challenges`
    """

    def __call__(self, data, col, sample_size, random_state=None):
        """
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
        # handle type
        ret, (col,) = self._preprocess(data, col, return_df=True)

        # hide and reorder
        ret = ret.groupby(col).sample(n=sample_size, random_state=random_state)
        ret = ret.drop(columns=[col]).reset_index(drop=True)

        # handle type and return
        ret = self._postprocess(ret)
        return ret
