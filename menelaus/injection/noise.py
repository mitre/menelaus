import numpy as np

from menelaus.injection.injector import Injector


class BrownianNoiseInjector(Injector):
    """
    Adds Brownian noise to a column of a given dataset. Accepts
    ``pandas.DataFrame`` with column names or ``numpy.ndarray``
    with column indices. Calls ``random_walk`` for full computation.

    Ref. :cite:t:`fields2019mitigating`
    """
    def __call__(self, data, from_index, to_index, col, x0, random_state=None):
        """
        Args:
            data (np.ndarray or pd.DataFrame): data to inject with drift
            from_index (int): row index at which to start injection
            to_index (int): row index at which to end (non-inclusive) injection
            col (int or str): index/name of column to add noise to
            x0 (int): starting point for random walk noise
            random_state (int): optional value for setting seed. Default None.

        Returns:
            np.ndarray or pd.DataFrame: copy of data, with Brownian noise
                injected over given indices
        """
        # handle type
        ret, (col,) = self._preprocess(data, col)

        # add RW noise
        steps = to_index - from_index
        ret[from_index:to_index, col] = ret[
            from_index:to_index, col
        ] + BrownianNoiseInjector._random_walk(steps, x0, random_state=random_state)

        # handle type and return
        ret = self._postprocess(ret)
        return ret

    @staticmethod
    def _random_walk(steps, x0, random_state=None):
        """
        Performs Weiner process given a number of steps and initial starting point.

        Ref. :cite:t:`fields2019mitigating`

        Args:
            steps (int): number of steps, corresponds to size of output vector
            x0 (int): initial starting value for output vector values
            random_state (int): optional value for setting seed. Default None.
        """
        # set seed if given
        np.random.seed(random_state)
        
        # perform random walk
        w = np.ones(steps) * x0
        for i in range(1, steps):
            yi = np.random.choice([1, -1])
            w[i] = w[i - 1] + (yi / np.sqrt(steps))
        return w
