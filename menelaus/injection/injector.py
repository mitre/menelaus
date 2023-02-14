from abc import ABC, abstractclassmethod

import numpy as np
import pandas as pd


class Injector(ABC):
    """
    Abstract base class for drift injection callables. Classes using this
    pattern should implement a ``__call__`` function, and also make use of the 
    ``_preprocess()`` and ``_postprocess`` type handling capabilities of this
    class. 

    When called, an ``Injector`` typically accepts either ``pandas.DataFrame``
    or ``numpy.ndarray`` data, along with other parameters for the injection
    technique it is modeling, and returns the data with drift injected over a
    specified window. 
    """
    def _preprocess(self, data, *columns, return_df=False):
        """
        Preprocesses data and column indicators into desired format. By default
        returns data as ``numpy.ndarray`` and columns as integer column indices
        unless told otherwise. 

        Args:
            data (numpy.ndarray or pd.DataFrame): data to preprocess
            *columns (str or int): 1 or more integers or names corresponding to
                columns in the data, to also be preprocessed
            return_df (bool): Whether to return data as ``pandas.DataFrame``,
                in which case ``*columns`` are returned as-is. Default False.

        Returns:
            tuple: first item is a deep-copy of the data in ``numpy.ndarray``,
                unless ``return_df`` is True, in which case a 
                ``pandas.DataFrame`` is returned. Second item is the column
                indicators as integer indexes, unless ``return_df`` is True, in
                which case the column indicators should be strings and are
                returned as-is.
        """
        # store columns if DF, otherwise set to None
        if isinstance(data, np.ndarray):
            self._columns = None
            column_idxs = columns

        # column str names to integer idxs if needed
        elif isinstance(data, pd.DataFrame):
            self._columns = data.columns
            column_idxs = [data.columns.get_loc(c) for c in columns]

        # only two types supported
        else:
            raise ValueError(f"Data of type {type(data)} not supported")

        # copy and return desired type
        copy = np.copy(data)
        if return_df:
            return pd.DataFrame(copy, columns=columns), columns
        else:
            return copy, column_idxs
    
    def _postprocess(self, data):
        """
        Postprocesses data and returns as its initial type, which was
        determined and recorded during preprocessing. 

        Args:
            data (``numpy.ndarray`` or ``pandas.DataFrame``): data to
                postprocess

        Returns:
            data (``numpy.ndarray`` ``or pandas.DataFrame``): data,
                transformed into its original state
        """
        # if column names were stored and data is not already DF, return DF
        if self._columns is not None and not isinstance(data, pd.DataFrame):   
            return pd.DataFrame(data, columns=self._columns)

        # if column names not stored and data is DF, return original ndarray
        elif self._columns is None and isinstance(data, pd.DataFrame):
            return data.to_numpy()
        
        # if column names stored and data is DF, return unchanged
        # if column names not stored and data is ndarray, return unchanged
        else:
            return data

    @abstractclassmethod
    def __call__(self):
        """ Implemented by sub-classes. """
        raise NotImplementedError



