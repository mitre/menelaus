from abc import ABC, abstractclassmethod

import numpy as np
import pandas as pd


class Injector(ABC):
    """
    """
    def _preprocess(self, data, *columns, return_df=False):
        if isinstance(data, np.ndarray):
            self._columns = None
            column_idxs = columns
        elif isinstance(data, pd.DataFrame):
            self._columns = data.columns
            column_idxs = [data.columns.get_loc(c) for c in columns]
        else:
            raise ValueError(f"Data of type {type(data)} not supported")

        copy = np.copy(data)
        if return_df:
            return pd.DataFrame(copy, columns=columns), columns
        else:
            return copy, column_idxs
    
    def _postprocess(self, data):
        if self._columns is not None and not isinstance(data, pd.DataFrame):   
            return pd.DataFrame(data, columns=self._columns)
        else:               
            return data

    @abstractclassmethod
    def __call__(self):
        raise NotImplementedError



