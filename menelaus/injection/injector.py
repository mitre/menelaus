from abc import ABC, abstractclassmethod

import numpy as np
import pandas as pd


class Injector(ABC):
    """
    """
    def _preprocess(self, data):
        if isinstance(data, np.ndarray):
            self._columns = None
            self._data = np.copy(data)
        elif isinstance(data, pd.DataFrame):
            self._columns = data.columns
            self._data = np.copy(data.to_numpy())
        else:
            raise ValueError(f"Data of type {type(data)} not supported")
    
    def _postprocess(self, data):
        if self._columns is not None:   
            return pd.DataFrame(data, columns=self._columns)
        else:               
            return data

    @abstractclassmethod
    def __call__(self):
        raise NotImplementedError



