import numpy as np
from typing import List
from toolz import pipe


class Detector:
    """
    Basic implementation of a (currently) batch-based drift detection scheme.
    A detector consists of two major components:
        *   a list of partial transform functions which are sequentially
            applied to preprocess the data
        *   an alarm, which operates on the final data representation
            to determine drift state per some algorithm

    The detector can then evaluate at each step when new data is given,
    recalibrate its data representations accordingly, and update its state.
    *This is a different design than ``BatchDetector``, ``StreamingDetector``,
    and is intended to be used with newer NLP capabilities.

    Attributes:
        alarm (Alarm): parameterized object for final data representations
        transforms (list): ordered list of partial or curried functions,
            which transform data from original state to one compatible with
            the alarm tool
        rep_reference: representation for reference data, whose type may vary
        rep_test: representation for test data, whose type may vary
        state (str): describes the most recent evaluation of the detector/alarm
    """

    def __init__(self, alarm, transforms: List = None):
        """
        Args:
            alarm (Alarm): parameterized object for final data representations.
            transforms (list): ordered list of partial or curried functions,
                which transform data from original state to one compatible with
                the alarm tool. Default ``None``.
        """
        self.alarm = alarm
        self.transforms = [] if transforms is None else transforms
        self.rep_reference = None
        self.rep_test = None

    def transform(self, raw_values):
        """
        Execute transforms pipeline on new batch of data.

        Args:
            raw_values (numpy.ndarray or tensorflow.tensor): new batch of data.

        Returns:
            numpy.ndarray : final transformed data, likely (but not always) a
                `numpy.ndarray`.
        """
        ret = pipe(raw_values, *self.transforms)
        return ret

    def recalibrate(self, raw_values):
        """
        Adjust reference and/or test data given new batch. If drift was found,
        test data is incorporated into reference data, and the new batch is
        now the test. Otherwise, the new batch is the test, and the prior test
        data is discarded.

        Args:
            raw_values (numpy.ndarray or tensorflow.tensor): new batch of data.
        """
        rep_new = self.transform(raw_values)
        if self.state == "alarm":
            self.rep_reference = np.vstack((self.rep_reference, self.rep_test))
        self.rep_test = rep_new

    @property
    def state(self):
        """Returns drift state of detector/alarm object as string."""
        return self.alarm._state

    def step(self, raw_values):
        """
        Steps the detector forward by introducing the next batch of data. Fills
        reference and/or test data if they are not yet filled, and performs
        evaluation/recalibration as needed thereafter.

        Args:
            raw_values (numpy.ndarray or tensorflow.tensor): new batch of data.
        """
        if self.rep_reference is None:
            self.rep_reference = self.transform(raw_values)
        else:
            if self.rep_test is None:
                self.rep_test = self.transform(raw_values)
            else:
                self.recalibrate(raw_values)
            self.alarm.evaluate(self.rep_reference, self.rep_test)
