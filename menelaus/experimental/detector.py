import numpy as np
from typing import List
from toolz import pipe


class Detector:
    def __init__(self, alarm, transforms: List = None):
        self.alarm = alarm
        self.transforms = [] if transforms is None else transforms
        self.rep_reference = None
        self.rep_test = None

    def transform(self, raw_values):
        ret = pipe(raw_values, *self.transforms)
        return ret

    def recalibrate(self, raw_values):
        if self.state == "alarm":
            rep_new = self.transform(raw_values)
            self.rep_test = np.vstack((self.rep_test, rep_new))
        else:
            self.rep_test = self.transform(raw_values)

    @property
    def state(self):
        return self.alarm._state

    def step(self, raw_values):
        if self.rep_reference is None:
            self.rep_reference = self.transform(raw_values)
        else:
            if self.rep_test is None:
                self.rep_test = self.transform(raw_values)
            else:
                self.recalibrate(raw_values)
            self.alarm.evaluate(self.rep_reference, self.rep_test)
