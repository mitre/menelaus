class Alarm():
    def __init__(self, threshold):
        self.threshold = threshold
        self._state = "baseline"

    def evaluate(self):
        raise NotImplementedError

class KullbackLeiblerAlarm(Alarm):
    def evaluate(self, rep_reference, rep_test):
        diff = rep_test - rep_reference
        if diff < self.threshold:       self._state = "baseline"
        elif diff == self.threshold:    self._state = "warning"
        else:                           self._state = "drift"



