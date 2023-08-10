class Alarm():
    def __init__(self, threshold):
        self.threshold = threshold
        self._state = "baseline"

    def evaluate(self, rep_reference, rep_test):
        # parent fn. can be, check threshold crossed
        # can reuse a lot of code for different divergence metrics I feel
        pass

class KullbackLeiblerAlarm(Alarm):
    def evaluate(self, rep_reference, rep_test):
        self._state = "warning"
        self._state = "drift"



