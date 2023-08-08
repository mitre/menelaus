""""""

class Alarm():
    def __init__(self, threshold):
        self.threshold = threshold
        self._state = "baseline"

    def compare(self):
        # either implement e.g. KL Divergence
        # or just call compare on whatever tool you're using
        pass

    def evaluate(self):
        # parent fn. can be, check threshold crossed
        # can reuse a lot of code for different divergence metrics I feel
        pass


class Detector():
    def __init__(self, representation, alarm):
        self.representation = representation
        self.alarm = alarm

    def update(self, data):
        pass

    def run(self):
        pass

    @property
    def state(self):
        return self.alarm._state