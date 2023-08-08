"""  """

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

# or should representation have compare(other)
# do all the encoding there and alarm just checks diff

class KullbackLeiblerAlarm(Alarm):
    pass


class ExampleNLPAlarm(Alarm):
    pass

