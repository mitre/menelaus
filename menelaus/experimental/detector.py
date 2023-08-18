from toolz import pipe

class Detector():
    def __init__(self, representation, alarm, operators: list = None):
        self.representation = representation
        self.alarm = alarm
        # summary statistics, live plots, etc.
        self.probes = probes if probes is not None else []
        self.rep_reference = None
        self._rep_pool = []
        self.rep_test = None

    def update(self, data):
        raise NotImplementedError

    def run(self):
        # ...
        self = pipe(self, *self.probes)

    @property
    def state(self):
        return self.alarm._state
    

class BasicDetector(Detector):
    """
    """
    def update(self, data):
        if self.rep_reference is None:
            self.rep_reference = self.representation.transform(data)
        else:
            initial_rep_test = self.rep_test is None
            self.rep_test = self.representation.transform(data)
            self.alarm.evaluate(self.rep_reference, self.rep_test)

            if not initial_rep_test:
                self.representation.recalibrate(data)
