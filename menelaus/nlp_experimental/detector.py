class Detector():
    def __init__(self, representation, alarm, probes : list = None):
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
        pass

    def recalibrate(self, rep_test):
        if self.state == "drift":
            if len(rep_test) == 1 and window_not_full:
                # TODO store until window size exceeded, then merge
                # TODO representation should be in charge of knowing
                #      how to add one / many representations to itself
                self._rep_pool.append(rep_test)
            elif len(rep_test) == 1 and window_full:
                self.rep_reference = self.rep_reference + self._rep_pool
            else:
                self.rep_reference = self.rep_reference + rep_test

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
                self.recalibrate(self.rep_test)
