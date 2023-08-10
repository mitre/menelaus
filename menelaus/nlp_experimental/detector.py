class Detector():
    def __init__(self, representation, alarm, probes):
        self.representation = representation
        self.alarm = alarm
        self.probes = probes # summary statistics, live plots, etc.

    def update(self, data):
        pass

    def run(self):
        pass

    @property
    def state(self):
        return self.alarm._state