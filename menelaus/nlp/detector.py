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