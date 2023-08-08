""" ALL implementations of a `Representation` should go here """
from toolz import pipe


class Representation():
    def __init__(self, preprocessors):
        self.preprocessors = preprocessors

    def fit(self, data):
        data = pipe(data, *self.preprocessors)
        return data


class KDQTreeRepresentation(Representation):
    """
    """
    def fit(self, data):
        data = super().fit(data)
        data = data + 1
        return data


class ExampleNLPRepresentation(Representation):
    """
    """
    def fit(self, data):
        data = super().fit(data)
        data = data + 1
        return data