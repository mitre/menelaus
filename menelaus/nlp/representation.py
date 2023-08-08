from toolz import pipe

class Representation():
    """
    A data representation ...
    """
    def __init__(self, preprocessors : list = None):
        # avoid default [] for arguments
        if preprocessors is not None:
            self.preprocessors = preprocessors
        else: 
            self.preprocessors = []

    def cmp(self, rep):
        return 0

    def fit(self, data):
        data = pipe(data, *self.preprocessors)
        return data

class ExampleNLPRepresentation(Representation):
    """
    """
    def fit(self, data):
        data = super().fit(data)
        # XXX   here an NLP representation may encode/transform, while a 
        #       kdq-tree may build/fill
        data = data + 1 
        return data