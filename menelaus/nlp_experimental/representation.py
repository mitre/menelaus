from toolz import pipe

class Representation():
    """
    A data representation ...
    """
    def __init__(self, preprocessors : list = None, postprocessors : list = None):
        # avoid default [] for arguments
        self.preprocessors = preprocessors if preprocessors is not None else []
        self.postprocessors = postprocessors if postprocessors is not None else []

    def transform(self, data):
        data = pipe(data, *self.preprocessors)
        return data

class ExampleNLPRepresentation(Representation):
    """
    """
    def transform(self, data):
        data = super().transform(data)
        
        # XXX   here an NLP representation may encode/transform, while a 
        #       kdq-tree may build/fill
        data = data + 1 
        
        data = pipe(data, *self.postprocessors)
        return data