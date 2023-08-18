from toolz import pipe

class Representation():
    """
    A data representation ...
    """
    def __init__(self, preprocessors : list = None, postprocessors : list = None):
        # avoid default [] for arguments
        self.preprocessors = preprocessors if preprocessors is not None else []
        self.postprocessors = postprocessors if postprocessors is not None else []
        self.reference = None
        self.pool = None

    def transform(self, data):
        self.reference = data
        data = pipe(data, *self.preprocessors)
        return data

    def recalibrate(self, data):
        window_not_full = True
        window_full = True



        if self.state == "drift":
            if len(data) == 1 and window_not_full:
                # TODO store until window size exceeded, then merge
                # TODO representation should be in charge of knowing
                #      how to add one / many representations to itself
                self.pool += data
            elif len(data) == 1 and window_full:
                self.rep_reference += self.pool
            else:
                self.rep_reference += data

class ExampleNLPRepresentation(Representation):
    """
    """
    def transform(self, data):
        data = super().transform(data)
        
        # may replace hellinger distance histograms with something else
        # list of preprocessors should be "transforms" written as functions
        # maybe user doesn't need to know about representation class at all

        # XXX   here an NLP representation may encode/transform, while a 
        #       kdq-tree may build/fill
        data += 1
        
        data = pipe(data, *self.postprocessors)
        return data