from menelaus.drift_detector import BatchDetector


def eval_simple_majority(detectors):
    # TODO - is this what simple majority means
    simple_majority_threshold = len(detectors) // 2
    num_drift = len([det for det in detectors if det.drift_state == 'drift'])
    if num_drift > simple_majority_threshold:
        return 'drift'
    else:
        return None


evaluators = {
    'simple-majority': eval_simple_majority
}


class Ensemble():
    def __init__(self, detectors: dict, evaluator: str, constraints: dict = None):
        self.detectors = detectors.copy()
        self.evaluator = evaluators[evaluator]
        self.constraints = constraints

    def set_reference(self, X, y_true, y_pred):
        for det_key in self.detectors:
            # XXX - Cannot re-define X = constrain(), else external reference is modified
            #       Need to see why this is happening and where to put e.g. a copy() stmt.
            X_constrained = self.constrain(X, det_key)
            self.detectors[det_key].set_reference(X=X_constrained, y_true=y_true, y_pred=y_pred)

    def update(self, X, y_true=None, y_pred=None):
        for det_key in self.detectors:
            # XXX - Cannot re-define X = constrain(), else external reference is modified
            #       Need to see why this is happening and where to put e.g. a copy() stmt.
            X_constrained = self.constrain(X, det_key)
            self.detectors[det_key].update(X=X_constrained, y_true=y_true, y_pred=y_pred)
        self.evaluate()

    def constrain(self, data, det_key: str):
        # TODO - can y_true, y_pred be supported in this pattern?
        # TODO - this allows for list manipulation of PD columns
        #           will need to think about cases where numpy arrays
        #           are mixed in
        ret = data.copy()
        if self.constraints:
            constraint = self.constraints[det_key]
            ret = data[constraint]
        return ret

    def evaluate(self):
        self.drift_state = self.evaluator(self.detectors.values())

    def reset(self):
        for det_key in self.detectors:
            self.detectors[det_key].reset()


class BatchEnsemble(BatchDetector, Ensemble):
    def __init__(self, detectors: dict, evaluator: str, constraints: dict = None):
        BatchDetector.__init__(self)
        Ensemble.__init__(self, detectors, evaluator, constraints)

    def update(self, X, y_true=None, y_pred=None):
        Ensemble.update(self, X=X, y_true=y_true, y_pred=y_pred)
        BatchDetector.update(self, X=X, y_true=y_true, y_pred=y_pred)

    def reset(self):
        Ensemble.reset(self)
        BatchDetector.reset(self)

    def set_reference(self, X, y_true=None, y_pred=None):
        Ensemble.set_reference(self, X=X, y_true=y_true, y_pred=y_pred)