from menelaus.drift_detector import BatchDetector


def eval_simple_majority(detectors):
    """
    Evaluator function that determines drift for an ensemble,
    based on whether a simple majority of the ensemble's
    detectors have voted for drift.

    TODO - determine if this correctly calculates simple majority

    Args:
        detectors (list): list of detector objects to examine

    Returns
        str: 'drift' if drift is determined, or ``None``
    """
    simple_majority_threshold = len(detectors) // 2
    num_drift = len([det for det in detectors if det.drift_state == "drift"])
    if num_drift > simple_majority_threshold:
        return "drift"
    else:
        return None


# Keeping all evaluator functions in a table enables
# ensembles to look up via str value, rather than
# needing a function passed in.
evaluators = {"simple-majority": eval_simple_majority}


class Ensemble:
    """
    Parent class for Ensemble detectors. Does not inherit from any
    detector parent class, but has similar functions to set reference,
    update, reset, filter data columns over each internal detector. Can
    also evaluate the results from all detectors per some voting scheme.

    Any class hoping to implement ensemble functionality should implement
    from this.
    """

    def __init__(self, detectors: dict, evaluator: str, columns: dict = None):
        self.detectors = detectors.copy()
        self.evaluator = evaluators[evaluator]
        self.columns = columns

    def set_reference(self, X, y_true, y_pred):
        """
        Initialize each detector in ensemble with a reference batch.

        Args:
            X (pandas.DataFrame or numpy.array): baseline dataset
            y_true (numpy.array): actual labels of dataset
            y_pred (numpy.array): predicted labels of dataset
        """
        for det_key in self.detectors:
            # XXX - Cannot re-define X = constrain(), else external reference is modified
            #       Need to see why this is happening and where to put e.g. a copy() stmt.
            X_selected = self.select_data(X, det_key)
            self.detectors[det_key].set_reference(
                X=X_selected, y_true=y_true, y_pred=y_pred
            )

    def update(self, X, y_true=None, y_pred=None):
        """
        Update each detector in ensemble with new batch of data.
        Calls self.evaluate() at the end, to determine voting result.

        Args:
            X (numpy.ndarray): input data
            y_true (numpy.ndarray): if applicable, true labels of input data
            y_pred (numpy.ndarray): if applicable, predicted labels of input data
        """
        for det_key in self.detectors:
            # XXX - Cannot re-define X = constrain(), else external reference is modified
            #       Need to see why this is happening and where to put e.g. a copy() stmt.
            X_selected = self.select_data(X, det_key)
            self.detectors[det_key].update(X=X_selected, y_true=y_true, y_pred=y_pred)
        self.evaluate()

    def select_data(self, data, det_key: str):
        """
        Filters given data according to the columns expected
        by each detector in the ensemble, as specified at
        initialization (uses ``self.columns``, if it exists).

        Intended for use within ``update`` or ``set_reference``.

        Args:
            data (pandas.DataFrame): data to be used
            det_key (str): identifies detector in ensemble by which
                to filter data columns
        """
        # TODO - can y_true, y_pred be supported in this pattern?
        # TODO - this allows for list manipulation of PD columns
        #           will need to think about cases where numpy arrays
        #           are mixed in
        ret = data.copy()
        if self.columns:
            cols = self.columns[det_key]
            ret = data[cols]
        return ret

    def evaluate(self):
        """
        Uses evaluator function specified to ensemble, to determine
        voting result of all detectors. Sets ensemble's own drift
        state accordingly.
        """
        self.drift_state = self.evaluator(self.detectors.values())

    def reset(self):
        """
        Initialize each detector's drift state and other relevant attributes.
        Intended for use after ``drift_state == 'drift'``.
        """
        for det_key in self.detectors:
            self.detectors[det_key].reset()


class BatchEnsemble(BatchDetector, Ensemble):
    """
    Implements Ensemble class for batch-based drift detectors. Inherits
    from Ensemble and BatchDetector (i.e., BatchEnsemble IS-A BatchDetector).
    As such it has the functions of a regular detector, to set reference,
    update, and reset. Only internally, these operate not only on the
    ensemble's own attributes, but on the set of detectors given to it.
    """

    def __init__(self, detectors: dict, evaluator: str, columns: dict = None):
        """
        Args:
            detectors (dict): Set of detectors in ensemble. Should be keyed by
                unique strings for convenient lookup, and valued by initialized
                detector objects.
            evaluator (str): String identifier for voting scheme by which to
                determine if drift is present. E.g., 'simple-majority' uses
                a function to determine if a simple majority of detectors
                found drift. See options in ``menelaus.ensemble.evaluators``.
            columns (dict, optional): Optional table of column lists to use
                for each detector. Should be keyed to match the format of
                ``detectors``. Will be used to filter the data columns passed
                to ensemble in ``update`` or ``set_reference`` according to
                each detector.
        """
        BatchDetector.__init__(self)
        Ensemble.__init__(self, detectors, evaluator, columns)

    def update(self, X, y_true=None, y_pred=None):
        """
        Update ensemble itself, and each constituent detector with new data.
        Calls ``Ensemble.update`` and ``BatchDetector.update`` to do so.

        Args:
            X (numpy.ndarray): input data
            y_true (numpy.ndarray): if applicable, true labels of input data
            y_pred (numpy.ndarray): if applicable, predicted labels of input data
        """
        Ensemble.update(self, X=X, y_true=y_true, y_pred=y_pred)
        BatchDetector.update(self, X=X, y_true=y_true, y_pred=y_pred)

    def reset(self):
        """
        Reset ensemble itself, and each constituent detector's drift state
        and other relevant attributes. Intended for use after ``drift_state == 'drift'``.
        Calls ``Ensemble.reset`` and ``BatchDetector.reset`` to do so.
        """
        Ensemble.reset(self)
        BatchDetector.reset(self)

    def set_reference(self, X, y_true=None, y_pred=None):
        """
        Initialize ensemble itself, and each constituent detector
        with a reference batch. Calls ``Ensemble.set_reference`` to do so.

        Args:
            X (pandas.DataFrame or numpy.array): baseline dataset
            y_true (numpy.array): actual labels of dataset
            y_pred (numpy.array): predicted labels of dataset
        """
        Ensemble.set_reference(self, X=X, y_true=y_true, y_pred=y_pred)