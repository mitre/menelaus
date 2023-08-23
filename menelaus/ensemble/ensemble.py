from collections import defaultdict

from menelaus.detector import BatchDetector, StreamingDetector


#############
# Ensembles
#############


class Ensemble:
    """
    Parent class for Ensemble detectors. Does not inherit from any
    detector parent class, but has similar functions ``set_reference``,
    ``update``, ``reset``. Can also evaluate the results from all
    detectors per some voting scheme.

    Any class hoping to implement ensemble functionality should implement
    from this.
    """

    def __init__(self, detectors: dict, election, column_selectors: dict = {}):
        # XXX - Since rigid type-checking is sort of discouraged in Python
        #       it makes the most sense to just treat election as (always)
        #       a function operating on detectors.
        self.detectors = detectors.copy()
        self.election = election

        def default_column_selector():
            return lambda data: data

        self.column_selectors = defaultdict(default_column_selector)
        self.column_selectors.update(column_selectors)

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
            X_selected = self.column_selectors[det_key](X)
            self.detectors[det_key].update(X=X_selected, y_true=y_true, y_pred=y_pred)

        det_list = list(self.detectors.values())
        self.drift_state = self.election(det_list)

    def reset(self):
        """
        Initialize each detector's drift state and other relevant attributes.
        Intended for use after ``drift_state == 'drift'``.
        """
        for det_key in self.detectors:
            self.detectors[det_key].reset()

    @property
    def drift_states(self):
        """
        Returns dict where keys are detector identifiers as specified in ``self.detectors``,
        values are drift states for corresponding detector objects.
        """
        return {
            detector_id: detector.drift_state
            for detector_id, detector in self.detectors.items()
        }

    @property
    def retraining_recs(self):
        """
        Returns dict where keys are detector identifiers as specified in ``self.detectors``,
        values are retraining recommendation values from corresponding detector objects.
        Detectors which do not have retraining recommendations are skipped in the output.
        """
        ret = {}
        for detector_id, detector in self.detectors.items():
            # XXX - some explicit handling may be better practice than hasattr
            if hasattr(detector, "retraining_recs"):
                ret[detector_id] = detector.retraining_recs
        return ret


class StreamingEnsemble(StreamingDetector, Ensemble):
    """
    Implements Ensemble class for streaming drift detectors. Inherits
    from ``Ensemble`` and ``StreamingDetector`` (i.e., ``StreamingEnsemble``
    IS-A ``StreamingDetector``). As such it has the functions of a regular
    detector: ``update``, ``reset``, etc. Internally, these operate not only
    on the ensemble's own attributes, but on the set of detectors given to it.

    Attributes:
        detectors (dict): ``dict`` of detector objects, keyed by unique string
            identifiers
        election (menelaus.ensemble.Election): election object to dictate how
            overall ensemble drift state is determined
        column_selectors (dict): ``dict`` of functions, whose keys match the
            keys of ``detectors``. A column selector function for a given
            key determines how data is selected to be fed into that detector.
        drift_state (str or None): Overall ensemble drift state. Can be ``"drift"``,
            ``"warning"``, or ``None``.
        drift_states (dict): ``dict`` of drift states whose keys match the keys
            of ``detectors``, and whose values are the drift states for those
            detector objects
        retraining_recs (dict): ``dict`` of retraining recommendations whose keys
            match the keys of ``detectors``, and whose values are the retraining
            recommendations for those detector objects. Detectors who do not
            have retraining options are skipped.
    """

    def __init__(self, detectors: dict, election, column_selectors: dict = {}):
        """
        Args:
            detectors (dict): Dictionary of detectors in ensemble, where the key
                is some unique identifier for a detector, and the value is the
                initialized detector object. For instance, ``{'a': ADWIN()}``.
            election (str): Initialized ``Election`` object for ensemble to evaluate
                drift among constituent detectors. See implemented election schemes
                in ``menelaus.ensemble``.
            columns_selectors (dict, optional): Functions to use
                for each detector. Functions should take data instance X
                and return the columns of X that the corresponding detector
                should operate on. Should match format of ``detectors`` i.e.
                ``{'a': ADWIN()}`` would need an entry ``{'a': function}`` to
                use this feature. By default, no column selection
                function is applied to any detector, and they will all use
                the entirely of the attributes in X.
        """
        StreamingDetector.__init__(self)
        Ensemble.__init__(self, detectors, election, column_selectors)

    def update(self, X, y_true, y_pred):
        """
        Update ensemble itself, and each constituent detector with new data.
        Calls ``Ensemble.update`` and ``StreamingDetector.update`` to do so.

        Args:
            X (numpy.ndarray): input data
            y_true (numpy.ndarray): if applicable, true labels of input data
            y_pred (numpy.ndarray): if applicable, predicted labels of input data
        """
        Ensemble.update(self, X=X, y_true=y_true, y_pred=y_pred)
        StreamingDetector.update(self, X=X, y_true=y_true, y_pred=y_pred)

    def reset(self):
        """
        Reset ensemble itself, and each constituent detector's drift state
        and other relevant attributes. Intended for use after ``drift_state == 'drift'``.
        Calls ``Ensemble.reset`` and ``StreamingDetector.reset`` to do so.
        """
        Ensemble.reset(self)
        StreamingDetector.reset(self)


class BatchEnsemble(BatchDetector, Ensemble):
    """
    Implements ``Ensemble`` class for batch-based drift detectors. Inherits
    from ``Ensemble`` and ``BatchDetector`` (i.e., ``BatchEnsemble`` IS-A ``BatchDetector``).
    As such it has the functions of a regular detector, ``set_reference``,
    ``update``, and ``reset``. These functions will operate not only on the
    ensemble's own attributes, but on the set of detectors given to it.

    Attributes:
        detectors (dict): ``dict`` of detector objects, keyed by unique string
            identifiers
        election (menelaus.ensemble.Election): election object to dictate how
            overall ensemble drift state is determined
        column_selectors (dict): ``dict`` of functions, whose keys match the
            keys of ``detectors``. A column selector function for a given
            key determines how data is selected to be fed into that detector.
        drift_state (str or None): Overall ensemble drift state. Can be ``"drift"``,
            ``"warning"``, or ``None``.
        drift_states (dict): ``dict`` of drift states whose keys match the keys
            of ``detectors``, and whose values are the drift states for those
            detector objects
        retraining_recs (dict): ``dict`` of retraining recommendations whose keys
            match the keys of ``detectors``, and whose values are the retraining
            recommendations for those detector objects. Detectors who do not
            have retraining options are skipped.
    """

    def __init__(self, detectors: dict, election, column_selectors: dict = {}):
        """
        Args:
            detectors (dict): Dictionary of detectors in ensemble, where the key
                is some unique identifier for a detector, and the value is the
                initialized detector object. For instance, ``{'p': PCA_CD()}``.
            election (str): Initialized ``Election`` object for ensemble to evaluate
                drift among constituent detectors. See implemented election schemes
                in ``menelaus.ensemble``.
            column_selectors (dict, optional): Table of functions to use
                for each detector. Functions should take data instance X
                and return the columns of X that the corresponding detector
                should operate on. Should match format of ``detectors`` i.e.
                ``{'p': PCA_CD()}`` would need an entry ``{'a': function}`` to
                use this feature. By default, no column selection
                function is applied to any detector, and they will all use
                the entirely of the attributes in X.
        """
        BatchDetector.__init__(self)
        Ensemble.__init__(self, detectors, election, column_selectors)

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
        for det_key in self.detectors:
            # XXX - Cannot re-define X = constrain(), else external reference is modified
            #       Need to see why this is happening and where to put e.g. a copy() stmt.
            X_selected = self.column_selectors[det_key](X)
            self.detectors[det_key].set_reference(
                X=X_selected, y_true=y_true, y_pred=y_pred
            )
