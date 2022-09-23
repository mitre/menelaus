from abc import ABC, abstractmethod
from collections import defaultdict

from menelaus.drift_detector import BatchDetector, StreamingDetector


#######################
# Ensemble Evaluators
#######################


class Election(ABC):
    """
    Abstract base class for implementations of election schemes
    used to evaluate drift state of ensembles, by operating on
    the drift states of constituent detectors.

    Constructors for sub-classes may differ, but all ``Election``
    classes are callable classes, where the call takes only the
    list of detectors to evaluate.

    The surrounding ``Ensemble`` class will update its drift state
    within its ``update`` function, by calling the ``Election``
    it is given at initialization-time, upon its detectors.
    """

    @abstractmethod
    def __call__(self, detectors: list):  # pragma: no cover
        raise NotImplemented


class SimpleMajorityElection(Election):
    """
    ``Election`` that determines drift for an ensemble,
    based on whether a simple majority of the ensemble's
    detectors have voted for drift.

    Ref :cite:t:`duelectionreference`
    """

    def __call__(self, detectors: list):
        """
        Args:
            detectors (list): detector objects to examine

        Returns
            str: ``"drift"`` if drift is determined, or ``None``
        """
        simple_majority_threshold = len(detectors) // 2
        alarms = [d for d in detectors if d.drift_state == "drift"]
        num_drift = len(alarms)
        if num_drift > simple_majority_threshold:
            return "drift"
        else:
            return None


class MinimumApprovalElection(Election):
    """
    ``Election`` that determines drift based on whether
    a minimum number of provided detectors have alarmed. This
    threshold can be 1 to the maximum number of detectors.
    """

    def __init__(self, approvals_needed: int = 1):
        """
        Args:
            approvals_needed (int): minimum approvals to alarm
        """
        self.approvals_needed = approvals_needed

    def __call__(self, detectors: list):
        """
        Args:
            detectors (list): detector objects to examine

        Returns:
            str: ``"drift_state"`` if drift is determined, or ``None``
        """
        num_approvals = 0
        for d in detectors:
            if d.drift_state == "drift":
                num_approvals += 1
            if num_approvals >= self.approvals_needed:
                return "drift"
        return None


class OrderedApprovalElection(Election):
    """
    ``Election`` that determines drift based on whether:
        1) An initial ``a`` count of detectors alarmed for drift.
        2) A subsequent ``c`` count of detectors confirmed drift.

    Hypothethically, the distinction between this and
    ``MinimumApprovalElection(a+c)``, is if the detectors were added to
    a collection in a meaningful order. As such this voting
    scheme iterates over detectors in preserved order of insertion into
    the user-defined list, and uses the first ``approvals_needed``
    amount for initial detection, and the next ``confirmations_needed``
    amount for confirmation of drift.
    """

    def __init__(self, approvals_needed: int = 1, confirmations_needed: int = 1):
        """
        Args:
            approvals_needed (int): Minimum number of detectors that
                must alarm for the ensemble to alarm.
            confirmations_needed (int): Minimum number of confirmations
                needed to alarm, after ``approvals_needed`` alarms have been
                observed.
        """
        self.approvals_needed = approvals_needed
        self.confirmations_needed = confirmations_needed

    def __call__(self, detectors: list):
        """
        Args:
            detectors (list): detector objects to examine

        Returns:
            str: ``"drift_state"`` if drift is determined, or ``None``
        """
        num_approvals = 0
        num_confirmations = 0

        for d in detectors:
            if d.drift_state == "drift":
                if num_approvals < self.approvals_needed:
                    num_approvals += 1
                else:
                    num_confirmations += 1

                if (
                    num_approvals >= self.approvals_needed
                    and num_confirmations >= self.confirmations_needed
                ):
                    return "drift"

        return None


class ConfirmedElection(Election):
    """
    ``Election`` for handling detectors (typically in
    streaming setting) with waiting logic. In this scheme,
    a detector that has alarmed will increment 'waiting' counters
    until enough other detectors confirm the drift or warning
    alarm.

    Derived from the Maciel ensemble evaluation scheme.

    Ref. :cite:t:`macielelectionreference`
    """

    def __init__(self, sensitivity: int, wait_time: int):
        """
        Args:
            sensitivity (int): how many combined waiting/new drift alarms
                should result in ensemble alarm
            wait_time (int): after how many steps of waiting, should each
                detector reset its time spent waiting post-drift-alarm
        """
        self.sensitivity = sensitivity
        self.wait_time = wait_time
        self.wait_period_counters = None

    def __call__(self, detectors: list):
        """
        Args:
            detectors (list): detector objects to examine

        Returns:
            str: ``"drift_state"`` if drift is determined, or ``None``
        """
        if self.wait_period_counters is None:
            self.wait_period_counters = [0] * len(detectors)

        num_drift = 0
        num_warning = 0

        states = [d.drift_state for d in detectors]
        for i, state in enumerate(states):
            if state == "drift" and self.wait_period_counters[i] == 0:
                num_drift += 1
                self.wait_period_counters[i] += 1
            elif state == "warning":
                num_warning += 1
            elif self.wait_period_counters[i] != 0:
                num_drift += 1
                self.wait_period_counters[i] += 1

        if num_drift >= self.sensitivity:
            ret = "drift"
        elif num_warning + num_drift >= self.sensitivity:
            ret = "warning"
        else:
            ret = None

        for i, count in enumerate(self.wait_period_counters):
            if count > self.wait_time:
                self.wait_period_counters[i] = 0

        return ret


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


class StreamingEnsemble(StreamingDetector, Ensemble):
    """
    Implements Ensemble class for streaming drift detectors. Inherits
    from ``Ensemble`` and ``StreamingDetector`` (i.e., ``StreamingEnsemble``
    IS-A ``StreamingDetector``). As such it has the functions of a regular
    detector: ``update``, ``reset``, etc. Internally, these operate not only 
    on the ensemble's own attributes, but on the set of detectors given to it.
    """

    def __init__(self, detectors: dict, election, column_selectors: dict = {}):
        """
        Args:
            detectors (dict): Set of detectors in ensemble. Should be keyed by
                unique strings for convenient lookup, and valued by initialized
                detector objects.
            election (str): Initialized ``Election`` object for ensemble to evaluate
                drift among constituent detectors. See implemented election schemes
                in ``menelaus.ensemble``.
            columns_selectors (dict, optional): Table of functions to use
                for each detector. Should be keyed to match the format of
                ``detectors``. Each function should take a data instance X,
                and return the columns of X that the corresponding detector
                should operate on.
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
    ``update``, and ``reset``. Only internally, these operate not only on the
    ensemble's own attributes, but on the set of detectors given to it.
    """

    def __init__(self, detectors: dict, election, column_selectors: dict = {}):
        """
        Args:
            detectors (dict): Set of detectors in ensemble. Should be keyed by
                unique strings for convenient lookup, and valued by initialized
                detector objects.
            election (str): Initialized ``Election`` object for ensemble to evaluate
                drift among constituent detectors. See implemented election schemes
                in ``menelaus.ensemble``.
            columns_selectors (dict, optional): Table of functions to use
                for each detector. Should be keyed to match the format of
                ``detectors``. Each function should take a data instance X,
                and return the columns of X that the corresponding detector
                should operate on.
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
