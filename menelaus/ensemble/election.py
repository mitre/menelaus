from abc import ABC, abstractmethod


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
    ``Election`` for handling detectors, typically in
    streaming setting. In this scheme, when a single
    detector alarms, the ``Election`` will wait for a
    certain number of samples, until one or more other
    detectors also alarm, confirming the drift.

    Derived from the ensemble scheme described in
    :cite:t:`macielelectionreference`.
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
