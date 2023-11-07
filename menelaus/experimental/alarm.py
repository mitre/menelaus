import numpy as np
from abc import ABC, abstractmethod
from scipy.stats import ks_2samp


STATE_INIT = "baseline"
STATE_DRIFT = "alarm"
STATE_WARNING = "warning"


class Alarm(ABC):
    """
    Abstract base class for alarm objects. An alarm scheme must have an
    ``evaluate`` method to compare reference and test data representations.
    Its state can take one of three values: ``"baseline"``, ``"alarm"``, or
    ``"warning"``.
    """

    @abstractmethod
    def evaluate(self, rep_reference, rep_test):
        """
        The ``evaluate()`` function is where alarms make the comparison between
        reference and test data representations.

        Args:
            rep_reference (numpy.ndarray or tensorflow.tensor) : reference data
            rep_test (numpy.ndarray or tensorflow.tensor) : test data
        """
        raise NotImplementedError(f"Called evaluate() from ABC")


class KolmogorovSmirnovAlarm(Alarm):
    """
    Alarm scheme based on the Kolmogorov-Smirnov test. Takes an alpha level to
    compare p-values to, and a proportion of 'drifting' features at which the
    alarm is raised. Performs a two-sided KS test, feature-by-feature.

    Attributes:
        alpha (float) : critical level to compare against p-values and
            determine significant drift. Default ``0.05``.
        critical_feature_proportion (float) : proportion of drifting features
            (*i.e.*, for which the null hypothesis is rejected) needed to raise
            the alarm for overall drift. Default ``0.25``.

    Ref. :cite:t:`alibi-detect`
    """

    def __init__(self, alpha=0.05, critical_feature_proportion=0.25):
        """
        Args:
            alpha (float) : critical level to compare against p-values and
                determine significant drift. Default ``0.05``.
            critical_feature_proportion (float) : proportion of drifting features
                (*i.e.*, for which the null hypothesis is rejected) needed to raise
                the alarm for overall drift. Default ``0.25``.
        """
        self._state = STATE_INIT
        self.alpha = alpha
        self.critical_feature_proportion = critical_feature_proportion

    def evaluate(self, rep_reference, rep_test):
        """
        Takes two data representations, and computes a two-sided KS test for each
        feature. If only some features drift, the alarm is in a warning state. If
        the ``critical_feature_proportion`` is reached, the alarm is in a drift
        state, otherwise it is in the baseline state.

        *Note:*
            - a Bonferroni correction is applied
            - the test and reference datasets should have the same number of
                features, or the behavior of the KS test will be unpredictable.

        Args:
            rep_reference (numpy.ndarray or tensorflow.tensor) : reference data
            rep_test (numpy.ndarray or tensorflow.tensor) : test data
        """
        rep_reference = rep_reference.reshape(rep_reference.shape[0], -1)
        rep_test = rep_test.reshape(rep_test.shape[0], -1)
        n_features = rep_reference.shape[1]
        p_values = np.zeros(n_features, dtype=np.float32)
        for i in range(n_features):
            _, p_values[i] = ks_2samp(
                rep_reference[:, i],
                rep_test[:, i],
                alternative="two-sided",
                mode="asymp",
            )
            p_values[i] /= n_features
        extreme_features_proportion = (
            len([p for p in p_values if p <= self.alpha]) / n_features
        )
        if 0 < extreme_features_proportion < self.critical_feature_proportion:
            self._state = STATE_WARNING
        elif extreme_features_proportion >= self.critical_feature_proportion:
            self._state = STATE_DRIFT
        else:
            self._state = STATE_INIT
