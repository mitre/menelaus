from molten.DriftDetector import DriftDetector
import numpy as np
import scipy.stats

# TODO: replace self._window with a numpy array - test efficiency
# TODO: replace self.retraining_recs with a numpy array - test efficiency


class STEPD(DriftDetector):
    """
    STEPD is a drift detection algorithm based on a binary classifier's accuracy, intended for an online classifier.

    Two windows are defined -- "recent" and "past", with corresponding accuracies p_r and p_p. Roughly, the distribution
    of their absolute difference, normalized by the accuracy of the two windows combined, T, is normally distributed.
    So, this test statistic's p-value P(T) defines the warning and drift regions:
        If p_r < p_p (the classifier's accuracy on recent samples is decreased):
            and P(T) < alpha_warning, the detector's state is set to "warning".
            and P(T) < alpha_drift, the detector's state is set to "drift".

    The index of the first sample which triggered a warning/drift state (relative to self.n) is stored in
    self.retraining_recs, for retraining the classifier when drift occurs.

    STEPD is intended for use with an online classifier, which is trained on every new sample. That is, with each new
    sample, the question is not whether the classifier will be retrained; it's whether some part of the previous
    training data should be excluded during retraining. The implementation depends on whether the classifier involved
    is able to incrementally retrain using only a single data point vs. being required to retrain on the entire set.

    Ref. K. Nishida and K. Yamauchi, "Detecting concept drift using statistical testing," in Proc. 10th Int. Conf.
    Discovery Science, V. Corruble, M. Takeda, and E. Suzuki, Eds. Berlin, Heidelberg: Springer Berlin Heidelberg, 2007,
    Conference Proceedings, pp 264-269.

    Attributes:
    :attribute retraining_recs: recommended indices for retraining. Usually [first warning index, drift index].
        If no warning state occurs, this will instead be [drift index, drift index] -- this indicates an abrupt
        change.
        Resets when self.drift_state returns to None (no drift nor warning).

    """

    def __init__(self, window_size=30, alpha_warning=0.05, alpha_drift=0.003):
        """
        :param window_size: the size of the "recent" window
        :param alpha_warning: defines the threshold over which to enter the warning state.
        :param alpha_drift: defines the threshold over which to enter the drift state.
        """
        # TODO: probably wants the property decorator on a bunch of these
        super().__init__()
        self.window_size = window_size
        self.alpha_warning = alpha_warning
        self.alpha_drift = alpha_drift
        self._s, self._r = 0, 0
        self._window = []
        self._test_statistic = None
        self._test_p = None
        self._initialize_retraining_recs()

    def reset(self):
        """
        Initialize the detector's drift state and other relevant attributes. Intended for use after
        drift_state == 'drift'.
        """
        super().reset()
        self._s, self._r = 0, 0
        self._window = []
        self._test_statistic = None
        self._test_p = None
        self._initialize_retraining_recs()

    def update(self, y_pred, y_true):
        """
        Update the detector with a new sample.
        :param y_pred: predicted class
        :param y_true: actual class
        """
        if self.drift_state == "drift":
            self.reset()
            self._initialize_retraining_recs()

        super().update()
        classifier_result = int(y_pred == y_true)
        self._s += classifier_result

        # update r and s and trim the window
        self._window.append(classifier_result)
        if len(self._window) > self.window_size:
            # points that are larger than the buffer are removed, and only tracked through r and n
            self._s -= self._window[0]
            self._r += self._window[0]
            self._window = self._window[1:]

        if self.n >= 2 * self.window_size:
            recent_accuracy = self.recent_accuracy()
            past_accuracy = self.past_accuracy()
            overall_accuracy = self.overall_accuracy()
            self._test_statistic = (
                np.absolute(past_accuracy - recent_accuracy)
                - 0.5 * ((1 / (self.n - self.window_size)) + (1 / self.window_size))
            ) / np.sqrt(
                overall_accuracy
                * (1 - overall_accuracy)
                * ((1 / (self.n - self.window_size)) + (1 / self.window_size))
            )
            self._test_p = 1 - scipy.stats.norm.cdf(
                self._test_statistic, 0, 1
            )  # one-sided test

            accuracy_decreased = past_accuracy > recent_accuracy
            if accuracy_decreased and self._test_p < self.alpha_drift:
                self.drift_state = "drift"
            elif accuracy_decreased and self._test_p < self.alpha_warning:
                self.drift_state = "warning"
            else:
                self.drift_state = None
                self._initialize_retraining_recs()

            if self.drift_state is not None:
                self._increment_retraining_recs()

    def recent_accuracy(self):
        """
        :return: the accuracy of the classifier among the last self.window_size samples the detector has seen
        """
        if self.window_size == 0:
            out = 0
        else:
            out = self._s / self.window_size
        return out

    def past_accuracy(self):
        """
        :return: the accuracy of the classifier among the samples the detector has seen before its current window, but
            after the last time the detector was reset
        """
        if (self.n - self.window_size) == 0:
            out = 0
        else:
            out = self._r / (self.n - self.window_size)
        return out

    def overall_accuracy(self):
        """
        :return: the accuracy of the classifier among the samples the detector has seen since the detector was last
            reset
        """
        if self.n == 0:
            out = 0
        else:
            out = (self._r + self._s) / (self.n)
        return out

    def _initialize_retraining_recs(self):
        self.retraining_recs = [None, None]

    def _increment_retraining_recs(self):
        if self.retraining_recs[0] is None:
            self.retraining_recs[0], self.retraining_recs[1] = (
                self.total_samples,
                self.total_samples,
            )
        else:
            self.retraining_recs[1] += 1
