import numpy as np
import scipy.stats
from menelaus.detector import StreamingDetector


class STEPD(StreamingDetector):
    """STEPD is a drift detection algorithm based on a binary classifier's
    accuracy, intended for an online classifier.

    Two windows are defined -- "recent" and "past", with corresponding
    accuracies ``p_r`` and ```p_p```. Roughly, the distribution of their absolute
    difference, normalized by the accuracy of the two windows combined, T, is
    normally distributed. So, this test statistic's p-value P(T) defines the
    warning and drift regions.

    If ``p_r`` < ``p_p`` (the classifier's accuracy on recent samples is decreased):

    * and P(T) < ``alpha_warning``, the detector's state is set to ``"warning"``.

    * and P(T) < ``alpha_drift``, the detector's state is set to ``"drift"``.

    The index of the first sample which triggered a warning/drift state
    (relative to ``self.samples_since_reset``) is stored in ``self._retraining_recs``,
    for retraining the classifier when drift occurs.

    STEPD is intended for use with an online classifier, which is trained on
    every new sample. That is, with each new sample, the question is not whether
    the classifier will be retrained; it's whether some part of the previous
    training data should be excluded during retraining. The implementation
    depends on whether the classifier involved is able to incrementally retrain
    using only a single data point vs. being required to retrain on the entire
    set.

    Ref. :cite:t:`nishida2007detecting`
    """

    input_type = "stream"

    def __init__(self, window_size=30, alpha_warning=0.05, alpha_drift=0.003):
        """
        Args:
            window_size (int, optional): the size of the "recent" window.
                Defaults to 30.
            alpha_warning (float, optional): defines the threshold over which to
                enter the warning state. Defaults to 0.05.
            alpha_drift (float, optional): defines the threshold over which to
                enter the drift state. Defaults to 0.003.
        """
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
        """Initialize the detector's drift state and other relevant attributes.
        Intended for use after ``drift_state == 'drift'``.
        """
        super().reset()
        self._s, self._r = 0, 0
        self._window = []
        self._test_statistic = None
        self._test_p = None
        self._initialize_retraining_recs()

    def update(self, y_true, y_pred, X=None):
        """Update the detector with a new sample.

        Args:
            y_true: one true label from input data
            y_pred: one predicted label from input data
            X: one row of features from input data. Not used by STEPD.
        """
        if self.drift_state == "drift":
            self.reset()

        _, y_true, y_pred = super()._validate_input(None, y_true, y_pred)
        super().update(None, y_true, y_pred)
        # the arrays should have a single element after validation.
        y_true, y_pred = y_true[0], y_pred[0]
        classifier_result = int(y_pred == y_true)
        self._s += classifier_result

        # update r and s and trim the window
        self._window.append(classifier_result)
        if len(self._window) > self.window_size:
            # points that are larger than the buffer are removed, and only
            # tracked through r and n
            self._s -= self._window[0]
            self._r += self._window[0]
            self._window = self._window[1:]

        if self.samples_since_reset >= 2 * self.window_size:
            recent_accuracy = self.recent_accuracy()
            past_accuracy = self.past_accuracy()
            overall_accuracy = self.overall_accuracy()
            self._test_statistic = (
                np.absolute(past_accuracy - recent_accuracy)
                - 0.5
                * (
                    (1 / (self.samples_since_reset - self.window_size))
                    + (1 / self.window_size)
                )
            ) / np.sqrt(
                overall_accuracy
                * (1 - overall_accuracy)
                * (
                    (1 / (self.samples_since_reset - self.window_size))
                    + (1 / self.window_size)
                )
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
        Returns:
            float: the accuracy of the classifier among the last
            ``self.window_size`` samples the detector has seen
        """
        if len(self._window) == 0:
            out = 0
        else:
            out = self._s / len(self._window)
        return out

    def past_accuracy(self):
        """
        Returns:
            float: the accuracy of the classifier among the samples the detector
            has seen before its current window, but after the last time the
            detector was reset
        """
        if (self.samples_since_reset - len(self._window)) == 0:
            out = 0
        else:
            out = self._r / (self.samples_since_reset - len(self._window))
        return out

    def overall_accuracy(self):
        """
        Returns:
            float: the accuracy of the classifier among the samples the detector
            has seen since the detector was last reset
        """
        if self.samples_since_reset == 0:
            out = 0
        else:
            out = (self._r + self._s) / (self.samples_since_reset)
        return out

    def _initialize_retraining_recs(self):
        """Sets `self._retraining_recs` to ``[None, None]``."""
        self._retraining_recs = np.array([None, None])

    def _increment_retraining_recs(self):
        """Set ``self._retraining_recs`` to the beginning and end of the current
        drift/warning region.
        """
        if self._retraining_recs[0] is None:
            self._retraining_recs[0], self._retraining_recs[1] = (
                self.total_samples - 1,
                self.total_samples - 1,
            )
        else:
            self._retraining_recs[1] += 1

    @property
    def retraining_recs(self):
        """
        Returns:
            list: the current retraining recommendations
        """
        return self._retraining_recs
