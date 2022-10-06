import numpy as np
from menelaus.detector import StreamingDetector


class DDM(StreamingDetector):
    """DDM is a drift detection algorithm which uses a binary classifier's error
    rate, which is binomially distributed. The minimum probability of an error
    and its standard deviation (``p_min``, ``s_min``) are found during training. If the
    running estimates for element i in the stream, probability (``p_i``) and its
    standard deviation (``s_i``), exceeds a certain threshold, then we assume that
    the distribution of the error rate is no longer stationary (drift has
    occurred).

    If ``p_i + s_i >= p_min + self.warning_scale * s_min`` the detector's state
    is set to ``"warning"``.

    If ``p_i + s_i >= p_min + self.drift_scale * s_min``, the
    detector's state is set to ``"drift"``.

    The index of the first sample which triggered a warning/drift state
    (relative to ``self.samples_since_reset``) is stored in ``self.retraining_recs``.

    Ref. :cite:t:`gama2004learning`
    """

    input_type = "stream"

    def __init__(self, n_threshold=30, warning_scale=2, drift_scale=3):
        """
        Args:
            n_threshold (int, optional): the minimum number of samples required
                to test whether drift has occurred. Defaults to 30.
            warning_scale (int, optional): defines the threshold over which to
                enter the warning state. Defaults to 2.
            drift_scale (int, optional): defines the threshold over which to
                enter the drift state. Defaults to 3.
        """
        super().__init__()
        self.n_threshold = n_threshold
        self.warning_scale = warning_scale
        self.drift_scale = drift_scale

        self._error_rate = 0
        self._error_std = 0
        self._error_rate_min = float("inf")
        self._error_std_min = float("inf")
        self._initialize_retraining_recs()

    def reset(self):
        """Initialize the detector's drift state and other relevant attributes.
        Intended for use after ``drift_state == 'drift'``.
        """
        super().reset()
        self._error_rate = 0
        self._error_std = 0
        self._error_rate_min = float("inf")
        self._error_std_min = float("inf")
        self._initialize_retraining_recs()

    # XXX - Order of y_true, y_pred, X differs from abstractmethod signature
    #       for update(). This is done for convenience, so users can call e.g.
    #       DDM.update(1,1) without misinterpretation, but exposes them to a
    #       potential issue where LFR.update(X, y, y) would assign arguments
    #       incorrectly.
    def update(self, y_true, y_pred, X=None):
        """Update the detector with a new sample.

        Args:
            y_true: one true label from input data.
            y_pred: one predicted label from input data.
            X: one row of features from input data. Not used in DDM.
        """

        if self.drift_state == "drift":
            self.reset()

        _, y_true, y_pred = super()._validate_input(None, y_true, y_pred)
        super().update(None, y_true, y_pred)
        # the arrays should have a single element after validation.
        y_true, y_pred = y_true[0], y_pred[0]
        classifier_result = int(y_pred != y_true)

        # with each sample, update estimate of error and its std, along with minimums
        error_rate_prev = self._error_rate
        self._error_rate = (
            self._error_rate
            + (classifier_result - self._error_rate) / self.samples_since_reset
        )
        self._error_std = self._error_std + (classifier_result - self._error_rate) * (
            classifier_result - error_rate_prev
        )
        self._error_std = np.sqrt(self._error_std / self.samples_since_reset)

        # it's unclear whether the 'burn-in' period should be updating the
        # minimums - seems like a bad idea though.
        if self.samples_since_reset < self.n_threshold:
            return

        if (
            self._error_rate + self._error_std
            <= self._error_rate_min + self._error_std_min
        ):
            self._error_rate_min = self._error_rate
            self._error_std_min = self._error_std

        if (
            self._error_rate + self._error_std
            >= self._error_rate_min + self.drift_scale * self._error_std
        ):
            self.drift_state = "drift"
        elif (
            self._error_rate + self._error_std
            >= self._error_rate_min + self.warning_scale * self._error_std
        ):
            self.drift_state = "warning"
        else:
            self.drift_state = None

        if self.drift_state is not None:
            self._increment_retraining_recs()

    def _initialize_retraining_recs(self):
        """Sets ``self._retraining_recs`` to ``[None, None]``."""
        self._retraining_recs = [None, None]

    def _increment_retraining_recs(self):
        """Set ``self._retraining_recs`` to the beginning and end of the current
        drift/warning region.
        """
        if self.drift_state == "warning" and self._retraining_recs[0] is None:
            self._retraining_recs[0] = self.total_samples - 1

        if self.drift_state == "drift":
            self._retraining_recs[1] = self.total_samples - 1
            if self._retraining_recs[0] is None:
                self._retraining_recs[0] = self.total_samples - 1

    @property
    def retraining_recs(self):
        """Indices of the first and last recommended training samples. A list
        of length 2, containing ``[warning index, drift index]``. If no warning
        occurs, this will instead be ``[drift index, drift index]``. The latter
        should cause caution, as it indicates an abrupt change. Resets when to
        ``[None, None]`` after drift is detected.

        Returns:
            list: the current retraining recommendations
        """
        return self._retraining_recs
