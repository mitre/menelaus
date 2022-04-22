import numpy as np
from mendelaus.drift_detector import DriftDetector


class CUSUM(DriftDetector):
    """CUSUM is a method from the field of statistical process control. This
    detector tests for changes in the mean of a time series by calculating a
    moving average over recent observations. CUSUM can be used for tracking a
    single model performance metric, or could be applied to the mean of a
    feature variable of interest.

    Ref. E.S.Page. 1954. Continuous Inspection Schemes. Biometrika 41, 1/2
    (1954),100-115

    Attributes:
        total_samples (int): number of samples the drift detector has ever
            been updated with
        samples_since_reset (int): number of samples since the last time the
            drift detector was reset
        drift_state (str): detector's current drift state. Can take values
            ``"drift"`` or ``None``.
    """

    _input_type = "stream"

    def __init__(
        self,
        target=None,
        sd_hat=None,
        burn_in=30,
        delta=0.005,
        threshold=50,
        direction=None,
    ):
        """
        Args:
            target (float, optional): Known mean of stream (e.g. validation
                accuracy). If ``None``, will be inferred from observations in the
                burn-in window. Defaults to ``None``.
            sd_hat (float, optional): Known standard deviation of stream (e.g.
                SD of validation accuracy). If ``None``, will be inferred from
                observations in the burn-in window. Defaults to ``None``.
            burn_in (int, optional): Length of the burn-in period, during which
                time no alarms will sound. Defaults to 30.
            delta (float, optional): The amount of "slack" in the CUSUM test
                statistic. Defaults to 0.005.
            threshold (int, optional): The threshold at which the CUSUM test
                statistic is evaluated against. Defaults to 50.
            direction (str, optional):
                * If ``'positive'``, drift is only considered when the stream drifts
                  in the positive direction.
                * If ``'negative'``, drift is only considered when the stream drifts
                  in the negative direction.
                * If ``None``, alarms to drift in either the positive or negative
                  direction. Defaults to ``None``.
        """
        super().__init__()
        self.target = target
        self.sd_hat = sd_hat
        self.burn_in = burn_in
        self.delta = delta
        self.threshold = threshold
        self.direction = direction
        self._all_drift_states = []
        self._upper_bound = [0]
        self._lower_bound = [0]
        self._stream = []

    def reset(self):
        """Initialize the detector's drift state and other relevant attributes.
        Intended for use after ``drift_state == 'drift'``.
        """
        # make last upper and lower bound = 0
        super().reset()
        self._upper_bound = [0]
        self._lower_bound = [0]

    def update(self, next_obs):
        """Update the detector with a new sample.

        Args:
          next_obs: The value of the new sample.
        """
        # if the last run resulted in drift, reset everything
        if self.drift_state == "drift":
            self.target = np.mean(self._stream[-30:])
            self.sd_hat = np.std(self._stream[-30:])
            self.reset()

        super().update()
        self._stream.append(next_obs)

        if self.samples_since_reset <= self.burn_in:
            self._all_drift_states.append(None)

        # cannot compute s_h/s_l, should we set those to 0?
        if (self.target is None) & (self.samples_since_reset < self.burn_in):
            s_h = 0
            s_l = 0
            self._upper_bound.append(s_h)
            self._lower_bound.append(s_l)

        # derive mean and sd from first n points if they are not specified
        if (self.target is None) & (self.samples_since_reset == self.burn_in):
            self.target = np.mean(self._stream)
            self.sd_hat = np.std(self._stream)

        # find new upper and lower cusum stats
        if self.target is not None:
            s_h = max(
                0,
                self._upper_bound[self.samples_since_reset - 1]
                + (self._stream[self.samples_since_reset - 1] - self.target)
                / self.sd_hat
                - self.delta,
            )
            s_l = max(
                0,
                self._lower_bound[self.samples_since_reset - 1]
                - self.delta
                - (self._stream[self.samples_since_reset - 1] - self.target)
                / self.sd_hat,
            )
            self._upper_bound.append(s_h)
            self._lower_bound.append(s_l)

        # check alarm if past burn in
        if self.samples_since_reset > self.burn_in:
            if self.direction is None:
                if (self._upper_bound[self.samples_since_reset] > self.threshold) | (
                    self._lower_bound[self.samples_since_reset] > self.threshold
                ):
                    self._all_drift_states.append("drift")
                    self.drift_state = "drift"

                else:
                    self._all_drift_states.append(None)
            elif self.direction == "positive":
                if self._upper_bound[self.samples_since_reset] > self.threshold:
                    self._all_drift_states.append("drift")
                    self.drift_state = "drift"
                else:
                    self._all_drift_states.append(None)
            elif self.direction == "negative":
                if self._lower_bound[self.samples_since_reset] > self.threshold:
                    self._all_drift_states.append("drift")
                    self.drift_state = "drift"
                else:
                    self._all_drift_states.append(None)
