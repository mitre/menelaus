#TODO: remove/make optional the internal tracking attributes, or set some limit --
    #  hypothetically several attributes could grow indefinitely in memory, if drift is never detected
    #  we do need some portion of the stream to actually calculate the necessary stats, though
#TODO (maybe): explain the meaning of delta
import numpy as np
import pandas as pd
from molten.DriftDetector import DriftDetector


class CUSUM(DriftDetector):
    """
    CUSUM is a method from the field of statistical process control. This detector tests for changes in the mean of a
    time series by calculating a moving average over recent observations. CUSUM can be used for tracking a single
    model performance metric, or could be applied to the mean of a feature variable of interest.

    Ref.
    E.S.Page. 1954. Continuous Inspection Schemes. Biometrika 41, 1/2 (1954),100-115
    """
    def __init__(self, target=None,sd_hat=None, burn_in=30, delta=0.005, threshold=50, direction=None):
        """
        :param target: Known mean of stream (e.g. validation accuracy). If None, will be inferred from observations in
            the burn-in window.
        :param sd_hat: Known standard deviation of stream (e.g. SD of validation accuracy). If None, will be inferred
            from observations in the burn-in window.
        :param burn_in: Length of the burn-in period, during which time no alarms will sound
        :param delta: The amount of "slack" in the CUSUM test statistic.
        :param threshold: The threshold at which the CUSUM test statistic is evaluated against.
        :param direction: If 'positive', drift is only considered when the stream drifts in the positive direction.
            If 'negative', drift is only considered when the stream drifts in the negative direction.
            If None, alarms to drift in either the positive or negative direction.
        """
        super().__init__()
        self.target = target
        self.sd_hat = sd_hat
        self.burn_in = burn_in
        self.delta = delta
        self.threshold = threshold
        self.direction = direction
        self.all_drift_states = []
        self.upper_bound = [0]
        self.lower_bound = [0]
        self.stream = []

    def reset(self):
        """
        Initialize the detector's drift state and other relevant attributes. Intended for use after
        drift_state == 'drift'.
        """
        # make last upper and lower bound = 0
        super().reset()
        self.upper_bound = [0]
        self.lower_bound = [0]

    def update(self, next_obs):
        """
        Update the detector with a new sample.
        :param next_obs: The value of the new sample.
        """
        # if the last run resulted in drift, reset everything
        if self.drift_state == 'drift':
            self.target = np.mean(self.stream[-30:])
            self.sd_hat = np.std(self.stream[-30:])
            self.reset()

        super().update()
        self.stream.append(next_obs)

        if self.n <= self.burn_in:
            self.all_drift_states.append(None)

        ### TODO what if self.target is None and we are less than 30 datapoints? cannot compute s_h/s_l, should we set those to 0?
        if (self.target is None) & (self.n < self.burn_in):
            s_h = 0
            s_l = 0
            self.upper_bound.append(s_h)
            self.lower_bound.append(s_l)

        # derive mean and sd from first n points if they are not specified
        if (self.target is None) & (self.n == self.burn_in):
            self.target = np.mean(self.stream)
            self.sd_hat = np.std(self.stream)

        # find new upper and lower cusum stats
        if self.target is not None:
            s_h = max(0, self.upper_bound[self.n - 1] + (self.stream[self.n - 1] - self.target) / self.sd_hat - self.delta)
            s_l = max(0, self.lower_bound[self.n - 1] - self.delta - (self.stream[self.n - 1] - self.target) / self.sd_hat)
            self.upper_bound.append(s_h)
            self.lower_bound.append(s_l)

        # check alarm if past burn in
        if self.n > self.burn_in:
            if self.direction == None:
                if (self.upper_bound[self.n] > self.threshold) | (self.lower_bound[self.n] > self.threshold):
                    self.all_drift_states.append('drift')
                    self.drift_state = 'drift'

                else:
                    self.all_drift_states.append(None)
            # TODO check if stats need to change for one sided
            elif self.direction == 'positive':
                if self.upper_bound[self.n] > self.threshold:
                    self.all_drift_states.append('drift')
                    self.drift_state = 'drift'
                else:
                    self.all_drift_states.append(None)
            elif self.direction == 'negative':
                if self.lower_bound[self.n] > self.threshold:
                    self.all_drift_states.append('drift')
                    self.drift_state = 'drift'
                else:
                    self.all_drift_states.append(None)


