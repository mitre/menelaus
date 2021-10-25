import pandas as pd
from molten.DriftDetector import DriftDetector
'''
TODO
similar to PCA_CD, need to, at a minimum, make the internal tracker objects optional
    this might break some code that PCA_CD relies on.
'''

class PageHinkley(DriftDetector):
    """
    Page-Hinkley is a univariate concept drift detection algorithm, designed to detect changes in a sequential Gaussian
    signal. Both the running mean and the running Page Hinkley (PH) statistic are incremented with each observation.
    The PH stat monitors how far the current observation is from the running mean of all previously encountered
    observations, while weighting it by a sensitivity parameter delta.The detector alarms when the difference between
    the maximum or minimum PH statistic encountered is larger than the cumulative PH statistic certain threshold (xi).

    1. Increment mean with next observations
    2. Increment running sum of difference between observations and mean
    3. Compute threshold & PH statistic
    4. Enter drift or warning state if PH value is outside threshold, and the number of samples is greater than the
        burn-in requirement.


    If the threshold is too small, PH may result in many false alarms. If too large, the PH test will be more robust,
    but may miss true drift.

    References:
    D.V.Hinkley. 1971. Inference about the change-point from cumulative sum tests. Biometrika 58, 3 (1971), 509-523


    """

    def __init__(self, delta = 0.01, xi = 20, burn_in=30, direction='positive'):
        """
        :param delta: float
            Minimum amplitude of change in data needed to sound alarm
        :param xi: float
            Threshold for sounding alarm. Corresponds with PH lambda. As suggested in PCA-CD, Qahton (2015)
            recommends setting to 1% of an appropriate window size for the dataset.
        :param burn_in: int (default = 0)
            Minimum number of data points required to be seen before declaring drift
        :param direction: str (default = 'positive')
            If 'positive', drift is only detected for an upward change in mean, when the cumulative PH statistic
            differs from the minimum PH statistic significantly
            If 'negative', drift is only detected for a downward change in mean, when the max PH statistic differs from
            the cumulative PH statistic significantly
        """
        super().__init__()

        self.burn_in = burn_in
        self.delta = delta
        self.xi = xi
        self.direction = direction

        self.max = 0
        self.min = 0
        self.sum = 0
        self.mean = 0

        self.ids = []
        self.change_scores = []
        self.page_hinkley_values = []
        self.page_hinkley_differences = []
        self.theta_threshold = []
        self.drift_detected = []
        self.maxes = []
        self.mins = []
        self.means = []

    def update(self, next_obs, obs_id=None):
        """
        Update the detector with a new sample.
        :param next_obs: new sample
        :param obs_id: index of new sample to store in dataframe
        """
        if self.drift_state == 'drift':
            self.reset()
        super().update()

        self.mean = self.mean + (next_obs - self.mean) / self.n
        self.sum = self.sum + next_obs - self.mean - self.delta
        theta = self.xi * self.mean

        if self.sum < self.min:
            self.min = self.sum

        if self.sum > self.max:
            self.max = self.sum

        if self.direction == 'positive':
            ph_difference = self.sum - self.min
        elif self.direction == 'negative':
            ph_difference = self.max - self.sum

        drift_check = (ph_difference > theta)

        if drift_check and self.n > self.burn_in:
            self.drift_state = 'drift'

        self.ids.append(obs_id)
        self.change_scores.append(next_obs)
        self.page_hinkley_values.append(self.sum)
        self.page_hinkley_differences.append(ph_difference)
        self.drift_detected.append(drift_check)
        self.theta_threshold.append(theta)

        self.maxes.append(self.max)
        self.mins.append(self.min)
        self.means.append(self.mean)

    def reset(self):
        """
        Initialize the detector's drift state and other relevant attributes. Intended for use after
        drift_state == 'drift'.
        """
        super().reset()
        self.max = 0
        self.min = 0
        self.sum = 0
        self.mean = 0

        self.ids = []
        self.change_scores = []
        self.page_hinkley_values = []
        self.page_hinkley_differences = []
        self.theta_threshold = []
        self.drift_detected = []

        self.maxes = []
        self.mins = []
        self.means = []

    def to_dataframe(self):
        """
        Returns a dataframe storing current statistics
        """
        return pd.DataFrame({
            "ids": self.ids,
            "change_scores": self.change_scores,
            "page_hinkley_values": self.page_hinkley_values,
            "page_hinkley_differences": self.page_hinkley_differences,
            "theta_threshold": self.theta_threshold,
            "drift_detected": self.drift_detected,
            "maximum_sum_values": self.maxes,
            "minimum_sum_values": self.mins,
            "mean_values": self.means
        })
