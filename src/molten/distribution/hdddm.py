import numpy as np
from molten.drift_detector import DriftDetector
import scipy.stats
from sklearn.model_selection import KFold  # TODO add to requirements.txt?


class HDDDM(DriftDetector):

    """HDDDM is a batch-based, unsupervised drift detection algorithm that detects changes in feature distributions.
    It relies upon the Hellinger distance between batches to detect gradual or abrupt changes in data.

    1. Estimates density functions of reference and test data using histograms. The number of bins in each histogram
    equals the square root of the length of reference window. Bins are exactly aligned by computing the minimum and
    maximum value for each feature from both test and reference window.
    2. Computes average Hellinger distance between reference and test distributions. Stores Hellinger distances
    for each feature
    3. Computes epsilon: the differences in Hellinger distances between sets of reference and test batches. Computes epsilon
    between individual features distances.
    4. Computes adaptive threshold beta
    5. Detects drift if distance between sets windows is signficant, by comparing current epsilon value
    to adaptive threshold.


    Inbuilt modifications to original algorithm:
    - For aid of identifying and visualizing features containing most significant drift, HDDDM stores both the
    Hellinger distance and epsilon values for each feature individually.
    - Original algorithm cannot detect drift until 3rd test batch. On the second test batch only, HDDDM uses
    a bootstrapped estimate of an initial epsilon value to allow for drift to be detected on the second test batch. On the third test batch,
    this value is removed from all proceeeding epsilon and beta calculations.

    Ref. G. Ditzler and R. Polikar, "Hellinger distance based drift detection for nonstationary
    environments," 2011 IEEE Symposium on Computational Intelligence in Dynamic and Uncertain
    Environments (CIDUE), 2011, pp. 41-48, doi: 10.1109/CIDUE.2011.5948491.

    Attributes:
        total_samples (int): number of batches the drift detector has ever
            been updated with
        samples_since_reset (int): number of batches since the last drift detection.
        drift_state (str): detector's current drift state. Can take values
            "drift", "warning", or None.
        epsilon (list): stores epsilon values since the last drift detection.
        reference_n (int): number of samples in reference batch
        total_epsilon (int): stores running sum of epsilon values until drift is detected,
            initialized to 0
        bins (int): number of bins in histograms, equivalent to square root of number of
            samples in reference batch
        d (int): number of features in reference batch
        lambda (int): batch number on which last drift was detected
    """

    def __init__(
        self,
        reference_batch,
        statistic="tstat",
        significance=0.05,
        subsets=5,
    ):

        """
        Args:
            reference_batch (DataFrame): initial baseline dataset
            statistic (str): statistical method used to compute adaptive threshold. Defaults to "tstat"
                "tstat" - t-statistic with desired significance level and degrees of freedom = 2
                            for hypothesis testing on two populations
                "stdev" - uses number of standard deviations deemed significant to compute threhsold
            significance (float): statistical significance used to identify adaptive threshold. Defaults to 0.05
                if statistic = "tstat" - statistical significance of t-statistic
                if statistic = "stdev" - number of standard deviations of change around the mean accepted
            subsets (int): the number of subsets of reference data to take to compute initial estimate of epsilon.
                If too small, initial epsilon value will be too small. Increases risk of missing drift
                If too high, intial epsilon value will be too large. Increases risk of false alarms

        TODO how to make it so statistic and significance correspond? If statistic = tstat, need significance to be a decimal corresponding to
        % signifiance (ex: 0.05 for 95%). If statistic = stdev, need significance to be an integer for number of standard deviations deemed significant
        """

        super().__init__()

        # Initialize parameters
        self.reference = reference_batch
        self.statistic = statistic
        self.significance = significance
        self.subsets = subsets

        # Initial attributes
        self.epsilon = []
        self.reference_n = self.reference.shape[0]
        self.total_epsilon = 0
        self.bins = int(np.floor(np.sqrt(self.reference_n)))
        self.d = self.reference.shape[1]
        self._lambda = 0

        # For visualizations
        self.distances = {}
        self.epsilon_values = {}
        self.thresholds = {}

    def update(self, test_batch):
        """Update the detector with a new test batch.

        Args:
          test_batch (DataFrame): next batch of data to detect drift on
        """

        # TODO return error if # features in ref != test? the dataframes must match exactly

        if self._drift_state == "drift":
            self.reset()

        super().update()
        test_n = test_batch.shape[0]

        # Estimate histograms
        mins = []
        maxes = []
        for f in range(self.d):
            reference_variable = self.reference.iloc[:, f]
            test_variable = test_batch.iloc[:, f]
            mins.append(np.concatenate((reference_variable, test_variable)).min())
            maxes.append(np.concatenate((reference_variable, test_variable)).max())
        self.reference_density = self.build_histograms(self.reference, mins, maxes)
        test_density = self.build_histograms(test_batch, mins, maxes)

        # Hellinger distances
        self.current_distance, feature_distances = self.hellinger_distance(
            self.reference_density, test_density
        )
        self.distances[self.total_samples] = self.current_distance

        # Difference in distances for each feature
        if self.total_samples > 1:
            self.feature_epsilons = [
                a_i - b_i
                for a_i, b_i in zip(feature_distances, self.prev_feature_distances)
            ]

        # Compute epsilon and beta after two test batches (two values of Hellinger distances)
        if self.samples_since_reset > 1:

            if self.samples_since_reset == 2:
                initial_epsilon = self.estimate_initial_epsilon(
                    self.reference, self.subsets, mins, maxes
                )
                self.epsilon.append(initial_epsilon)

            current_epsilon = abs(self.current_distance - self.prev_hellinger_distance)
            self.epsilon.append(current_epsilon)

            self.beta = self.adaptive_threshold(self.statistic, test_n)

            self.epsilon_values[self.total_samples] = current_epsilon
            self.thresholds[self.total_samples] = self.beta

            # Detect drift
            if current_epsilon > self.beta:

                self.feature_info = {
                    "differences in distances": self.feature_epsilons,
                    "hellinger distances": feature_distances,
                    "largest drift in variable ": self.feature_epsilons.index(
                        max(self.feature_epsilons)
                    ),
                }
                self._drift_state = "drift"
                self.reference = test_batch
                self._lambda = self.total_samples

        if self._drift_state != "drift":
            self.prev_hellinger_distance = self.current_distance
            self.prev_feature_distances = feature_distances
            self.reference = self.reference.append(test_batch)
            self.reference_n = self.reference.shape[0]
            self.bins = int(np.floor(np.sqrt(self.reference_n)))

    def reset(self):
        """Initialize relevant attributes to original values, to ensure information only stored from
        samples_since_reset (lambda) onwards. Intended for use after drift_state == 'drift'."""

        super().reset()
        self.reference_n = self.reference.shape[0]
        self.bins = int(np.floor(np.sqrt(self.reference_n)))
        self.epsilon = []
        self.total_epsilon = 0

    def build_histograms(self, dataset, min_values, max_values):
        """Computes histogram for each feature in dataset. Bins are equidistantly spaced from minimum
        value to maximum value to ensure exact alignment of bins between test and reference data sets.

        Args:
            dataset (DataFrame): DataFrame on which to estimate density using histograms
            min_values (list): List of the minimum value for each feature
            max_values (list): List of the maximum value for each feature

        Returns:
            List of histograms for each feature. Histograms stored as list of frequency count of data in each bin.

        """

        histograms = [
            np.histogram(
                dataset.iloc[:, f], bins=self.bins, range=(min_values[f], max_values[f])
            )[0]
            for f in range(self.d)
        ]

        return histograms

    def hellinger_distance(self, reference_density, test_density):
        """Computes hellinger distance between reference and test histograms, averaging across features.

        Args:
            reference_density (list): Output of build_histograms on reference batch
            test_density (list): Output of build_histograms on test batch

        Returns:
            Average hellinger distance across features.
            List of undividual hellinger distance for each feature

        """

        feature_distances = []
        total_distance = 0
        for f in range(self.d):
            f_distance = 0
            r_density = reference_density[f]
            r_length = sum(r_density)
            t_density = test_density[f]
            t_length = sum(t_density)
            for b in range(self.bins):
                f_distance += (
                    np.sqrt(t_density[b] / t_length) - np.sqrt(r_density[b] / r_length)
                ) ** 2
            feature_distances.append(f_distance)
            total_distance += np.sqrt(f_distance)

        return (1 / self.d) * total_distance, feature_distances

    def adaptive_threshold(self, stat, test_n):
        """Computes adaptive threshold. If computing threshold for third test
        batch, removes our estimate of epsilon_0 from future estimates of
        epsilon_hat and std.

        Args:
            stat (string): Desired statistical method for computing threshold
            test_n (integer): Number of samples in test batch

        Returns:
            Adaptive threshold Beta
        """

        if self.samples_since_reset == 3:
            self.total_epsilon -= self.epsilon[0]
            self.epsilon = self.epsilon[1:]

        # update scale for denominator (t - lambda - 1), accounting for our initial epsilon 0 estimate
        if self.samples_since_reset == 1:
            d_scale = 1
        else:
            d_scale = self.total_samples - self._lambda - 1

        # Increment running mean of epsilon from samples_since_reset (lambda) -> t-1
        self.total_epsilon += self.epsilon[-2]
        epsilon_hat = (1 / d_scale) * self.total_epsilon

        # Compute standard deviation for samples_since_reset (lambda) -> t-1
        total_stdev = sum(
            (self.epsilon[i] - epsilon_hat) ** 2 for i in range(len(self.epsilon) - 1)
        )
        stdev = np.sqrt(total_stdev / (d_scale))

        if stat == "tstat":
            t_stat = scipy.stats.t.ppf(
                1 - (self.significance / 2), self.reference_n + test_n - 2
            )
            beta = epsilon_hat + t_stat * (stdev / np.sqrt(d_scale))

        else:
            beta = epsilon_hat + self.significance * stdev

        return beta

    def estimate_initial_epsilon(
        self, reference, num_subsets, histogram_mins, histogram_maxes
    ):
        """Computes a bootstrapped initial estimate of epsilon on 2nd test batch, allowing HDDDM
        to detect drift on the 2nd batch.

        1. Uses k-fold method to sample subsets reference data
        2. Computes hellinger distance between each subset
        3. Computes epsilon: difference in hellinger distances
        4. Averages epsilon estimates

        Args:
            reference (DataFrame): DataFrame consists of reference batch and first test batch
            num_subsets (Int): K, desired number of subsets to be sampled from reference data
            histogram_mins (List): List of minimum values for each feature align histogram bins
            histogram_maxes (List): List of maximum values for each feature align histogram bins

        Returns:
            Bootstrapped estimate of intial epsilon value
        """

        # Resampling data
        bootstraps = []
        kf = KFold(n_splits=num_subsets, random_state=1, shuffle=True)
        for train, test in kf.split(reference):
            bootstraps.append(
                self.build_histograms(
                    reference.iloc[test, :], histogram_mins, histogram_maxes
                )
            )

        # Hellinger distance between each subset
        distances = []
        for df_indx in range(len(bootstraps)):
            j = df_indx + 1
            while j < len(bootstraps):
                distances.append(
                    self.hellinger_distance(bootstraps[df_indx], bootstraps[j])[0]
                )
                j += 1

        # Epsilons between each distance
        epsilon = 0
        for delta_indx in range(len(distances)):
            j = delta_indx + 1
            while j < len(distances):
                epsilon += abs(distances[delta_indx] - distances[j])
                j += 1

        epsilon0 = epsilon / num_subsets

        return epsilon0
