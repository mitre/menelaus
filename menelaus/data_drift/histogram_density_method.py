import copy
import numpy as np
import pandas as pd
import scipy.stats
from menelaus.detector import BatchDetector
from scipy.spatial.distance import jensenshannon


class HistogramDensityMethod(BatchDetector):
    """
    The Histogram Density Method (HDM) is the base class for both HDDDM and
    CDBD. HDDDM differs from CDBD by relying upon the Hellinger distance measure
    while CDBD uses KL divergence.

    This method relies upon three statistics:

        * Distance metric:

            * Hellinger distance (default if called via HDDDM): the sum of the
              normalized, squared differences in frequency counts for each bin
              between reference and test datasets, averaged across all features.

            * KL divergence (default if called via CDBD): Jensen-Shannon
              distances, a symmetric and bounded measure based upon
              Kullback-Leibler Divergence

            * Optional user-defined distance metric

        * Epsilon: the differences in Hellinger distances between sets of
          reference and test batches.

        * Beta: the adaptive threshold adapted at each time stamp. It is based
          on the mean of Epsilon plus the scaled standard deviation of Epsilon.
          The scale applied to the standard deviation is determined by the
          ``statistic`` parameter. It is either the number of standard
          deviations deemed significant (``"stdev"``) or the t-statistic
          (``"tstat"``).

    HDM operates by:

        #. Estimating density functions of reference and test data using
           histograms. The number of bins in each histogram equals the square
           root of the length of reference window. Bins are aligned by computing
           the minimum and maximum value for each feature from both test and
           reference window.

        #. Computing the distance between reference and test distributions. In
           HDDDM, the Hellinger distance is first calculated between each
           feature in the reference and test batches. Then, the final Hellinger
           statistic used is the average of each feature's distance. In CDBD,
           the KL divergence metric is used to calculate the distance between
           univariate histograms.

        #. Computing Epsilon.

        #. Computing the adaptive threshold Beta.

        #. Comparing current Epsilon to Beta. If Epsilon > Beta, drift is
           detected. The new reference batch is now the test batch on which
           drift was detected. All statistics are reset. If Epsilon <= Beta,
           drift is not detected. The reference batch is updated to include this
           most recent test batch. All statistics are maintained.

    Two key modifications were added to Ditzler and Polikar's presentation:

        * For HDDDM, to answer the question of "where is drift occuring?", it
          stores the distance values and Epsilon values for each feature. These
          statistics can be used to identify and visualize the features
          containing the most significant drifts.

            * The Hellinger distance values are calculated for each feature in
              the test batch. These values can be accessed when drift occurs
              using the self.feature_info dictionary.

            * The Epsilon values for each feature are stored, for each set of
              reference and test batches. For each feature, these values
              represent the difference in Hellinger distances within the test
              and reference batch at time *t*, to the Hellinger distances within
              the test and reference batch at time *t-1*. These can be acccessed
              with each update call using the self.feature_epsilons variable.
              They also can be accessed when drift occurs using the
              self.feature_info dictionary.

        * The original algorithm cannot detect drift until it is updated with
          the third test batch after either a) initilization or b) reset upon
          drift, because the threshold for drift detection is defined from the
          *difference* Epsilon. To have sufficient values to define this
          threshold, then, three batches are needed. The ``detect_batch``
          parameter can be set such that bootstrapping is used to define this
          threshold earlier than the third test batch.

            * if ``detect_batch == 3``, HDM will operate as described in
              :cite:t:`ditzler2011hellinger`.

            * if ``detect_batch == 2``, HDM will detect drift on the second test
              batch. On the second test batch, HDM uses bootstrapped samples
              from the reference batch to estimate the mean and standard
              deviation of Epsilon; this is used to calculate the necessary
              threshold. On the third test batch, this value is removed from all
              proceeding

            * if ``detect_batch`` = 1, HDM will detect drift on the first test
              batch. The initial reference batch is split randomly into two
              halves. The first half will serve as the original reference batch.
              The second half will serve as a proxy for the first test batch,
              allowing us to calculate the distance statistic. When HDM is
              updated with the first actual test batch, HDM will perform the
              method for bootstrapping Epsilon, as described in the above bullet
              for ``detect_batch`` = 2. This will allow a Beta threshold to be
              calculated using the first test batch, allowing for detection of
              drift on this batch.

    Ref. :cite:t:`lindstrom2013drift` and :cite:t:`ditzler2011hellinger`

    Attributes:
        Epsilon (list): stores Epsilon values since the last drift detection.
        reference_n (int): number of samples in reference batch.
        total_epsilon (int): stores running sum of Epsilon values until drift is
            detected, initialized to 0.
        distances (dict): For each batch seen (key), stores the distance between
            test and reference batch (value). Useful for visualizing drift
            detection statistics.
        epsilon_values (dict):For each batch seen (key), stores the Epsilon
            value between the current and previous test and reference batches
            (value). Useful for visualizing drift detection statistics. Does not
            store the bootstrapped estimate of Epsilon, if used.
        thresholds (dict): For each batch seen (key), stores the Beta thresholds
            between test and reference batch (value). Useful for visualizing
            drift detection statistics.
    """

    input_type = "batch"

    def __init__(
        self,
        divergence,
        detect_batch,
        statistic,
        significance,
        subsets,
    ):

        """
        Args:
            divergence (str or function): divergence measure used to compute
                distance between histograms. Default is "H".

                * "H"  - Hellinger distance, original use is for HDDDM

                * "KL" - Kullback-Leibler Divergence, original use is for CDBD

                * User can pass in custom divergence function. Input is two
                  two-dimensional arrays containing univariate histogram
                  estimates of density, one for reference, one for test. It must
                  return the distance value between histograms. To be a valid
                  distance metric, it must satisfy the following properties:
                  non-negativity, identity, symmetry, and triangle inequality,
                  e.g. that in examples/cdbd_example.py or
                  examples/hdddm_example.py.

            detect_batch (int): the test batch on which drift will be detected.
                See class docstrings for more information on this modification.
                Defaults to 1.

                * if ``detect_batch == 1`` - HDM can detect drift on the first
                  test batch passed to the update method

                * if ``detect_batch == 2`` - HDM can detect drift on the second
                  test batch passed to the update method

                * if ``detect_batch == 3`` - HDM can detect drift on the third
                  test batch passed to the update method

            statistic (str): statistical method used to compute adaptive
                threshold. Defaults to ``"tstat"``.

                * ``"tstat"`` - t-statistic with desired significance level and
                  degrees of freedom = 2 for hypothesis testing on two
                  populations

                * ``"stdev"`` - uses number of standard deviations deemed
                  significant to compute threhsold

            significance (float): statistical significance used to identify
                adaptive threshold. Defaults to 0.05.

                * if statistic = ``"tstat"`` - statistical significance of
                  t-statistic, e.g. .05 for 95% significance level

                * if statistic = ``"stdev"`` - number of standard deviations of
                  change around the mean accepted

            subsets (int): the number of subsets of reference data to take to
                compute initial estimate of Epsilon.

                * if too small - initial Epsilon value will be too small.
                  Increases risk of missing drift

                * if too high - intial Epsilon value will be too large.
                  Increases risk of false alarms.
        """

        super().__init__()

        # Initialize parameters
        self.detect_batch = detect_batch
        self.statistic = statistic
        self.significance = significance
        self.subsets = subsets
        if divergence == "H":
            self.distance_function = self._hellinger_distance
        elif divergence == "KL":
            self.distance_function = self._KL_divergence
        else:
            self.distance_function = divergence

        self._lambda = 0  # batch number on which last drift was detected.

        # For visualizations
        self.distances = {}
        self.epsilon_values = {}
        self.thresholds = {}

    def set_reference(self, X, y_true=None, y_pred=None):
        """
        Initialize detector with a reference batch. After drift, reference batch
        is automatically set to most recent test batch. Option for user to
        specify alternative reference batch using this method.

        Args:
            X (pandas.DataFrame): initial baseline dataset
            y_true (numpy.array): true labels for dataset - not used by HDM
            y_pred (numpy.array): predicted labels for dataset - not used by HDM
        """
        X, _, _ = super()._validate_input(X, None, None)
        X = pd.DataFrame(
            X, columns=self._input_cols
        )  # TODO: subsequent operations expect dataframes, not numpy arrays
        # Initialize attributes
        self.reference = copy.deepcopy(X)
        self.reset()

    def update(self, X, y_true=None, y_pred=None):

        """
        Update the detector with a new test batch. If drift is detected, new
        reference batch becomes most recent test batch. If drift is not
        detected, reference batch is updated to include most recent test batch.

        Args:
            X (DataFrame): next batch of data to detect drift on.
            y_true (numpy.ndarray): true labels of next batch - not used in HDM
            y_pred (numpy.ndarray): predicted labels of next batch - not used in HDM
        """

        if self._drift_state == "drift":
            self.reset()

        X, _, _ = super()._validate_input(X, None, None)
        X = pd.DataFrame(
            X, columns=self._input_cols
        )  # TODO: subsequent operations expect dataframes, not numpy arrays

        super().update(X, None, None)
        test_n = X.shape[0]

        # Estimate reference and test histograms
        mins = []
        maxes = []
        for f in range(self._input_col_dim):
            reference_variable = self.reference.iloc[:, f]
            test_variable = X.iloc[:, f]
            mins.append(np.concatenate((reference_variable, test_variable)).min())
            maxes.append(np.concatenate((reference_variable, test_variable)).max())
        self._reference_density = self._build_histograms(self.reference, mins, maxes)
        test_density = self._build_histograms(X, mins, maxes)

        # Divergence metric
        total_distance = 0
        feature_distances = []
        for f in range(self._input_col_dim):
            f_distance = self.distance_function(
                self._reference_density[f], test_density[f]
            )
            total_distance += f_distance
            feature_distances.append(f_distance)
        self.current_distance = (1 / self._input_col_dim) * total_distance
        self.distances[self.total_batches] = self.current_distance

        # For each feature, calculate Epsilon, difference in distances
        if self.total_batches > 1:
            self.feature_epsilons = [
                a_i - b_i
                for a_i, b_i in zip(feature_distances, self._prev_feature_distances)
            ]

        # Compute Epsilon and Beta
        if self.batches_since_reset >= 2:

            if self.batches_since_reset == 2 and self.detect_batch != 3:
                initial_epsilon = self._estimate_initial_epsilon(
                    self.reference, self.subsets, mins, maxes
                )
                self.epsilon.append(initial_epsilon)

            current_epsilon = abs(self.current_distance - self._prev_distance) * 1.0
            self.epsilon.append(current_epsilon)
            self.epsilon_values[self.total_batches] = current_epsilon

            condition1 = bool(self.batches_since_reset >= 2 and self.detect_batch != 3)
            condition2 = bool(self.batches_since_reset >= 3 and self.detect_batch == 3)
            if condition1 or condition2:

                self.beta = self._adaptive_threshold(self.statistic, test_n)
                self.thresholds[self.total_batches] = self.beta

                # Detect drift
                if current_epsilon > self.beta:

                    # Feature information
                    if self._input_col_dim > 1:

                        self.feature_info = {
                            "Epsilons": self.feature_epsilons,
                            "Feature_Distances": feature_distances,
                            "Significant_drift_in_variable ": self.feature_epsilons.index(
                                max(self.feature_epsilons)
                            ),
                        }

                    self._drift_state = "drift"
                    self.reference = X
                    self._lambda = self.total_batches

        if self._drift_state != "drift":
            self._prev_distance = self.current_distance
            self._prev_feature_distances = feature_distances
            self.reference = pd.concat([self.reference, X])
            self.reference_n = self.reference.shape[0]
            # number of bins for histogram, from reference batch
            self._bins = int(np.floor(np.sqrt(self.reference_n)))

    def reset(self):
        """
        Initialize relevant attributes to original values, to ensure information
        only stored from batches_since_reset (lambda) onwards. Intended for use
        after ``drift_state == 'drift'``.
        """

        super().reset()

        if self.detect_batch == 1:
            # The reference and test data will be (re-)concatenated by the later
            # call to update(), since drift cannot be detected on the first
            # batch, in this case.
            test_proxy = self.reference.iloc[
                int(len(self.reference) / 2) :,
            ]
            self.reference = self.reference.iloc[
                0 : int(len(self.reference) / 2),
            ]

        self.reference_n = self.reference.shape[0]
        self._bins = int(np.floor(np.sqrt(self.reference_n)))
        self.epsilon = []
        self.total_epsilon = 0

        if self.detect_batch == 1:
            self.update(test_proxy)

    def _build_histograms(self, dataset, min_values, max_values):
        """
        Computes histogram for each feature in dataset. Bins are equidistantly
        spaced from minimum value to maximum value to ensure exact alignment of
        bins between test and reference data sets.

        Args:
            dataset (DataFrame): DataFrame on which to estimate density using
                histograms.
            min_values (list): List of the minimum value for each feature.
            max_values (list): List of the maximum value for each feature.

        Returns:
            List of histograms for each feature. Histograms stored as list of
            frequency count of data in each bin.

        """

        histograms = [
            np.histogram(
                dataset.iloc[:, f],
                bins=self._bins,
                range=(min_values[f], max_values[f]),
            )[0]
            for f in range(self._input_col_dim)
        ]

        return histograms

    def _hellinger_distance(self, reference_density, test_density):
        """
        Computes Hellinger distance between reference and test histograms

        Args:
            reference_density (list): Univariate output of _build_histograms
                from reference batch.
            test_density (list): Univariate tput of _build_histograms from test
                batch.

        Returns:
            Hellinger distance between univariate reference and test density

        """

        f_distance = 0
        r_length = sum(reference_density)
        t_length = sum(test_density)
        for b in range(self._bins):
            f_distance += (
                np.sqrt(test_density[b] / t_length)
                - np.sqrt(reference_density[b] / r_length)
            ) ** 2

        return np.sqrt(f_distance)

    def _adaptive_threshold(self, stat, test_n):
        """
        Computes adaptive threshold. If computing threshold for third test
        batch, removes our estimate of initial Epsilon from future estimates of
        epsilon_hat and std.

        Args:
            stat (string): Desired statistical method for computing threshold.
            test_n (integer): Number of samples in test batch.

        Returns:
            Adaptive threshold Beta.
        """

        if self.batches_since_reset == 3 and self.detect_batch != 3:
            self.total_epsilon -= self.epsilon[0]
            self.epsilon = self.epsilon[1:]

        # update scale for denominator (t - lambda - 1), accounting for our initial Epsilon estimate
        if self.batches_since_reset == 2 and self.detect_batch != 3:
            d_scale = 1
        else:
            d_scale = self.total_batches - self._lambda - 1

        # Increment running mean of epsilon from batches_since_reset (lambda) -> t-1
        self.total_epsilon += self.epsilon[-2]  # was -2 before total samples change...

        epsilon_hat = (1 / d_scale) * self.total_epsilon

        # Compute standard deviation for batches_since_reset (lambda) -> t-1
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

    def _estimate_initial_epsilon(
        self, reference, num_subsets, histogram_mins, histogram_maxes
    ):
        """Computes a bootstrapped initial estimate of Epsilon on 2nd test
        batch, allowing HDM to detect drift on the 2nd batch.

        1. Subsets reference data with replacement
        2. Computes distance between each subset.
        3. Computes Epsilon: difference in distances.
        4. Averages Epsilon estimates.

        Args:
            reference (DataFrame): DataFrame consists of reference batch and
                first test batch.
            num_subsets (int): desired number of subsets to be sampled from
                reference data.
            histogram_mins (list): List of minimum values for each feature align
                histogram bins.
            histogram_maxes (list): List of maximum values for each feature
                align histogram bins.

        Returns:
            Bootstrapped estimate of intial Epsilon value.
        """

        # Resampling data
        bootstraps = []
        size = int((1 - (1 / num_subsets)) * self.reference_n)
        for i in range(num_subsets):
            subset = reference.sample(n=size, replace=True)
            bootstraps.append(
                self._build_histograms(subset, histogram_mins, histogram_maxes)
            )

        # Distance between each subset
        distances = []
        for df_indx in range(len(bootstraps)):
            j = df_indx + 1
            while j < len(bootstraps):

                subset1 = bootstraps[df_indx]
                subset2 = bootstraps[j]

                # Divergence metric
                total_distance = 0
                for f in range(self._input_col_dim):
                    f_distance = self.distance_function(subset1[f], subset2[f])
                    total_distance += f_distance
                distances.append(total_distance)

                j += 1

        # Epsilons between each distance
        epsilon = 0
        for delta_indx in range(len(distances)):
            j = delta_indx + 1
            while j < len(distances):
                epsilon += abs(distances[delta_indx] - distances[j]) * 1.0
                j += 1

        epsilon0 = epsilon / num_subsets

        return epsilon0

    def _KL_divergence(self, reference_density, test_density):
        """
        Computes Jensen Shannon (JS) divergence between reference and test
        histograms. JS is a bounded, symmetric form of KL divergence.

        Args:
            reference_density (list): Univariate output of _build_histograms
                from reference batch.
            test_density (list): Univariate output of _build_histograms from
                test batch.

        Returns:
            JS divergence between univariate reference and test density

        """

        return jensenshannon(reference_density, test_density)
