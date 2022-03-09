import numpy as np
from molten.drift_detector import DriftDetector
import scipy.stats

class HDDDM(DriftDetector):

    """ HDDDM is a batch-based, unsupervised drift detection algorithm that
    detects changes in feature distributions. It uses the Hellinger
    distance metric to compare test and reference batches and is capable of
    detecting gradual or abrupt changes in data. HDDDM relies upon three
    statistics: 
        - Hellinger distance: the sum of the normalized, squared differences in
          frequency counts for each bin between reference and test datasets,
          averaged across all features. 
        - Epsilon: the differences in Hellinger distances between sets
           of reference and test batches.
        - Beta: the adaptive threshold adapted at each time stamp. It is
          based on the mean of Epsilon plus the scaled standard
          deviation of Epsilon. The scale applied to the standard deviation is
          determined by the "statistic" parameter. It is either the number of
          standard deviations deemed significant ("stdev") or the t-statistic
          ("tstat"). 

    HDDDM operates by: 

        1. Estimating density functions of reference and test data using
           histograms. The number of bins in each histogram equals the square
           root of the length of reference window. Bins are aligned
           by computing the minimum and maximum value for each feature from both
           test and reference window.

        2. Computing the Hellinger distance between reference and test
           distributions. The Hellinger distance is first calculated between
           each feature in the reference and test batches. Then, the final
           Hellinger statistic used is the average of each feature's distance.

        3. Computing Epsilon.

        4. Computing the adaptive threshold Beta.

        5. Comparing current Epsilon to Beta. If Epsilon > Beta, drift is
           detected. The new reference batch is now the test batch on which
           drift was detected. All statistics are reset. If Epsilon <= Beta,
           drift is not detected. The reference batch is updated to include this
           most recent test batch. All statistics are maintained. 

    Two key modifications were added to Ditzler and Polikar's presentation of
    HDDDM: 
        - To answer the research question of "Where is drift occuring?", HDDDM
          stores the Hellinger distance values and Epsilon values for each
          feature. These statistics can be used to identify and visualize the
          features containing the most significant drifts. 
            - The Hellinger distance values are calculated for each feature in
              the test batch. These values can be accessed when drift
              occurs using the self.feature_info dictionary. 
            - The Epsilon values for each feature are stored, for each set of
              reference and test batches. For each feature, these values
              represent the difference in Hellinger distances within the test
              and reference batch at time t, to the Hellinger distances within
              the test and reference batch at time t-1. These can be acccessed
              with each update call using the self.feature_epsilons variable.
              They also can be accessed when drift occurs using the
              self.feature_info dictionary. 

        - The original algorithm cannot detect drift until it is updated with the
          third 3rd test batch after initial setup or 3rd test batch after each drift
          detection. When HDDDM is updated with the first test batch (T=1), the
          Hellinger distance is calculated between test and reference
          batch. WHen updated with the second test batch (T=2), HDDDM calculates the
          Hellinger distance and Epsilon, the difference in
          distances. When HDDDM is updated with the third test batch (T=3), the
          Hellinger distance and Epsilon are calculated. The adaptive threshold
          Beta is computed using Epsilon values from lambda -> T-1 (lambda is
          either T=0 or the test batch number on which drift was last detected).
          Beta needs at least one prior value of Epsilon to be calculated. Now
          that Beta can be calculated, HDDDM can detect drift on the 3rd test
          batch. In order to allow for more immediate detection of drift, we
          added the following options, specified through the parameter
          "detect_batch":
            - if "detect_batch" = 3, HDDDM will operate as intended, as
              described above. 
            - if "detect_batch = 2, HDDDM will detect drift on the second test
              batch. On the second test batch only, HDDDM uses a bootstrapped estimate of
              an initial Epsilon value, to serve as a proxy for the first value of
              Epsilon occurring at T-1. This initial estimate of Epsilon is used to calculate Beta,
              allowing to allow for drift to be detected on the
              second test batch. On the third test batch, this value is removed from
              all proceeeding Epsilon and Beta calculations. 
            - if "detect_batch" = 1, HDDDM will detect drift on the first test
              batch. The initial reference batch is split randomly into two
              halves. The first halve will serve as the original reference
              batch. The second half will serve as a proxy for the first test
              batch, allowing us to calculate the Hellinger distance statistic.
              When HDDDM is updated with the first actual test batch, HDDDM will
              perform the method for bootstrapping Epsilon, as described in the
              above bullet for "detect_batch" = 2. This will allow a Beta
              threshold to be calculated using the first test batch, allowing
              for detection of drift on this batch. 
          

        Ref. G. Ditzler and R. Polikar, "Hellinger distance based drift detection for nonstationary 
        environments," 2011 IEEE Symposium on Computational Intelligence in Dynamic and Uncertain 
        Environments (CIDUE), 2011, pp. 41-48, doi: 10.1109/CIDUE.2011.5948491.

        Attributes:
            total_samples (int): number of batches the drift detector has ever
                been updated with.
            samples_since_reset (int): number of batches since the last drift detection. 
            drift_state (str): detector's current drift state. Can take values
                "drift", "warning", or None.
            Epsilon (list): stores Epsilon values since the last drift detection. 
            reference_n (int): number of samples in reference batch .
            total_epsilon (int): stores running sum of Epsilon values until drift is detected, 
                initialized to 0.
            bins (int): number of bins in histograms, equivalent to square root of number of 
                samples in reference batch .
            num_feat (int): number of features in reference batch.
            lambda (int): batch number on which last drift was detected.
            distances (dict): For each batch seen (key), stores the Hellinger distance between test and
                reference batch (value). Useful for visualizing drift detection statistics.
            epsilon_values (dict):For each batch seen (key), stores the Epsilon
                value between the current and previous test and reference
                batches (value). Useful for visualizing drift detection
                statistics. Does not store the bootstrapped estimate of Epsilon,
                if used
            thresholds (dict): For each batch seen (key), stores the Beta thresholds between test and
                reference batch (value). Useful for visualizing drift detection statistics.
    """

    def __init__(
        self, 
        reference_batch, 
        statistic = "tstat", 
        significance = 0.05, 
        subsets = 5,
        detect_batch = 1
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
            subsets (int): the number of subsets of reference data to take to compute initial estimate of Epsilon. 
                if too small - initial Epsilon value will be too small. Increases risk of missing drift
                if too high - intial Epsilon value will be too large. Increases
                            risk of false alarms
            detect_batch (int): the test batch on which drift will be detected.
            See class docstrings for more information on this modification.
            Defaults to 1
                if detect_batch = 1 - HDDDM can detect drift on the first test
                            batch passed to the update method
                if detect_batch = 2 - HDDDM can detect drift on the second test
                            batch passed to the update method
                if detect_batch = 3 - HDDDM can detect drift on the third test
                            batch passed to the update method

        Please note: If statistic = 'tstat', need significance to be a decimal
        corresponding to % signifiance (ex: 0.05 for 95%). If statistic =
        'stdev', need significance to be an integer for number of standard
        deviations deemed significant
        """

        super().__init__()

        # Initialize parameters
        self.detect_batch = detect_batch
        self.reference = reference_batch
        self.statistic = statistic
        self.significance = significance
        self.subsets = subsets

        # For visualizations
        self.distances = {}
        self.epsilon_values = {}
        self.thresholds = {}

        # Initial attributes
        self.num_features = self.reference.shape[1]
        self._lambda = 0
        self.reset()

    def update(self, test_batch):
        """Update the detector with a new test batch. If drift is detected, new
        reference batch becomes most recent test batch. If drift is not
        detected, reference batch is updated to include most recent test batch.

        Args:
          test_batch (DataFrame): next batch of data to detect drift on.
        """

        # Ensure batch datasets have equivalent features 
        if not self.num_features == test_batch.shape[1]:
            raise ValueError("Batch features must match")

        if self._drift_state == "drift":
            self.reset()

        super().update()
        test_n = test_batch.shape[0]

        # Estimate histograms
        mins = []
        maxes = []
        for f in range(self.num_features):
            reference_variable = self.reference.iloc[:,f]
            test_variable = test_batch.iloc[:,f]
            mins.append(np.concatenate((reference_variable,test_variable)).min())
            maxes.append(np.concatenate((reference_variable,test_variable)).max())
        self.reference_density = self.build_histograms(self.reference, mins, maxes) 
        test_density = self.build_histograms(test_batch, mins, maxes)

        # Hellinger distances
        self.current_distance, feature_distances = self.hellinger_distance(
            self.reference_density, test_density
        )
        self.distances[self.total_samples] = self.current_distance

        # For each feature, calculate Epsilon, difference in distances
        if self.total_samples > 1:
            self.feature_epsilons = [
                a_i - b_i
                for a_i, b_i in zip(feature_distances, self.prev_feature_distances)
            ]

        # Compute Epsilon and Beta 
        if self.samples_since_reset >= 2:

            if self.samples_since_reset == 2 and self.detect_batch != 3:
                initial_epsilon = self.estimate_initial_epsilon(self.reference, self.subsets, mins, maxes)
                self.epsilon.append(initial_epsilon)
             

            current_epsilon = abs(self.current_distance - self.prev_hellinger_distance)
            self.epsilon.append(current_epsilon)
            self.epsilon_values[self.total_samples]= current_epsilon

            condition1 = bool(self.samples_since_reset >= 2 and self.detect_batch != 3)
            condition2 = bool(self.samples_since_reset >= 3 and self.detect_batch == 3)
            if condition1 or condition2:
                
                self.beta = self.adaptive_threshold(self.statistic, test_n)
                self.thresholds[self.total_samples] = self.beta
            
            
                # Detect drift
                if current_epsilon > self.beta:

                    self.feature_info = {
                        'Epsilons': self.feature_epsilons,
                        'Hellinger_distances': feature_distances,
                        'Significant_drift_in_variable ': self.feature_epsilons.index(max(self.feature_epsilons)),
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
        samples_since_reset (lambda) onwards. Intended for use after drift_state
        == 'drift'. """

        super().reset() 

        if self.detect_batch == 1:
            test_proxy = self.reference.iloc[int(len(self.reference)/2):,]
            self.reference = self.reference.iloc[0:int(len(self.reference)/2),]

        self.reference_n = self.reference.shape[0]
        self.bins = int(np.floor(np.sqrt(self.reference_n)))
        self.epsilon = []
        self.total_epsilon = 0

        if self.detect_batch == 1:
            self.update(test_proxy)


    def build_histograms(self, dataset, min_values, max_values):
        """Computes histogram for each feature in dataset. Bins are equidistantly spaced from minimum
        value to maximum value to ensure exact alignment of bins between test and reference data sets.

        Args:
            dataset (DataFrame): DataFrame on which to estimate density using histograms.
            min_values (list): List of the minimum value for each feature.
            max_values (list): List of the maximum value for each feature.

        Returns:
            List of histograms for each feature. Histograms stored as list of frequency count of data in each bin.

        """

        histograms = [np.histogram(dataset.iloc[:, f], bins=self.bins, range=(min_values[f], max_values[f]))[0] for f in range(self.num_features)]
        
        return histograms

        return histograms

    def hellinger_distance(self, reference_density, test_density):
        """Computes Hellinger distance between reference and test histograms, averaging across features. 

        Args:
            reference_density (list): Output of build_histograms on reference batch. 
            test_density (list): Output of build_histograms on test batch.

        Returns:
            Average Hellinger distance across features. 
            List of individual Hellinger distances for each feature.

        """

        #TODO convert to apply or iteritems if speed is a concern

        feature_distances = []
        total_distance = 0
        for f in range(self.num_features):
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

        return (1/self.num_features)*total_distance, feature_distances


    def adaptive_threshold(self, stat, test_n):
        """Computes adaptive threshold. If computing threshold for third test batch, removes our
         estimate of initial Epsilon from future estimates of epsilon_hat and std.

        Args:
            stat (string): Desired statistical method for computing threshold.
            test_n (integer): Number of samples in test batch.

        Returns:
            Adaptive threshold Beta.
        """

        if self.samples_since_reset == 3 and self.detect_batch != 3:
            self.total_epsilon -= self.epsilon[0]
            self.epsilon = self.epsilon[1:]

        #update scale for denominator (t - lambda - 1), accounting for our initial Epsilon estimate
        if self.samples_since_reset == 2 and self.detect_batch != 3:
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

    def estimate_initial_epsilon(self, reference, num_subsets, histogram_mins, histogram_maxes):
        """ Computes a bootstrapped initial estimate of Epsilon on 2nd test batch, allowing HDDDM 
        to detect drift on the 2nd batch. 
       
        1. Subsets reference data with replacement 
        2. Computes Hellinger distance between each subset.
        3. Computes Epsilon: difference in Hellinger distances.
        4. Averages Epsilon estimates.
        
        Args:
            reference (DataFrame): DataFrame consists of reference batch and first test batch.
            num_subsets (Int): desired number of subsets to be sampled from reference data.
            histogram_mins (List): List of minimum values for each feature align histogram bins.
            histogram_maxes (List): List of maximum values for each feature align histogram bins.

        Returns:
            Bootstrapped estimate of intial Epsilon value.
        """

        # Resampling data
        bootstraps = []
        size = int((1 - (1/num_subsets))*self.reference_n)
        for i in range(num_subsets):
            subset = reference.sample(n = size, replace = True)
            bootstraps.append(self.build_histograms(subset, histogram_mins, histogram_maxes))
        
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
