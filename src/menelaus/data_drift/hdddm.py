from menelaus.data_drift.histogram_density_method import HistogramDensityMethod


class HDDDM(HistogramDensityMethod):

    """HDDDM is a batch-based, unsupervised drift detection algorithm that
    detects changes in feature distributions. It uses the Hellinger
    distance metric to compare test and reference batches and is capable of
    detecting gradual or abrupt changes in data.

    This method relies upon three statistics:

        * Hellinger distance: the sum of the normalized, squared differences in
          frequency counts for each bin between reference and test datasets,
          averaged across all features.

        * Epsilon: the differences in Hellinger distances between sets
          of reference and test batches.

        * Beta: the adaptive threshold adapted at each time stamp. It is
          based on the mean of Epsilon plus the scaled standard
          deviation of Epsilon. The scale applied to the standard deviation is
          determined by the ``statistic`` parameter. It is either the number of
          standard deviations deemed significant (``"stdev"``) or the t-statistic
          (``"tstat"``).

    HDDDM operates by:

        #. Estimating density functions of reference and test data using
           histograms. The number of bins in each histogram equals the square
           root of the length of reference window. Bins are aligned
           by computing the minimum and maximum value for each feature from both
           test and reference window.

        #. Computing the distance between reference and test
           distributions. The Hellinger distance is first calculated between
           each feature in the reference and test batches. Then, the final
           Hellinger statistic used is the average of each feature's distance.

        #. Computing Epsilon.

        #. Computing the adaptive threshold Beta.

        #. Comparing current Epsilon to Beta. If Epsilon > Beta, drift is
           detected. The new reference batch is now the test batch on which
           drift was detected. All statistics are reset. If Epsilon <= Beta,
           drift is not detected. The reference batch is updated to include this
           most recent test batch. All statistics are maintained.

    Two key modifications, were added to Ditzler and Polikar's presentation of
    HDDDM:

        * To answer the research question of "Where is drift occuring?", it
          stores the distance values and Epsilon values for each
          feature. These statistics can be used to identify and visualize the
          features containing the most significant drifts.

            * The Hellinger distance values are calculated for each feature in
              the test batch. These values can be accessed when drift
              occurs using the ``self.feature_info`` dictionary.

            * The Epsilon values for each feature are stored, for each set of
              reference and test batches. For each feature, these values
              represent the difference in Hellinger distances within the test
              and reference batch at time t, to the Hellinger distances within
              the test and reference batch at time t-1. These can be acccessed
              with each update call using the ``self.feature_epsilons`` variable.
              They also can be accessed when drift occurs using the
              ``self.feature_info`` dictionary.

        * The original algorithm cannot detect drift until it is updated with
          the third 3rd test batch after initial setup or 3rd test batch after
          each drift detection. When HDDDM is updated with the first test batch
          (T=1), the distance is calculated between test and reference batch.
          When updated with the second test batch (T=2), HDDDM calculates the
          distance and Epsilon, the difference in distances. When HDDDM is
          updated with the third test batch (T=3), the distance and Epsilon are
          calculated. The adaptive threshold Beta is computed using Epsilon
          values from lambda -> T-1 (lambda is either T=0 or the test batch
          number on which drift was last detected). Beta needs at least one
          prior value of Epsilon to be calculated. Now that Beta can be
          calculated, HDDDM can detect drift on the 3rd test batch. In order to
          allow for more immediate detection of drift, we added the following
          options, specified through the parameter ``detect_batch``:

            * if ``detect_batch`` = 3, HDDDM will operate as described above.

            * if ``detect_batch`` = 2, HDDDM will detect drift on the second test
              batch. On the second test batch only, HDDDM uses a bootstrapped
              estimate of an initial Epsilon value, to serve as a proxy for the
              first value of Epsilon occurring at T-1. This initial estimate of
              Epsilon is used to calculate Beta, allowing to allow for drift to
              be detected on the second test batch. On the third test batch,
              this value is removed from all proceeeding Epsilon and Beta
              calculations.

            * if ``detect_batch`` = 1, HDDDM will detect drift on the first test
              batch. The initial reference batch is split randomly into two
              halves. The first halve will serve as the original reference
              batch. The second half will serve as a proxy for the first test
              batch, allowing us to calculate the distance statistic. When HDM
              is updated with the first actual test batch, HDM will perform the
              method for bootstrapping Epsilon, as described in the above bullet
              for ``detect_batch`` = 2. This will allow a Beta threshold to be
              calculated using the first test batch, allowing for detection of
              drift on this batch.

    Ref. :cite:t:`ditzler2011hellinger`

    Attributes:
        total_updates (int): total number of batches the drift detector has
            been updated with. If detect_batch = 1, attr refers to total
            number of batches + 1 to account for additional update call
            due to the initial splitting of the reference batch.
        updates_since_reset (int): number of batches since the last drift
            detection. If detect_batch = 1, attr refers to total
            number of batches + 1 to account for additional update call
            due to the initial splitting of the reference batch.
        drift_state (str): detector's current drift state. Can take values
            ``"drift"`` or ``None``.
        Epsilon (list): stores Epsilon values since the last drift detection.
        reference_n (int): number of samples in reference batch.
        total_epsilon (int): stores running sum of Epsilon values until drift is detected,
            initialized to 0.
        bins (int): number of bins in histograms, equivalent to square root of number of
            samples in reference batch.
        num_feat (int): number of features in reference batch.
        lambda (int): batch number on which last drift was detected.
        distances (dict): For each batch seen (key), stores the Hellinger distance between test and
            reference batch (value). Useful for visualizing drift detection statistics.
        epsilon_values (dict):For each batch seen (key), stores the Epsilon
            value between the current and previous test and reference
            batches (value). Useful for visualizing drift detection
            statistics. Does not store the bootstrapped estimate of Epsilon,
            if used.
        thresholds (dict): For each batch seen (key), stores the Beta thresholds between test and
            reference batch (value). Useful for visualizing drift detection statistics.
    """

    input_type = "batch"

    def __init__(
        self,
        detect_batch=1,
        divergence="H",
        statistic="tstat",
        significance=0.05,
        subsets=5,
    ):

        """
        Args:
            divergence (str): divergence measure used to compute distance
                between histograms. Default is "H".

                * "H"  - Hellinger distance, original use is for HDDDM

                * "KL" - Kullback-Leibler Divergence, original use is for CDBD

                * User can pass in custom divergence function. Input is two
                  two-dimensional arrays containing univariate histogram
                  estimates of density, one for reference, one for test. It
                  must return the distance value between histograms. To be
                  a valid distance metric, it must satisfy the following
                  properties: non-negativity, identity, symmetry,
                  and triangle inequality, e.g. that in
                  examples/hdddm_example.py.

            detect_batch (int): the test batch on which drift will be detected.
              See class docstrings for more information on this modification.
              Defaults to 1.

                * if detect_batch = 1 - HDDDM can detect drift on the first test
                  batch passed to the update method. BE AWARE, total samples
                  and samples since reset will be number of batches passed to HDM
                  plus 1, due to splitting of reference batch

                * if detect_batch = 2 - HDDDM can detect drift on the second test
                  batch passed to the update method.

                * if detect_batch = 3 - HDDDM can detect drift on the third test
                  batch passed to the update method.

            statistic (str): statistical method used to compute adaptive
                threshold. Defaults to ``"tstat"``.

                * ``"tstat"`` - t-statistic with desired significance level and
                  degrees of freedom = 2 for hypothesis testing on two
                  populations.

                * ``"stdev"`` - uses number of standard deviations deemed
                  significant to compute threhsold.

            significance (float): statistical significance used to identify
                adaptive threshold. Defaults to 0.05.

                * if statistic = ``"tstat"`` - statistical significance of
                  t-statistic, e.g. .05 for 95% significance level.

                * if statistic = ``"stdev"`` - number of standard deviations of
                  change around the mean accepted.

            subsets (int): the number of subsets of reference data to take to
                compute initial estimate of Epsilon.

                * if too small - initial Epsilon value will be too small.
                  Increases risk of missing drift.

                * if too high - intial Epsilon value will be too large.
                  Increases risk of false alarms.

        """

        super().__init__(
            divergence=divergence,
            detect_batch=detect_batch,
            statistic=statistic,
            significance=significance,
            subsets=subsets,
        )

    def set_reference(self, reference_batch):
        """
        Initialize detector with a reference batch. After drift, reference batch is
        automatically set to most recent test batch. Option for user to specify
        alternative reference batch using this method.

        Args:
            reference_batch (DataFrame): initial baseline dataset
        """

        super().set_reference(reference_batch)

    def update(self, test_batch):
        """
        Update the detector with a new test batch. If drift is detected, new
        reference batch becomes most recent test batch. If drift is not
        detected, reference batch is updated to include most recent test batch.

        Args:
          test_batch (DataFrame): next batch of data to detect drift on.
        """

        # Ensure batch datasets have equivalent features
        if not self._num_features == test_batch.shape[1]:
            raise ValueError("Batch features must match")

        super().update(test_batch)

    def reset(self):
        """
        Initialize relevant attributes to original values, to ensure information
        only stored from updates_since_reset (lambda) onwards. Intended for use
        after ``drift_state == 'drift'``.
        """
        # This is here to make sphinx behave.
        super().reset()
