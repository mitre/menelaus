from molten.data_drift.histogram_density_method import HistogramDensityMethod


class CDBD(HistogramDensityMethod):

    """
    The Confidence Distribution Batch Detection (CDBD) algorithm is a
    statistical test that seeks to detect concept drift in classifiers,
    without the use of labeled data. It is intended to monitor a classifier's
    confidence scores but could be substituted with any univariate performance
    related statistic obtained from a learner., e.g. posterior probabilities.

    This method relies upon three statistics:

        * KL divergence: the Kullback-Leibler Divergence (KLD) measure

        * Epsilon: the differences in Hellinger distances between sets
          of reference and test batches.

        * Beta: the adaptive threshold adapted at each time stamp. It is based
          on the mean of Epsilon plus the scaled standard deviation of Epsilon.
          The scale applied to the standard deviation is determined by the
          ``statistic`` parameter. It is either the number of standard deviations
          deemed significant (``"stdev"``) or the t-statistic (``"tstat"``).

    CDBD operates by:

        #. Estimating density functions of reference and test data using
           histograms. The number of bins in each histogram equals the square
           root of the length of reference window. Bins are aligned by computing
           the minimum and maximum value for each feature from both test and
           reference window.

        #. Computing the distance between reference and test distributions. The
           KL divergence metric is used to calculate the distance between
           univariate histograms.

        #. Computing Epsilon.

        #. Computing the adaptive threshold Beta.

        #. Comparing current Epsilon to Beta. If Epsilon > Beta, drift is
           detected. The new reference batch is now the test batch on which
           drift was detected. All statistics are reset. If Epsilon <= Beta,
           drift is not detected. The reference batch is updated to include this
           most recent test batch. All statistics are maintained.


    Attributes:
            total_samples (int): number of batches the drift detector has ever
                been updated with.
            samples_since_reset (int): number of batches since the last drift
                detection. drift_state (str): detector's current drift state.
                Can take values ``"drift"``, ``"warning"``, or ``None``.
            Epsilon (list): stores Epsilon values since the last drift detection.
            reference_n (int): number of samples in reference batch.
            total_epsilon (int): stores running sum of Epsilon values until
                drift is detected, initialized to 0.
            bins (int): number of bins in histograms, equivalent to square root
                of number of samples in reference batch .
            num_feat (int): number of features in reference batch.
            lambda (int): batch number on which last drift was detected.
            distances (dict): For each batch seen (key), stores the distance
                between test and reference batch (value). Useful for visualizing
                drift detection statistics.
            epsilon_values (dict): For each batch seen (key), stores the Epsilon
                value between the current and previous test and reference
                batches (value). Useful for visualizing drift detection
                statistics. Does not store the bootstrapped estimate of Epsilon,
                if used.
            thresholds (dict): For each batch seen (key), stores the Beta
                thresholds between test and reference batch (value). Useful for
                visualizing drift detection statistics.

    Ref. Lindstrom, P., Mac Namee, B., Delany, S. J., 2013. Drift detection
    using uncertainty distribution divergence. Evolving Systems 4 (1), 13–25.

    """

    _input_type = "batch"

    def __init__(
        self,
        reference_batch,
        detect_batch=1,
        statistic="tstat",
        significance=0.05,
        subsets=5,
    ):

        """
        Args:
            reference_batch (DataFrame): initial baseline dataset
            detect_batch (int): the test batch on which drift will be detected.
                See class docstrings for more information on this modification.
                Defaults to 1.

                * if detect_batch = 1 - HDDDM can detect drift on the first test
                  batch passed to the update method.

                * if detect_batch = 2 - HDDDM can detect drift on the second test
                  batch passed to the update method.

                * if detect_batch = 3 - HDDDM can detect drift on the third test
                  batch passed to the update method.

            statistic (str): statistical method used to compute adaptive threshold. Defaults to ``"tstat"``

                * ``"tstat"`` - t-statistic with desired significance level and
                  degrees of freedom = 2 for hypothesis testing on two
                  populations.

                * ``"stdev"`` - uses number of standard deviations deemed
                  significant to compute threshold.

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
            reference_batch=reference_batch,
            divergence="KL",
            detect_batch=detect_batch,
            statistic=statistic,
            significance=significance,
            subsets=subsets,
        )

        # Ensure only being used with 1 variable in reference
        if self._num_features != 1:
            raise ValueError("CDBD should only be used to monitor 1 variable")

    def update(self, test_batch):
        """Update the detector with a new test batch. If drift is detected, new
        reference batch becomes most recent test batch. If drift is not
        detected, reference batch is updated to include most recent test batch.

        Args:
          test_batch (DataFrame): next batch of data to detect drift on.
        """

        # Ensure only being used with 1 variable in test
        if test_batch.shape[1] != 1:
            raise ValueError("CDBD should only be used to monitor 1 variable")

        super().update(test_batch)

    def reset(self):
        """
        Initialize relevant attributes to original values, to ensure information
        only stored from samples_since_reset (lambda) onwards. Intended for use
        after ``drift_state == 'drift'``.
        """
        # This is here to make sphinx behave.
        super().reset()