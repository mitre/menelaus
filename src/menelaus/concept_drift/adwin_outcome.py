from menelaus.change_detection.adwin import ADWIN


class ADWINOutcome(ADWIN):
    """ADWIN (ADaptive WINdowing) is a drift detection algorithm which uses a
    sliding window to estimate the running mean and variance of a given
    real-valued number. ADWINOutcome specifically expects ``y_true``, ``y_pred``, and
    uses that input to monitor the running accuracy of a classifier.

    As each sample is added, ADWIN stores a running estimate (mean and variance)
    for a given statistic, calculated over a sliding window which will grow to
    the right until drift is detected. The condition for drift is defined over
    pairs of subwindows at certain cutpoints within the current window. If, for
    any such pair, the difference between the running estimates of the statistic
    is over a certain threshold (controlled by delta), we identify drift, and
    remove the oldest elements of the window until all differences are again
    below the threshold.

    The running estimates in each subwindow are maintained by storing summaries
    of the elements in "buckets," which, in this implementation, are themselves
    stored in the ``bucket_row_list`` attribute, whose total size scales with the
    ``max_buckets`` parameter.

    When drift occurs, the index of the element at the beginning of ADWIN's new
    window is stored in ``self.retraining_recs``.

    Ref. :cite:t:`bifet2007learning`

    Attributes:
        total_samples (int): number of samples the drift detector has ever
            been updated with
        samples_since_reset (int): number of samples since the last time the
            drift detector was reset

    """

    def __init__(
        self,
        delta=0.002,
        max_buckets=5,
        new_sample_thresh=32,
        window_size_thresh=10,
        subwindow_size_thresh=5,
        conservative_bound=False,
    ):
        """

        Args:
            delta (float, optional): confidence value on on 0 to 1. ADWIN will
                incorrectly detect drift with at most probability ``delta``, and
                correctly detect drift with at least probability ``1 - delta``.
                Defaults to 0.002.
            max_buckets (int, optional): the maximum number of buckets to
                maintain in each BucketRow. Corresponds to the "M" parameter in
                Bifet 2006. Defaults to 5.
            new_sample_thresh (int, optional): the drift detection procedure
                will run every ``new_sample_thresh samples``, not in between.
                Defaults to 32.
            window_size_thresh (int, optional): the minimum number of samples in
                the window required to check for drift. Defaults to 10.
            subwindow_size_thresh (int, optional): the minimum number of samples
                in each subwindow reqired to check it for drift. Defaults to 5.
            conservative_bound (bool, optional): whether to assume a 'large
                enough' sample when constructing drift cutoff. Defaults to ``False``.

        Raises:
            ValueError: If ``ADWIN.delta`` is not on the range 0 to 1.
        """
        super().__init__(
            delta=0.002,
            max_buckets=5,
            new_sample_thresh=32,
            window_size_thresh=10,
            subwindow_size_thresh=5,
            conservative_bound=False,
        )

    def update(self, y_true, y_pred, X=None):
        """Update the detector with a new sample.

        Args:
          y_true: actual class of next sample
          y_pred: predicted class of next sample
          X: next sample in the stream of data. Not used for this
            accuracy-based ADWIN. See ``change_detection.ADWIN`` for that
            application.
        """

        new_value = int(y_true == y_pred)

        # This class is here to avoid asking the user to provide such a direct
        # function of (y_true, y_pred) in the X argument, which is unintuitive.
        super().update(new_value, y_true=None, y_pred=None)
