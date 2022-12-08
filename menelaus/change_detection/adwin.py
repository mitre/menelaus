# most of these get called within loops (or an outer loop on the detector),
# so this is more efficient
from numpy import (
    power,
    log,
    sqrt,
    absolute,
    empty_like,
    zeros,
)

from menelaus.detector import StreamingDetector


class ADWIN(StreamingDetector):
    """ADWIN (ADaptive WINdowing) is a change detection algorithm which uses a
    sliding window to estimate the running mean and variance of a given
    real-valued number.

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
        super().__init__()
        self.delta = delta
        # Although sklearn's standard for estimators is that parameter
        # sanitization occurs with .fit, a detector might be instantiated and
        # then not use a given parameter for quite a while in a pipeline. May
        # need to re-examine this later if user expectations differ from
        # behavior.
        if not 0 <= self.delta <= 1:
            raise ValueError("ADWIN.delta must take values on the range 0 to 1.")
        self.max_buckets = max_buckets
        self.new_sample_thresh = new_sample_thresh
        self.window_size_thresh = window_size_thresh
        self.subwindow_size_thresh = subwindow_size_thresh
        self.conservative_bound = conservative_bound
        self._bucket_row_list = _BucketRowList(self.max_buckets)
        self._curr_total = (
            0  # this attribute is not scaled by the window size; use .mean
        )
        self._curr_variance = (
            0  # this attribute is not scaled by the window size; use .variance
        )
        self._window_size = 0
        self._retraining_recs = [None, None]

    def update(self, X, y_true=None, y_pred=None):
        """Update the detector with a new sample.

        Args:
            X: one row of features from input data.
            y_true: one true label from input data. Not used by ADWIN.
            y_pred: one predicted label from input data. Not used by ADWIN.
        """

        if self.drift_state is not None:
            # note that the other attributes should *not* be initialized after drift
            self.reset()

        X, _, _ = super()._validate_input(X, None, None)
        if len(X.shape) > 1 and X.shape[1] != 1:
            raise ValueError("ADWIN should only be used to monitor 1 variable.")
        super().update(X, None, None)

        # the array should have a single element after validation.
        X = X[0][0]

        # add new sample to the head of the window
        self._window_size += 1
        self._add_sample(X)
        self._shrink_window()

    def reset(self):
        """Initialize the detector's drift state and other relevant attributes.
        Intended for use after ``drift_state == 'drift'``.
        """
        super().reset()
        self._initialize_retraining_recs()

    def _initialize_retraining_recs(self):
        """Sets ``self._retraining_recs`` to ``[None, None]``."""
        self._retraining_recs = [None, None]

    @property
    def retraining_recs(self):
        """Recommended indices for retraining. If drift is detected,
        set to ``[beginning of ADWIN's new window, end of ADWIN's new window]``.
        If these are e.g. the 5th and 13th sample that ADWIN has been updated
        with, the values will be ``[4, 12]``.

        Returns:
            list: the current retraining recommendations
        """
        return self._retraining_recs

    def _add_sample(self, new_value):
        """Make a new bucket containing a single new sample and add it to the
        ``BucketList``. Compress any ``BucketRows`` which have reached maximum size.

        Args:
            new_value: new value to be added to the ``BucketList``.

        """
        # this new bucket should have only one member, so 0 variance.
        # compress_buckets will maintain proper estimates.
        self._bucket_row_list.head.add_bucket(new_value, 0)

        # update running estimates
        if self._window_size > 1:
            self._curr_variance += (
                (self._window_size - 1)
                * (new_value - self._curr_total / (self._window_size - 1))
                * (new_value - self._curr_total / (self._window_size - 1))
                / self._window_size
            )
        self._curr_total += new_value
        self._compress_buckets()

    def _compress_buckets(self):
        """Traverse ``ADWIN.bucket_row_list``, merging the oldest buckets when we
        find that a given ``BucketRow`` is full."""
        curr_bucket_row = self._bucket_row_list.head
        list_position = 0

        while curr_bucket_row is not None:
            # BucketRow is full
            if curr_bucket_row.bucket_count == self.max_buckets + 1:
                next_bucket_row = curr_bucket_row.next_bucket
                if next_bucket_row is None:
                    self._bucket_row_list.append_tail()
                    next_bucket_row = curr_bucket_row.next_bucket

                # take the first two buckets in this row and combine them the
                # number of elements stored in a bucket is 2^(position in the
                # list)
                n_elements = power(2, list_position)
                mean1 = curr_bucket_row.bucket_totals[0] / n_elements
                mean2 = curr_bucket_row.bucket_totals[1] / n_elements
                new_total = (
                    curr_bucket_row.bucket_totals[0] + curr_bucket_row.bucket_totals[1]
                )

                #  Chan, et. al variance calculation doi.org/10.1007/978-3-642-51461-6_3
                new_variance = (
                    curr_bucket_row.bucket_variances[0]
                    + curr_bucket_row.bucket_variances[1]
                    + n_elements * (mean1 - mean2) * (mean1 - mean2) / 2
                    # equivalent: n_elements * n_elements * (mean1 - mean2) * \
                    # (mean1 - mean2) / (n_elements + n_elements)
                )
                next_bucket_row.add_bucket(new_total, new_variance)

                # remove the compressed buckets from the current row
                curr_bucket_row.remove_buckets(2)

                # if the next bucket isn't full, neither can any of the remaining be
                if next_bucket_row.bucket_count <= self.max_buckets:
                    break
            else:
                # similarly, if the first bucket isn't full, neither are any others
                break
            curr_bucket_row = curr_bucket_row.next_bucket
            list_position += 1

    def _shrink_window(self):
        """Check whether all subwindows (the empty set vs. the whole window,
        then the oldest element vs all others, then the oldest two vs all
        others, then ...) satisfy the drift threshold. If not, reduce the window
        size and set the ``drift_state`` to ``"drift"``.
        """
        if (
            self.total_samples % self.new_sample_thresh == 0
            and self._window_size > self.window_size_thresh
        ):
            # either we reduced the window and must restart to check the new
            # subwindows, or this is the initial loop
            start_from_empty_subwindow = True
            while start_from_empty_subwindow:
                start_from_empty_subwindow = False
                exit_shrink = False

                # window0 begins empty, window1 begins with the full set
                n_elements0, n_elements1 = 0, self._window_size
                total0, total1 = 0, self._curr_total

                # traverse the BucketRowList from tail to head
                curr_bucket_row = self._bucket_row_list.tail
                list_pos = (
                    self._bucket_row_list.size - 1
                )  # note that list position begins at 0

                # window0 adds oldest elements first: tail of the BucketRowList,
                # front of the BucketRow arrays
                while (not exit_shrink) and (curr_bucket_row is not None):
                    n_increment = power(2, list_pos)
                    for bucket_index in range(curr_bucket_row.bucket_count):
                        n_elements0 += n_increment
                        n_elements1 -= n_increment
                        total0 += curr_bucket_row.bucket_totals[bucket_index]
                        total1 -= curr_bucket_row.bucket_totals[bucket_index]

                        # reached the youngest element before finding drift;
                        # remember, the buckets have an extra element at the end
                        # to avoid overflow before compression
                        if (list_pos == 0) and (
                            bucket_index == curr_bucket_row.bucket_count - 1
                        ):
                            exit_shrink = True
                            break

                        # check whether to drop elements
                        if (
                            (n_elements0 >= self.subwindow_size_thresh)
                            and (n_elements1 >= self.subwindow_size_thresh)
                            and self._check_epsilon(
                                n_elements0, total0, n_elements1, total1
                            )
                        ):
                            start_from_empty_subwindow = True
                            self.drift_state = "drift"
                            if self._window_size > 0:
                                n_elements0 -= self._remove_last()
                                self._retraining_recs = (
                                    self.total_samples - self._window_size,
                                    self.total_samples - 1,
                                )
                                exit_shrink = True
                                break

                    curr_bucket_row = curr_bucket_row.prev_bucket
                    list_pos -= 1

    def _check_epsilon(self, n_elements0, total0, n_elements1, total1):
        """Calculate ``epsilon_cut`` given the size and totals of two windows
        (equation 3.1 from Bifet 2006). If the difference between the estimated
        mean of the two windows (defined by ``n_elements*``, ``total*``) is greater
        than the calculated threshold, it indicates that drift has occurred.


        Args:
          n_elements0 (int): number of elements in the first window
          total0 (float): running total for elements in the first window
          n_elements1 (int): number of elements in the second window
          total1 (float): running total for elements in the second window

        Returns:
            bool: whether or not the difference between the two windows' means
                exceeds the current threshold defined by ADWIN's variance
                estimate and the chosen delta value.
        """

        window_diff = 1.0 * ((total0 / n_elements0) - (total1 / n_elements1))

        variance = self.variance()
        n_elements = self._window_size

        # note that this is defined as its reciprocal in Bifet
        n_harmonic = 1 / (n_elements0 - self.subwindow_size_thresh + 1) + 1 / (
            n_elements1 - self.subwindow_size_thresh + 1
        )

        if not self.conservative_bound:
            # form below is under normality assumption for 'large' window sizes
            delta_prime_den = log(
                2 * log(n_elements) / self.delta
            )  # noted in Bifet as sufficient in practice vs. delta/n
            eps_cut = (
                sqrt((2 * n_harmonic) * variance * delta_prime_den)
                + 1.0 * (2 / 3) * n_harmonic * delta_prime_den
            )
        else:
            delta_prime_den = log(4 * log(n_elements) / self.delta)
            eps_cut = sqrt(
                (0.5 * n_harmonic) * delta_prime_den
            )  # for "totally rigorous performance guarantees"

        return absolute(window_diff) > eps_cut

    def _remove_last(self):
        """Drop the oldest bucket from the tail of ``bucket_row_list``.

        Returns:
            int: the number of elements removed after discarding the bucket

        """
        curr_bucket_row = self._bucket_row_list.tail
        n_curr = power(2, (self._bucket_row_list.size - 1))
        self._window_size -= n_curr
        self._curr_total -= curr_bucket_row.bucket_totals[0]
        mean_curr = curr_bucket_row.bucket_totals[0] / n_curr
        self._curr_variance -= curr_bucket_row.bucket_variances[
            0
        ] + n_curr * self._window_size * (
            mean_curr - self._curr_total / self._window_size
        ) * (
            mean_curr - self._curr_total / self._window_size
        ) / (
            n_curr + self._window_size
        )
        curr_bucket_row.remove_buckets(1)
        if curr_bucket_row.bucket_count == 0:
            self._bucket_row_list.remove_tail()
        return n_curr

    def mean(self):
        """
        Returns:
            float: the estimated average of the passed stream, using the current window
        """
        if self._window_size == 0:
            out = 0
        else:
            out = self._curr_total / self._window_size
        return out

    def variance(self):
        """
        Returns:
            float: the estimated variance of the passed stream, using the current window
        """
        if self._window_size == 0:
            out = 0
        else:
            out = self._curr_variance / self._window_size
        return out


class _BucketRowList:
    """Doubly-linked list for use by ADWIN. ``max_buckets`` corresponds to the "M"
    parameter from Bifet 2006. At each update step, if the ``BucketRows`` are at
    overflow, their oldest buckets will be moved into the next largest ``BucketRow``
    by ``ADWIN._compress_buckets``. So, the tail of the ``bucket_row_list`` will be the
    estimates corresponding to the oldest elements.

    Note that each ``BucketRow`` only stores estimates related to 2^i elements in
    each bucket: so, each position in ``BucketRowList.head``'s arrays corresponds to
    2^0 = 1 elements; those for the next correspond to 2^1 = 2 elements; etc.
    """

    def __init__(self, max_buckets):
        super().__init__()
        self.max_buckets = max_buckets
        self.head = None
        self.tail = None
        self.size = 0
        self.append_head()

    def append_head(self):
        """Add an empty ``BucketRow`` to the head of the list."""
        new_head = _BucketRow(self.max_buckets, next_bucket=self.head)
        if self.head is not None:
            self.head.prev_bucket = new_head
        self.head = new_head
        if self.tail is None:
            self.tail = self.head
        self.size += 1

    def append_tail(self):
        """Add an empty ``BucketRow`` to the tail of the list."""
        self.tail = _BucketRow(self.max_buckets, prev_bucket=self.tail)
        if self.head is None:  # somehow we've removed all the other buckets
            self.head = self.tail
        self.size += 1

    def remove_tail(self):
        """Remove the last BucketRow from the tail of the BucketRowList."""
        self.tail = self.tail.prev_bucket
        if self.tail is None:
            self.head = None
        else:
            self.tail.next_bucket = None
        self.size -= 1


class _BucketRow:
    """Helper class for ADWIN. A given BucketRow is a single node in ADWIN's
    bucket_row_list doubly linked list. It stores all the buckets of a
    particular size in reverse order: the oldest bucket is at index 0.
    """

    def __init__(self, max_buckets, prev_bucket=None, next_bucket=None):
        """
        Args:
            max_buckets (int): maximum allowed buckets in this row
            prev_bucket (_BucketRow, optional): prior BucketRow in the BucketRowList.
                Defaults to None.
            next_bucket (_BucketRow, optional): next BucketRow in the BucketRowList.
                Defaults to None.

        Attributes:
            bucket_count (int): the number of buckets currently in this row
            bucket_totals (numpy.array): estimated total for each bucket
            bucket_variances (numpy.array): estimated variance for each bucket
        """
        super().__init__()
        self.bucket_count = 0
        self.max_buckets = max_buckets
        self.bucket_totals = zeros(self.max_buckets + 1, dtype=float)
        self.bucket_variances = zeros(self.max_buckets + 1, dtype=float)

        # maintain the linked list
        self.prev_bucket = prev_bucket
        self.next_bucket = next_bucket
        if next_bucket is not None:
            next_bucket.prev_bucket = self
        if self.prev_bucket is not None:
            prev_bucket.next_bucket = self

    def add_bucket(self, total, variance):
        """Add a new bucket to the end of this BucketRow with total and variance.

        Args:
          total:
          variance:
        """
        self.bucket_totals[self.bucket_count] = total
        self.bucket_variances[self.bucket_count] = variance
        self.bucket_count += 1

    def remove_buckets(self, num_buckets):
        """Remove num_buckets (oldest) buckets from the front of the rows's array.

        Args:
          num_buckets: the number of buckets to remove from the front of the row
        """
        self.bucket_totals = self.shift(self.bucket_totals, num_buckets)
        self.bucket_variances = self.shift(self.bucket_variances, num_buckets)
        self.bucket_count -= num_buckets

    @staticmethod
    def shift(arr, num, fill_value=0):
        """Shift arr toward the 0 index by num indices; the last num indices
        will be filled with fill_value.

        Args:
          arr: the array whose elements should be shifted
          num: The number of indices to shift the array forward
          fill_value: The fill value for the now-empty indices, default 0

        Returns:

        """
        num = num * -1
        result = empty_like(arr)
        result[num:] = fill_value
        result[:num] = arr[-num:]
        return result
