import copy
import numpy as np
from numpy import unique
import pandas as pd
from pandas import DataFrame
import scipy.stats

from menelaus.detector import StreamingDetector, BatchDetector
from menelaus.partitioners.KDQTreePartitioner import KDQTreePartitioner


class KdqTreeDetector:
    """
    Parent class for kdqTree-based drift detector classes. Whether
    reliant on streaming or batch data, kdqTree detectors have some
    common attributes, logic, and functions.

    kdqTree is a drift detection algorithm which detects drift via the
    Kullback-Leibler divergence, calculated after partitioning the data space
    via constructing a k-d-quad-tree (kdq-tree). A reference window of initial
    data is compared to a test window of later data. The Kullback-Leibler
    divergence between the empirical distributions of the reference and test
    windows is calculated, and drift is alarmed when a threshold is reached.
    A kdqtree is a combination of k-d trees and quad-trees; it is a binary tree
    (k-d) whose nodes contain square cells (quad) which are created via
    sequential splits along each dimension. This structure allows the
    calculation of the K-L divergence for continuous distributions, as the K-L
    divergence is defined on probability mass functions. The number of samples
    in each leaf of the tree is an empirical distribution for either dataset;
    this allows us to calculate the K-L divergence.

    If used in a streaming data setting, the reference window is used to
    construct a kdq-tree, and the data in both the reference and test window are
    filed into it. If used in a batch data setting, the reference window - the
    first batch passed in - is used to construct a kdq-tree, and data in test
    batches are compared to it. When drift is detected on a test batch, that
    test batch is set to be the new reference window - unless the user specifies
    a reference window using the set_reference method.

    The threshold for drift is determined using the desired alpha level by a
    bootstrap estimate for the critical value of the K-L divergence, drawing a
    sample of ``num_boostrap_samples`` repeatedly, ``2 * window_size`` times,
    from the reference window.

    Additionally, the Kulldorff spatial scan statistic, which is a special case
    of the KL-divergence, can be calculated at each node of the kdq-tree, which
    gives a measure of the regions of the data space which have the greatest
    divergence between the reference and test windows. This can be used to
    visualize which regions of data space have the greatest drift. Note that
    these statistics are specific to the partitions of the data space by the
    kdq-tree, rather than (necessarily) the maximally different region in
    general. KSSs are made available via ``to_plotly_dataframe``, which produces
    output structured for use with ``plotly.express.treemap``.

    Note that this algorithm could be used with other types of trees; the
    reference paper and this implementation use kdq-trees.

    Note that the current implementation does not explicitly handle categorical
    data.

    Ref. :cite:t:`dasu2006information`
    """

    def __init__(
        self,
        alpha=0.01,
        bootstrap_samples=500,
        count_ubound=100,
        cutpoint_proportion_lbound=2e-10,
    ):
        """
        Args:
            alpha (float, optional): Achievable significance level. Defaults to
                0.01.
            bootstrap_samples (int, optional): The number of bootstrap samples
                to use to approximate the empirical distributions. Equivalent to
                kappa in Dasu (2006), which recommends 500-1000 samples.
                Defaults to 500.
            count_ubound (int, optional): An upper bound for the number of
                samples stored in a leaf node of the kdqTree. No leaf shall
                contain more samples than this value, unless further divisions
                violate the cutpoint_proportion_lbound restriction. Default 100.
            cutpoint_proportion_lbound (float, optional): A lower bound for the
                size of the leaf nodes. No node shall have a size length smaller
                than this proportion, relative to the original feature length.
                Defaults to 2e-10.
        """
        self.alpha = alpha
        self.bootstrap_samples = bootstrap_samples
        self.count_ubound = count_ubound
        self.cutpoint_proportion_lbound = cutpoint_proportion_lbound

    def reset(self):
        """
        Initialize the detector's drift state and other relevant attributes.
        Intended for use after ``drift_state == "drift"`` or initialization.
        """
        self._ref_data = np.array([])
        self._test_data_size = 0
        self._kdqtree = None
        self._critical_dist = None

    def _evaluate_kdqtree(self, ary, input_type):
        """
        Deploy internal KDQTreePartitioner and evaluate results on
        new data, setting drift state accordingly.

        Args:
            ary (numpy.ndarray): new data to be evaluated
            input_type (str): context for incoming data, either
                ``'stream'`` or ``'batch'``
        """
        # XXX - This is one final spot where we are still technically
        #       'branching' on input_type. However, in practice, this
        #       decomposes the update function so much that the two
        #       classes always call this with their respective input
        #       type - so it may not be much of an issue? @Anmol-Srivastava
        if self._kdqtree is None:  # ary part of new ref tree
            self._ref_data = (
                np.vstack([self._ref_data, ary]) if self._ref_data.size else ary
            )

            if (
                input_type == "stream" and len(self._ref_data) == self.window_size
            ) or input_type == "batch":
                # XXX - _inner_set_reference calls KLD itself, so needs to know sample_size
                #       and hence input type; we may streamline this later - Anmol-Srivastava
                self._inner_set_reference(self._ref_data, input_type)

        else:  # new test sample(s)
            self._kdqtree.fill(ary, tree_id="test", reset=(input_type == "batch"))
            if input_type == "stream":
                self._test_data_size += 1
            # check for drift if either: we're streaming and the reference
            # window is full, or we're doing batch detection
            if input_type == "batch" or (self._test_data_size >= self.window_size):
                test_dist = self._kdqtree.kl_distance(tree_id1="build", tree_id2="test")
                if test_dist > self._critical_dist:
                    if input_type == "stream":
                        self._drift_counter += 1
                        if self._drift_counter > self.persistence * self.window_size:
                            self.drift_state = "drift"
                    else:
                        self.drift_state = "drift"
                        self.ref_data = ary

    def _inner_set_reference(self, ary, input_type):
        """
        Initialize detector with a reference batch. This does not perform
        validation, and is therefore not intended to be called directly by the
        user.

        Args:
            ary (numpy.ndarray): baseline dataset
            input_type (str): context for incoming data, either
                ``"stream"`` or ``"batch"``
        """
        # TODO ensure self.reset (or e.g. any function in that place uses right local version)
        self.reset()
        self._kdqtree = KDQTreePartitioner(
            count_ubound=self.count_ubound,
            cutpoint_proportion_lbound=self.cutpoint_proportion_lbound,
        )
        self._kdqtree.build(ary)
        ref_counts = self._kdqtree.leaf_counts("build")
        sample_size = self.window_size if input_type == "stream" else sum(ref_counts)
        self._critical_dist = self._get_critical_kld(ref_counts, sample_size)
        if input_type == "stream":
            self._ref_data = np.array([])

    def to_plotly_dataframe(
        self, tree_id1="build", tree_id2="test", max_depth=None, input_cols=None
    ):
        """
        Generates a dataframe containing information about the kdqTree's structure
        and some node characteristics, intended for use with ``plotly``.

        Args:
            tree_id1 (str, optional): Reference tree. If ``tree_id2`` is not
                specified, the only tree described. Defaults to ``"build"``.
            tree_id2 (str, optional): Test tree. If this is specified, the
                dataframe will also contain information about the difference
                between counts in each node for the reference vs. the test tree.
                Defaults to ``"test"``.
            max_depth (int, optional): Depth in the tree to which to recurse.
                Defaults to ``None``.
            input_cols (list, optional): List of column names for the input
                data. Defaults to ``None``.
        Returns:
            pd.DataFrame: A dataframe where each row corresponds to a node, and
            each column contains some information:

                * ``name``: a label corresponding to which feature this split is on
                * ``idx``: a unique ID for the node, to pass
                  ``plotly.express.treemap``'s id argument
                * ``parent_idx``: the ID of the node's parent
                * ``cell_count``: how many samples are in this node in the
                  reference tree.
                * ``depth``: how deep the node is in the tree
                * ``count_diff``: if ``tree_id2`` is specified, the change in
                  counts from the reference tree.
                * ``kss``: the Kulldorff Spatial Scan Statistic for this node,
                  defined as the Kullback-Leibler divergence for this node
                  between the reference and test trees, using the individual
                  node and all other nodes combined as the bins for the
                  distributions.
        """
        if input_cols is not None:
            return self._kdqtree.to_plotly_dataframe(
                tree_id1, tree_id2, max_depth, input_cols
            )
        else:
            return self._kdqtree.to_plotly_dataframe(
                tree_id1, tree_id2, max_depth, self._input_cols
            )

    def _get_critical_kld(self, ref_counts, sample_size):
        """
        Find the critical value of the Kullback-Leibler divergence, using the
        empirical distribution defined by the counts of the reference tree
        across the leaves.

        Args:
            ref_counts (numpy array): The counts from the reference tree across
                its leaves.
            sample_size (int): Size of reference data. Has slightly different
                contexts in streaming and batch cases.

        Returns:
            float: the critical distance corresponding to ``self.alpha``, the
                achievable significance level.
        """
        ref_dist = KDQTreePartitioner._distn_from_counts(ref_counts)
        b_dist_pairs = []
        bin_indices = list(range(len(ref_counts)))
        bin_indices_df = DataFrame({"leaf": bin_indices})

        for _ in range(self.bootstrap_samples):
            # note the maintenance of the leaf order!
            b_sample = np.random.choice(bin_indices, size=2 * sample_size, p=ref_dist)
            b_hist1 = unique(b_sample[:sample_size], return_counts=True)
            b_hist2 = unique(b_sample[sample_size:], return_counts=True)
            b_hist1 = DataFrame({"leaf": b_hist1[0], "count": b_hist1[1]})
            b_hist2 = DataFrame({"leaf": b_hist2[0], "count": b_hist2[1]})
            b_hist1 = (
                b_hist1.merge(bin_indices_df, on="leaf", how="outer")
                .fillna(0)
                .sort_values(by="leaf")
            )
            b_hist2 = (
                b_hist2.merge(bin_indices_df, on="leaf", how="outer")
                .fillna(0)
                .sort_values(by="leaf")
            )
            b_hist1 = KDQTreePartitioner._distn_from_counts(b_hist1["count"])
            b_hist2 = KDQTreePartitioner._distn_from_counts(b_hist2["count"])
            b_dist_pairs.append([b_hist1, b_hist2])

        critical_distances = [scipy.stats.entropy(a, b) for a, b in b_dist_pairs]
        return np.quantile(critical_distances, 1 - self.alpha, method="nearest")


class KdqTreeStreaming(KdqTreeDetector, StreamingDetector):
    """
    Implements the kdqTree drift detection algorithm in a
    streaming data context. Inherits from ``KdqTreeDetector``
    and ``StreamingDetector`` (see docs).

    kdqTree is a drift detection algorithm which detects drift via the
    Kullback-Leibler divergence, calculated after partitioning the data space
    via constructing a k-d-quad-tree (kdq-tree).

    If used in a streaming data setting, the reference window is used to
    construct a kdq-tree, and the data in both the reference and test window are
    filed into it. If used in a batch data setting, the reference window - the
    first batch passed in - is used to construct a kdq-tree, and data in test
    batches are compared to it. When drift is detected on a test batch, that
    test batch is set to be the new reference window - unless the user specifies
    a reference window using the set_reference method.

    The threshold for drift is determined using the desired alpha level by a
    bootstrap estimate for the critical value of the K-L divergence, drawing a
    sample of ``num_boostrap_samples`` repeatedly, ``2 * window_size`` times,
    from the reference window.

    Additionally, the Kulldorff spatial scan statistic, which is a special case
    of the KL-divergence, can be calculated at each node of the kdq-tree, which
    gives a measure of the regions of the data space which have the greatest
    divergence between the reference and test windows. This can be used to
    visualize which regions of data space have the greatest drift. Note that
    these statistics are specific to the partitions of the data space by the
    kdq-tree, rather than (necessarily) the maximally different region in
    general. KSSs are made available via ``to_plotly_dataframe``, which produces
    output structured for use with ``plotly.express.treemap``.

    Ref. :cite:t:`dasu2006information`
    """

    def __init__(
        self,
        window_size,
        persistence=0.05,
        alpha=0.01,
        bootstrap_samples=500,
        count_ubound=100,
        cutpoint_proportion_lbound=2e-10,
    ):
        """
        Args:
            window_size (int): The minimum number of samples required
                to test whether drift has occurred.
            persistence (float, optional): Persistence factor: how many samples
                in a row, as a proportion of the window size, must be in the
                "drift region" of K-L divergence, in order for ``kdqTree`` to
                alarm and reset. Defaults to 0.05.
            alpha (float, optional): Achievable significance level. Defaults to
                0.01.
            bootstrap_samples (int, optional): The number of bootstrap samples
                to use to approximate the empirical distributions. Equivalent to
                kappa in Dasu (2006), which recommends 500-1000 samples.
                Defaults to 500.
            count_ubound (int, optional): An upper bound for the number of
                samples stored in a leaf node of the kdqTree. No leaf shall
                contain more samples than this value, unless further divisions
                violate the cutpoint_proportion_lbound restriction. Default 100.
            cutpoint_proportion_lbound (float, optional): A lower bound for the
                size of the leaf nodes. No node shall have a size length smaller
                than this proportion, relative to the original feature length.
                Defaults to 2e-10.
        """
        if not isinstance(window_size, int) or window_size < 1:
            raise ValueError(f"window_size must be positive integer, was {window_size}")

        StreamingDetector.__init__(self)
        KdqTreeDetector.__init__(
            self, alpha, bootstrap_samples, count_ubound, cutpoint_proportion_lbound
        )
        self.window_size = (
            window_size  # XXX - could be generalized into StreamingDetector.init
        )
        self.persistence = persistence
        self.reset()

    def reset(self):
        """
        Initialize the detector's drift state and other relevant attributes.
        Intended for use after ``drift_state == "drift"`` or initialization.
        """
        StreamingDetector.reset(self)
        KdqTreeDetector.reset(self)
        self._drift_counter = 0  # samples consecutively in the drift region

    def update(self, X, y_true=None, y_pred=None):
        """
        Update the detector with a new sample point. Constructs the reference
        data's kdqtree; then, when sufficient samples have been received, puts
        the test data into the same tree; then, checks divergence between the
        reference and test data.

        The reference window is maintained as the initial window until drift.
        Upon drift, the user may continue passing data to update and new
        reference windows will be constructed once sufficient samples are received.

        Args:
            X (pandas.DataFrame or numpy array): If just reset/initialized,
            the reference data. Otherwise, a new sample to put into the test
            window.
            y_true (numpy.ndarray): true labels of input data - not used in KdqTree
            y_pred (numpy.ndarray): predicted labels of input data - not used in KdqTree
        """
        if self.drift_state == "drift":
            self.reset()

        X, _, _ = super()._validate_input(X, None, None)
        StreamingDetector.update(self, X, None, None)
        ary = copy.deepcopy(X)
        KdqTreeDetector._evaluate_kdqtree(self, ary, "stream")


class KdqTreeBatch(KdqTreeDetector, BatchDetector):
    """
    Implements the kdqTree drift detection algorithm in a
    batch data context. Inherits from ``KdqTreeDetector``
    and ``BatchDetector`` (see docs).

    kdqTree is a drift detection algorithm which detects drift via the
    Kullback-Leibler divergence, calculated after partitioning the data space
    via constructing a k-d-quad-tree (kdq-tree). A reference window of initial
    data is compared to a test window of later data. The Kullback-Leibler
    divergence between the empirical distributions of the reference and test
    windows is calculated, and drift is alarmed when a threshold is reached.
    A kdqtree is a combination of k-d trees and quad-trees; it is a binary tree
    (k-d) whose nodes contain square cells (quad) which are created via
    sequential splits along each dimension. This structure allows the
    calculation of the K-L divergence for continuous distributions, as the K-L
    divergence is defined on probability mass functions. The number of samples
    in each leaf of the tree is an empirical distribution for either dataset;
    this allows us to calculate the K-L divergence.

    If used in a streaming data setting, the reference window is used to
    construct a kdq-tree, and the data in both the reference and test window are
    filed into it. If used in a batch data setting, the reference window - the
    first batch passed in - is used to construct a kdq-tree, and data in test
    batches are compared to it. When drift is detected on a test batch, that
    test batch is set to be the new reference window - unless the user specifies
    a reference window using the set_reference method.

    The threshold for drift is determined using the desired alpha level by a
    bootstrap estimate for the critical value of the K-L divergence, drawing a
    sample of ``num_boostrap_samples`` repeatedly, ``2 * window_size`` times,
    from the reference window.

    Additionally, the Kulldorff spatial scan statistic, which is a special case
    of the KL-divergence, can be calculated at each node of the kdq-tree, which
    gives a measure of the regions of the data space which have the greatest
    divergence between the reference and test windows. This can be used to
    visualize which regions of data space have the greatest drift. Note that
    these statistics are specific to the partitions of the data space by the
    kdq-tree, rather than (necessarily) the maximally different region in
    general. KSSs are made available via ``to_plotly_dataframe``, which produces
    output structured for use with ``plotly.express.treemap``.

    Ref. :cite:t:`dasu2006information`
    """

    def __init__(
        self,
        alpha=0.01,
        bootstrap_samples=500,
        count_ubound=100,
        cutpoint_proportion_lbound=2e-10,
    ):
        """
        Args:
            alpha (float, optional): Achievable significance level. Defaults to
                0.01.
            bootstrap_samples (int, optional): The number of bootstrap samples
                to use to approximate the empirical distributions. Equivalent to
                kappa in Dasu (2006), which recommends 500-1000 samples.
                Defaults to 500.
            count_ubound (int, optional): An upper bound for the number of
                samples stored in a leaf node of the kdqTree. No leaf shall
                contain more samples than this value, unless further divisions
                violate the cutpoint_proportion_lbound restriction. Default 100.
            cutpoint_proportion_lbound (float, optional): A lower bound for the
                size of the leaf nodes. No node shall have a size length smaller
                than this proportion, relative to the original feature length.
                Defaults to 2e-10.
        """
        BatchDetector.__init__(self)
        KdqTreeDetector.__init__(
            self, alpha, bootstrap_samples, count_ubound, cutpoint_proportion_lbound
        )
        self.reset()

    def reset(self):
        """
        Initialize the detector's drift state and other relevant attributes.
        Intended for use after ``drift_state == "drift"`` or initialization.
        """
        BatchDetector.reset(self)
        KdqTreeDetector.reset(self)

    def set_reference(self, X, y_true=None, y_pred=None):
        """
        Initialize detector with a reference batch. The user may specify an
        alternate reference batch than the one maintained by kdq-Tree. This will
        reset the detector.

        Args:
            X (pandas.DataFrame or numpy.array): baseline dataset
            y_true (numpy.array): actual labels of dataset - not used in KdqTree
            y_pred (numpy.array): predicted labels of dataset - not used in KdqTree
        """
        X, _, _ = super()._validate_input(X, None, None)
        ary = copy.deepcopy(X)
        self._inner_set_reference(ary, input_type="batch")

    def update(self, X, y_true=None, y_pred=None):
        """
        Update the detector with a new batch. Constructs the reference
        data's kdqtree; then, when sufficient samples have been received, puts
        the test data into the same tree; then, checks divergence between the
        reference and test data.

        The initial batch will be used as the reference at each update step,
        regardless of drift state.  If the user wishes to change reference
        batch, use the ``set_reference`` method and then continue passing new
        batches to ``update``.

        Args:
            X (pandas.DataFrame or numpy array): If just reset/initialized,
            the reference data. Otherwise, a new batch of data to be compared
            to the reference window.
            y_true (numpy.ndarray): true labels of input data - not used in KdqTree
            y_pred (numpy.ndarray): predicted labels of input data - not used in KdqTree
        """

        if self.drift_state == "drift":
            # Note that set_reference resets the detector.
            self.set_reference(self.ref_data)

        X, _, _ = super()._validate_input(X, None, None)
        BatchDetector.update(self, X, None, None)
        ary = copy.deepcopy(X)

        KdqTreeDetector._evaluate_kdqtree(self, ary, "batch")
