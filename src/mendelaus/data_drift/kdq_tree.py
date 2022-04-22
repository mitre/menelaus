import pandas as pd
import numpy as np
import scipy.stats
from mendelaus.drift_detector import DriftDetector
from mendelaus.partitioners.KDQTreePartitioner import KDQTreePartitioner


class KdqTree(DriftDetector):
    """kdqTree is a drift detection algorithm which detects drift via the
    Kullback-Leibler divergence, calculated after partitioning the data space
    via constructing a k-d-quad-tree (kdq-tree). A reference window of initial
    data is compared to a test window of later data. The Kullback-Leibler
    divergence between the empirical distributions of the reference and test
    windows is calculated, and drift is alarmed when a threshold is reached.

    A kdqtree is a combination of k-d trees and quad-trees; it is a binary tree
    (k-d) whose nodes contain square cells (quad) which are created via
    sequential splits along each dimension. This structure allows the
    calculation of the K-L divergence for continuous distributions, as the K-L
    divergence is defined on probability mass functions. The reference window is
    used to construct a kdq-tree, and the data in both the reference and test
    window are filed into it. The number of samples in each leaf of the tree is
    an empirical distribution for either dataset; this allows us to calculate
    the K-L divergence.

    The threshold for drift is determined using the desired alpha level by a
    bootstrap estimate for the critical value of the K-L divergence, drawing
    num_bootstrap_samples samples, ``2 * window_size times``, from the reference
    window.

    Additionally, the Kulldorff spatial scan statistic, which is a special case
    of the KL-divergence, can be calculated at each node of the kdq-tree, which
    gives a measure of the regions of the data space which have the greatest
    divergence between the reference and test windows. This can be used to
    visualize which regions of data space have the greatest drift. Note that
    these statistics are specific to the partitions of the data space by the
    kdq-tree, rather than (necessarily) the maximally different region in
    general. KSS is made available via ``to_plotly_dataframe``, which produces
    output structured for use with ``plotly.express.treemap``.

    Note that this algorithm could be used with other types of trees; the
    reference paper and this implementation use kdq-trees.

    Note that the current implementation does not explicitly handle categorical
    data.

    Ref. T. Dasu, S. Krishnan, S. Venkatasubramanian, and K. Yi, “An
    information-theoretic approach to detecting changes in multidimensional data
    streams,” in Proc. Symp. the Interface of Statistics, Computing Science, and
    Applications. Citeseer, 2006, Conference Proceedings, pp. 1-24.


    Attributes:
        total_samples (int): number of samples/batches the drift detector has
            ever been updated with.
        samples_since_reset (int): number of samples/batches since the last time
            the drift detector was reset
        drift_state (str): detector's current drift state. Can take values
            ``"drift"`` or ``None``.
    """

    _input_type = None

    def __init__(
        self,
        alpha=0.01,
        bootstrap_samples=500,
        count_ubound=100,
        cutpoint_proportion_lbound=2e-10,
        window_size=None,
        persistence=0.05,
        input_type="stream",
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
            window_size (int, optional): The minimum number of samples required
                to test whether drift has occurred. Only meaningful if stream
                is ``True``. Defaults to ``None``.
            persistence (float, optional): Persistence factor. If stream is ``True``,
                how many samples in a row, as a proportion of the window size,
                must be in the "drift region" of K-L divergence, in order for
                ``kdqTree`` to alarm and reset. Defaults to 0.05.
            input_type (str, optional): Whether data will be received one sample
                at a time, vs. received in batches of arbitrary size. For
                ``input_type == "batch"``, the reference window is defined with the
                first call to ``update()`` after a reset/initialization. For
                ``input_type == "stream"``, new samples are passed to update one at
                a time, and the reference data is defined by ``window_size``.
                Defaults to "stream".
        """
        if input_type == "stream" and window_size is None:
            raise ValueError(
                "Streaming kdqTree's window_size must be a positive integer."
            )
        if window_size is not None and window_size < 1 and input_type == "stream":
            raise ValueError(
                "Streaming kdqTree's window_size must be a positive integer."
            )
        super().__init__()
        self._input_type = input_type
        self.window_size = window_size
        self.persistence = persistence
        self.alpha = alpha
        self.bootstrap_samples = bootstrap_samples
        self.count_ubound = count_ubound
        self.cutpoint_proportion_lbound = cutpoint_proportion_lbound
        self.reset()

    def reset(self):
        """Initialize the detector's drift state and other relevant attributes.
        Intended for use after ``drift_state == 'drift'`` or initialization.
        """
        super().reset()
        self._ref_data = np.array([])
        self._test_data_size = 0
        self._kdqtree = None
        self._critical_dist = None
        self._drift_counter = 0  # samples consecutively in the drift region

    def update(self, ary):
        """Update the detector with a new sample (if ``input_type`` is "stream") or batch.
        Constructs the reference data's kdqtree; then, when sufficient samples
        have been received, puts the test data into the same tree; then, checks
        divergence between the reference and test data.

        Args:
            ary (numpy array): If just reset/initialized, the reference data.
            Otherwise, a new sample to put into the test window (if streaming)
            or a new batch of data to be compared to the reference window (if
            not streaming).
        """
        # TODO: validation. #17
        # if self._input_type == "stream" and ary.ndim != 1:
        #     raise ValueError(
        #         "Streaming kdqTree update only takes one sample at a time."
        #     )
        if self.drift_state == "drift":
            # TODO: depends on whether we want to dump the reference window vs. automatically replacing, HDDDM-style. #59
            self.reset()
        super().update()

        if self._kdqtree is None:  # ary is part of the new reference tree
            self._ref_data = (
                np.vstack([self._ref_data, ary]) if self._ref_data.size else ary
            )

            # check for drift if either: we're streaming and the reference
            # window is full, or we're doing batch detection
            if (
                self._input_type == "stream" and len(self._ref_data) == self.window_size
            ) or self._input_type == "batch":
                self._kdqtree = KDQTreePartitioner(
                    count_ubound=self.count_ubound,
                    cutpoint_proportion_lbound=self.cutpoint_proportion_lbound,
                )
                self._kdqtree.build(self._ref_data)
                self._ref_data = np.array([])  # don't need it anymore

                ref_counts = self._kdqtree.leaf_counts("build")
                self._critical_dist = self._get_critical_kld(ref_counts)
        else:  # new test sample(s)
            self._kdqtree.fill(ary, tree_id="test", reset=(self._input_type == "batch"))
            if self._input_type == "stream":
                self._test_data_size += (
                    1  # TODO after validation, should always be 1 #17
                )
            if self._input_type == "batch" or (
                self._test_data_size >= self.window_size
            ):
                test_dist = self._kdqtree.kl_distance(tree_id1="build", tree_id2="test")
                if test_dist > self._critical_dist:
                    if self._input_type == "stream":
                        self._drift_counter += 1
                        if self._drift_counter > self.persistence * self.window_size:
                            self.drift_state = "drift"
                    else:
                        self.drift_state = "drift"

    def _get_critical_kld(self, ref_counts):
        """Find the critical value of the Kullback-Leibler divergence, using the
        empirical distribution defined by the counts of the reference tree
        across the leaves.

        Args:
            ref_counts (numpy array): The counts from the reference tree across
                its leaves.

        Returns:
            float: the critical distance corresponding to ``self.alpha``, the
                achievable significance level.
        """
        ref_dist = KDQTreePartitioner._distn_from_counts(ref_counts)

        if self._input_type == "stream":
            sample_size = self.window_size
        else:
            sample_size = sum(ref_counts)

        # TODO vectorize?
        b_dist_pairs = []
        bin_indices = list(range(len(ref_counts)))
        bin_indices_df = pd.DataFrame({"leaf": bin_indices})
        for _ in range(self.bootstrap_samples):
            # note the maintenance of the leaf order!
            b_sample = np.random.choice(bin_indices, size=2 * sample_size, p=ref_dist)
            b_hist1 = np.unique(b_sample[:sample_size], return_counts=True)
            b_hist2 = np.unique(b_sample[sample_size:], return_counts=True)
            b_hist1 = pd.DataFrame({"leaf": b_hist1[0], "count": b_hist1[1]})
            b_hist2 = pd.DataFrame({"leaf": b_hist2[0], "count": b_hist2[1]})
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

    def to_plotly_dataframe(self, tree_id1="build", tree_id2="test", max_depth=None):
        """Generates a dataframe containing information about the kdqTree's structure
        and some node characteristics, intended for use with plotly.

        Args:
            tree_id1 (str, optional): Reference tree. If ``tree_id2`` is not
                specified, the only tree described. Defaults to "build".
            tree_id2 (str, optional): Test tree. If this is specified, the
                dataframe will also contain information about the difference
                between counts in each node for the reference vs. the test tree.
                Defaults to "test".
            max_depth (int, optional): Depth in the tree to which to recurse.
                Defaults to None.

        Returns:
            pd.DataFrame: A dataframe where each row corresponds to a node, and
            each column contains some information:

                * name: a label corresponding to which feature this split is on
                * idx: a unique ID for the node, to pass ``plotly.express.treemap``'s
                  id argument
                * parent_idx: the ID of the node's parent
                * cell_count: how many samples are in this node in the reference
                  tree.
                * depth: how deep the node is in the tree
                * count_diff: if ``tree_id2`` is specified, the change in counts from
                  the reference tree.
                * kss: the Kulldorff Spatial Scan Statistic for this node,
                  defined as the Kullback-Leibler divergence for this node
                  between the reference and test trees, using the individual
                  node and all other nodes combined as the bins for the
                  distributions.
        """

        return self._kdqtree.to_plotly_dataframe(tree_id1, tree_id2, max_depth)
