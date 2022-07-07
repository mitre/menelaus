import numpy as np
from numpy import unique
import pandas as pd
from pandas import DataFrame
import scipy.stats

from menelaus.drift_detector import StreamingDetector, BatchDetector
from menelaus.partitioners.KDQTreePartitioner import KDQTreePartitioner


class KdqTreeDetector():
    def __init__(
        self,
        window_size,
        persistence=0.05,
        alpha=0.01,
        bootstrap_samples=500,
        count_ubound=100,
        cutpoint_proportion_lbound=2e-10
    ):
        self.window_size = window_size
        self.persistence = persistence
        self.alpha = alpha
        self.bootstrap_samples = bootstrap_samples
        self.count_ubound = count_ubound
        self.cutpoint_proportion_lbound = cutpoint_proportion_lbound

    def reset(self):
        self._ref_data = np.array([])
        self._test_data_size = 0
        self._kdqtree = None
        self._critical_dist = None
        self._drift_counter = 0 # samples consecutively in the drift region
        self.input_cols = None

    def _prepare_data(self, data):
        if isinstance(data, pd.DataFrame):
            if self.input_cols is None:
                # The first update with a dataframe will constrain subsequent
                # input. This will also fire if set_reference has been used with
                # a dataframe.
                self.input_cols = data.columns
            elif self.input_cols is not None:
                if not data.columns.equals(self.input_cols):
                    raise ValueError(
                        "Columns of new data must match with columns of reference data."
                    )
            ary = data.values
        elif isinstance(data, np.ndarray):
            # This allows starting with a dataframe, then later passing bare
            # numpy arrays. For now, assume users are not miscreants.
            ary = data
        else:
            raise ValueError(
                """This method is only available for data inputs in the form of 
                a Pandas DataFrame or a Numpy Array."""
            )
        return ary

    def _evaluate_kdqtree(self, ary, input_type):
        # XXX - This is one final spot where we are still technically
        #       'branching' on input_type. However, in practice, this
        #       decomposes the update function so much that the two
        #       classes always call this with their respective input
        #       type - so it may not be much of an issue? @Anmol-Srivastava
        if self._kdqtree is None: # ary part of new ref tree
            self._ref_data = np.vstack([self._ref_data, ary]) if self._ref_data.size else ary
            
            if (input_type == 'stream' and len(self._ref_data) == self.window_size) or input_type == 'batch':
                # XXX - _inner_set_reference calls KLD itself, so needs to know sample_size
                #       We may streamline this later - Anmol-Srivastava
                self._inner_set_reference(self._ref_data, input_type)

        else: # new test sample(s)
            self._kdqtree.fill(ary, tree_id='test', reset=(input_type == 'batch'))
            if input_type == 'stream':
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
        # TODO ensure self.reset (or e.g. any function in that place uses right local version)
        self.reset()
        self._kdqtree = KDQTreePartitioner(
            count_ubound=self.count_ubound,
            cutpoint_proportion_lbound=self.cutpoint_proportion_lbound
        )
        self._kdqtree.build(ary)
        ref_counts = self._kdqtree.leaf_counts('build')
        sample_size = self.window_size if input_type == 'stream' else sum(ref_counts)
        self._critical_dist = self._get_critical_kld(ref_counts, sample_size)

    def to_plotly_dataframe(
        self,
        tree_id1="build",
        tree_id2="test",
        max_depth=None,
        input_cols=None
    ):
        if input_cols is not None:
            return self._kdqtree.to_plotly_dataframe(
                tree_id1, tree_id2, max_depth, input_cols
            )
        else:
            return self._kdqtree.to_plotly_dataframe(
                tree_id1, tree_id2, max_depth, self.input_cols
            )

    def _get_critical_kld(self, ref_counts, sample_size):
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
    def __init__(
        self,
        window_size,
        persistence=0.05,
        alpha=0.01,
        bootstrap_samples=500,
        count_ubound=100,
        cutpoint_proportion_lbound=2e-10
    ):
        if not isinstance(window_size, int) or window_size < 1:
            raise ValueError(f'window_size must be positive integer, was {window_size}')

        StreamingDetector.__init__(self)
        KdqTreeDetector.__init__(
            self,
            window_size,
            persistence,
            alpha,
            bootstrap_samples,
            count_ubound,
            cutpoint_proportion_lbound
        )
        self.reset()

    def reset(self):
        StreamingDetector.reset(self)
        KdqTreeDetector.reset(self)

    def update(self, data):
        ary = self._prepare_data(data)

        if self.drift_state == 'drift':
            self.reset()

        StreamingDetector.update(self)
        KdqTreeDetector._evaluate_kdqtree(self, ary, 'stream')


class KdqTreeBatch(KdqTreeDetector, BatchDetector):
    def __init__(
        self,
        alpha=0.01,
        bootstrap_samples=500,
        count_ubound=100,
        cutpoint_proportion_lbound=2e-10,
        window_size=None,
        persistence=0.05
    ):
        BatchDetector.__init__(self)
        KdqTreeDetector.__init__(
            self,
            window_size,
            persistence,
            alpha,
            bootstrap_samples,
            count_ubound,
            cutpoint_proportion_lbound
        )
        self.reset()

    def reset(self):
        BatchDetector.reset(self)
        KdqTreeDetector.reset(self)

    def set_reference(self, data):
        if isinstance(data, pd.DataFrame):
            # XXX - notice how inner_set calling KLD requires us to continue
            #       branching on input_type, which is not ideal - Anmol Srivastava
            self._inner_set_reference(data.values, input_type='batch')
            self.input_cols = data.columns
        elif isinstance(data, np.ndarray):
            self._inner_set_reference(data, input_type='batch')
        else:
            raise ValueError(
                "This method is only available for data inputs in the form of a Pandas DataFrame or a Numpy Array."
            )

    def update(self, data):
        ary = self._prepare_data(data)

        if self.drift_state == "drift":
            self.set_reference(self.ref_data)

        BatchDetector.update(self)
        KdqTreeDetector._evaluate_kdqtree(self, ary, 'batch')

        # if _evaluate_kdqtree resulted in drift for batch data, do a redundant check
        if self.drift_state == 'drift' and isinstance(data, pd.DataFrame):
            self.input_cols = data.columns
