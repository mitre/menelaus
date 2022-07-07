import numpy as np
from numpy import unique
import pandas as pd
from pandas import DataFrame
import scipy.stats

from menelaus.drift_detector import StreamingDetector, BatchDetector
from menelaus.partitioners.KDQTreePartitioner import KDQTreePartitioner

class KdqTreeStreaming(StreamingDetector):
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
        super().__init__()
        self.window_size = window_size
        self.persistence = persistence
        self.alpha = alpha
        self.bootstrap_samples = bootstrap_samples
        self.count_ubound = count_ubound
        self.cutpoint_proportion_lbound = cutpoint_proportion_lbound
        self.reset()

    def reset(self):
        super().reset()
        self._ref_data = np.array([])
        self._test_data_size = 0
        self._kdqtree = None
        self._critical_dist = None
        self._drift_counter = 0 # samples consecutively in the drift region
        self.input_cols = None

    def _inner_set_reference(self, ary):
        self.reset()
        self._kdqtree = KDQTreePartitioner(
            count_ubound=self.count_ubound,
            cutpoint_proportion_lbound=self.cutpoint_proportion_lbound
        )
        self._kdqtree.build(ary)
        ref_counts = self._kdqtree.leaf_counts('build')
        self._critical_dist = self._get_critical_kld(ref_counts)

    def update(self, data):
        if isinstance(data, DataFrame):
            if self.input_cols is None:
                self.input_cols = data.columns
            elif self.input_cols is not None:
                if not data.columns.equals(self.input_cols):
                    raise ValueError(f'Columns {data.columns} != reference data columns {self.input_cols}')
            ary = data.values
        elif isinstance(data, np.ndarray):
            ary = data
        else:
            raise ValueError(f'Invalid data type {type(data)}')

        if self.drift_state == 'drift':
            self.reset()

        super().update()

        if self._kdqtree is None:
            self._ref_data = np.vstack([self._ref_data, ary]) if self._ref_data.size else ary
            if len(self._ref_data) == self.window_size:
                self._inner_set_reference(self._ref_data)
        else:
            self._kdqtree.fill(ary, tree_id="test", reset=False) # F, b/c input type is streaming
            self._test_data_size += 1
            
            if self._test_data_size >= self.window_size:
                test_dist = self._kdqtree.kl_distance(tree_id1="build", tree_id2="test")
                if test_dist > self._critical_dist:
                    self._drift_counter += 1
                    if self._drift_counter > self.persistence * self.window_size:
                        self.drift_state = "drift"

    def _get_critical_kd(self, ref_counts):
        ref_dist = KDQTreePartitioner._distn_from_counts(ref_counts)
        sample_size = self.window_size
     
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


class KdqTreeBatch(BatchDetector):
    pass

