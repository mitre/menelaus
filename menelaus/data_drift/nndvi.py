import numpy as np
from scipy.stats import norm
import warnings

from menelaus.detector import BatchDetector
from menelaus.partitioners import NNSpacePartitioner


class NNDVI(BatchDetector):
    """
    This class encodes the Nearest Neigbors Density Variation
    Identification (NN-DVI) drift detection algorithm, introduced
    in Liu et al. (2018). Note that this implementation is intended
    for batch datasets, rather than the streaming context.

    Broadly, NN-DVI combines a reference and test data batch, creates
    a normalized version of the subsequent adjacency matrix (after a
    k-NN search), and then analyzes distance changes in the reference
    and test sections of the combined adjacency matrix. Those changes
    are compared against a threshold distance value, which is found
    by randomly sampling new reference and test sections, then fitting
    a Gaussian distribution to distance changes for those trials.

    Attributes:
        total_samples (int): number of batches the drift detector has ever
            been updated with.
        samples_since_reset (int): number of batches since the last drift detection.
        drift_state (str): detector's current drift state. Can take values
            ``"drift"``, ``"warning"``, or ``None``.
        k_nn (int): the 'k' in k-Nearest-Neighbor (k-NN) search
        reference_batch (numpy.array): initial batch of data
        sampling_times (int): number of times to perform sampling for threshold estimation
        alpha (float): significance level for detecting drift
    """

    def __init__(self, k_nn: int = 30, sampling_times: int = 500, alpha: float = 0.01):
        """
        Attributes:
            k_nn (int, optional): the 'k' in k-Nearest-Neighbor (k-NN) search. Default 30.
            sampling_times (int, optional): number of times to perform sampling for threshold
                estimation. Default 500.
            alpha (float, optional): significance level for detecting drift. Default 0.01.
        """
        super().__init__()
        self.k_nn = k_nn
        self.sampling_times = sampling_times
        self.alpha = alpha

    def update(self, X: np.array, y_true=None, y_pred=None):
        """
        Update the detector with a new test batch. If drift is detected, new
        reference batch becomes most recent test batch.

        Args:
          X (numpy.array): next batch of data to detect drift on.
          y_true (numpy.array): true labels, not used in NN-DVI
          y_pred (numpy.array): predicted labels, not used in NN-DVI
        """
        if self._drift_state == "drift":
            self.reset()

        X, _, _ = super()._validate_input(X, None, None)

        super().update(X=X, y_true=None, y_pred=None)
        test_batch = np.array(X)

        nnsp = NNSpacePartitioner(self.k_nn)
        nnsp.build(self.reference_batch, test_batch)
        M_nnps = nnsp.nnps_matrix
        v_ref, v_test = nnsp.v1, nnsp.v2
        d_act = NNSpacePartitioner.compute_nnps_distance(M_nnps, v_ref, v_test)

        theta_drift = self._compute_drift_threshold(
            M_nnps, v_ref, v_test, self.sampling_times, self.alpha
        )
        if d_act > theta_drift:
            self._drift_state = "drift"
            self.set_reference(test_batch)

    def set_reference(self, X, y_true=None, y_pred=None):
        """
        Set the detector's reference batch to an updated value; typically
        used in ``update``.

        Attributes:
            X (numpy.array): updated reference batch
            y_true (numpy.array): true labels, not used in NNDVI
            y_pred (numpy.array): predicted labels, not used in NNDVI
        """
        X, _, _ = super()._validate_input(X, None, None)
        self.reference_batch = X

    def reset(self):
        """
        Initialize relevant attributes to original values, to ensure information
        only stored from samples_since_reset onwards. Intended for use
        after ``drift_state == 'drift'``.
        """
        super().reset()

    @staticmethod
    def _compute_drift_threshold(M_nnps, v_ref, v_test, sampling_times, alpha):
        """
        Compute critical value for drift detection, by repeatedly sampling
        NNPS distance calculations and estimating a normal distribution from
        the same. Takes an NNPS matrix, indices of reference/test sections in
        the same, number of times to sample, and a significance level for which
        to determine the critical value. Intended for use within ``update`` after
        the data has been partitioned.

        Attributes:
            M_nnps (numpy.array): NNPS matrix that results from ``NNSpacePartitioner.build``
            v_ref (numpy.array): 1D array of same length as ``M_nnps``, with a one-hot encoding
                of reference data point locations in ``M_nnps``.
            v_test (numpy.array): 1D array of same length as ``M_nnps``, with a one-hot
                encoding of test data point locations in ``M_nnps``.
            sampling_times (int): number of times to perform sampling for threshold estimation
            alpha (float): significance level for detecting drift
        """
        # TODO - Would like to parallelize this - Anmol
        d_shuffle = []
        for _ in range(sampling_times):
            v1_shuffle = np.random.permutation(v_ref)
            v2_shuffle = 1 - v1_shuffle

            d_i_shuffle = NNSpacePartitioner.compute_nnps_distance(
                M_nnps, v1_shuffle, v2_shuffle
            )
            d_shuffle.append(d_i_shuffle)
        mu, std = norm.fit(d_shuffle)
        drift_threshold = norm.ppf(1 - alpha, mu, std)
        return drift_threshold
