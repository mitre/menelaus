import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KernelDensity
from menelaus.detector import StreamingDetector
from menelaus.change_detection.page_hinkley import PageHinkley


class PCACD(StreamingDetector):
    """Principal Component Analysis Change Detection (PCA-CD) is a drift
    detection algorithm which checks for change in the distribution of the given
    data using one of several divergence metrics calculated on the data's
    principal components.

    First, principal components are built from the reference window - the
    initial ``window_size`` samples. New samples from the test window, of the same
    width, are projected onto these principal components. The divergence metric
    is calculated on these scores for the reference and test windows; if this
    metric diverges enough, then we consider drift to have occurred. This
    threshold is determined dynamically through the use of the Page-Hinkley test.

    Once drift is detected, the reference window is replaced with the current
    test window, and the test window is initialized.

    Ref. :cite:t:`qahtan2015pca`

    Attributes:
        step (int): how frequently (by number of samples), to detect drift.
            This is either 100 samples or ``sample_period * window_size``, whichever
            is smaller.
        ph_threshold (float): threshold parameter for the internal Page-Hinkley
            detector. Takes the value of ``.01 * window_size``.
        num_pcs (int): the number of principal components being used to meet
            the specified ``ev_threshold`` parameter.
    """

    input_type = "streaming"

    def __init__(
        self,
        window_size,
        ev_threshold=0.99,
        delta=0.1,
        divergence_metric="kl",
        sample_period=0.05,
        online_scaling=True,
    ):
        """
        Args:
            window_size (int): size of the reference window. Note that
                ``PCA_CD`` will only try to detect drift periodically, either
                every 100 observations or 5% of the ``window_size``, whichever
                is smaller.
            ev_threshold (float, optional): Threshold for percent explained
                variance required when selecting number of principal components.
                Defaults to 0.99.
            delta (float, optional): Parameter for Page Hinkley test. Minimum
                amplitude of change in data needed to sound alarm. Defaults to
                0.1.
            divergence_metric (str, optional): divergence metric when comparing
                the two distributions when detecting drift. Defaults to "kl".

                * "kl" - Jensen-Shannon distance, a symmetric bounded form of
                  Kullback-Leibler divergence, uses kernel density estimation
                  with Epanechnikov kernel

                * "intersection" - intersection area under the curves for the
                  estimated density functions, uses histograms to estimate
                  densities of windows. A discontinuous, less accurate estimate
                  that should only be used when efficiency is of concern.

            sample_period (float, optional): how often to check for drift. This
                is 100 samples or ``sample_period * window_size``, whichever is
                smaller. Default .05, or 5% of the window size.
            online_scaling (bool, optional): whether to standardize the data as
                it comes in, using the reference window, before applying PCA.
                Defaults to ``True``.
        """
        super().__init__()
        self.window_size = window_size
        self.ev_threshold = ev_threshold
        self.divergence_metric = divergence_metric
        self.sample_period = sample_period

        # Initialize parameters
        self.step = min(100, round(self.sample_period * window_size))
        self.ph_threshold = round(0.01 * window_size)
        self.bins = int(np.floor(np.sqrt(self.window_size)))
        self.delta = delta

        self._drift_detection_monitor = PageHinkley(
            delta=self.delta, threshold=self.ph_threshold, burn_in=0
        )

        self.num_pcs = None

        self.online_scaling = online_scaling
        if self.online_scaling is True:
            self._reference_scaler = StandardScaler()

        self._build_reference_and_test = True
        self._reference_window = pd.DataFrame()
        self._test_window = pd.DataFrame()
        self._pca = None
        self._reference_pca_projection = pd.DataFrame()
        self._test_pca_projection = pd.DataFrame()
        self._density_reference = {}
        self._change_score = [0]

    def update(self, X, y_true=None, y_pred=None):
        """Update the detector with a new observation.

        Args:
            X (numpy.ndarray): next observation
            y_true (numpy.ndarray): true label of observation - not used in PCACD
            y_pred (numpy.ndarray): predicted label of observation - not used in PCACD
        """
        X, _, _ = super()._validate_input(X, None, None)
        super().update(X, None, None)

        if self._build_reference_and_test:
            if self.drift_state is not None:
                self._reference_window = self._test_window.copy()
                if self.online_scaling is True:
                    # we'll need to refit the scaler. this occurs when both
                    # reference and test windows are full, so, inverse_transform
                    # first, here
                    self._reference_window = pd.DataFrame(
                        self._reference_scaler.inverse_transform(self._reference_window)
                    )
                self._test_window = pd.DataFrame()
                self.reset()
                self._drift_detection_monitor.reset()

            elif len(self._reference_window) < self.window_size:
                self._reference_window = pd.concat(
                    [self._reference_window, pd.DataFrame(X)]
                )

            elif len(self._test_window) < self.window_size:
                self._test_window = pd.concat([self._test_window, pd.DataFrame(X)])

            if len(self._test_window) == self.window_size:
                self._build_reference_and_test = False

                # Fit Reference window onto PCs
                if self.online_scaling is True:
                    self._reference_window = pd.DataFrame(
                        self._reference_scaler.fit_transform(self._reference_window)
                    )
                    self._test_window = pd.DataFrame(
                        self._reference_scaler.transform(self._test_window)
                    )

                # Compute principal components
                self._pca = PCA(self.ev_threshold)
                self._pca.fit(self._reference_window)
                self.num_pcs = len(self._pca.components_)

                # Project reference window onto PCs
                self._reference_pca_projection = pd.DataFrame(
                    self._pca.transform(self._reference_window),
                )

                # Project test window onto PCs
                self._test_pca_projection = pd.DataFrame(
                    self._pca.transform(self._test_window),
                )

                # Compute reference distribution
                for i in range(self.num_pcs):

                    if self.divergence_metric == "intersection":
                        # Histograms need the same bin edges so find bounds from
                        # both windows to inform range for reference and test
                        self.lower = min(
                            self._reference_pca_projection.iloc[:, i].min(),
                            self._test_pca_projection.iloc[:, i].min(),
                        )

                        self.upper = max(
                            self._reference_pca_projection.iloc[:, i].max(),
                            self._test_pca_projection.iloc[:, i].max(),
                        )

                        self._density_reference[f"PC{i + 1}"] = self._build_histograms(
                            self._reference_pca_projection.iloc[:, i],
                            bins=self.bins,
                            bin_range=(self.lower, self.upper),
                        )

                    else:
                        self._density_reference[f"PC{i + 1}"] = self._build_kde(
                            self._reference_pca_projection.iloc[:, i]
                        )

        else:

            # Add new obs to test window
            if self.online_scaling is True:
                next_obs = pd.DataFrame(self._reference_scaler.transform(X))
            self._test_window = pd.concat([self._test_window.iloc[1:, :], next_obs])

            # Project new observation onto PCs
            next_proj = pd.DataFrame(
                self._pca.transform(np.array(next_obs).reshape(1, -1)),
            )

            # Winsorize incoming data to align with reference and test histograms
            if self.divergence_metric == "intersection":
                for i in range(self.num_pcs):
                    if next_proj.iloc[0, i] < self.lower:
                        next_proj.iloc[0, i] = self.lower

                    elif next_proj.iloc[0, i] > self.upper:
                        next_proj.iloc[0, i] = self.upper

            # Add projection to test projection data
            self._test_pca_projection = pd.concat(
                [self._test_pca_projection.iloc[1:, :], next_proj]
            )

            # Compute change score
            if (((self.total_samples - 1) % self.step) == 0) and (
                (self.total_samples - 1) != 0
            ):
                # Compute density distribution for test data
                self._density_test = {}
                for i in range(self.num_pcs):

                    if self.divergence_metric == "intersection":

                        self._density_test[f"PC{i + 1}"] = self._build_histograms(
                            self._test_pca_projection.iloc[:, i],
                            bins=self.bins,
                            bin_range=(self.lower, self.upper),
                        )

                    elif self.divergence_metric == "kl":
                        self._density_test[f"PC{i + 1}"] = self._build_kde(
                            self._test_pca_projection.iloc[:, i]
                        )

                # Compute current score
                change_scores = []

                if self.divergence_metric == "kl":
                    for i in range(self.num_pcs):

                        change_scores.append(
                            self._jensen_shannon_distance(
                                self._density_reference[f"PC{i + 1}"],
                                self._density_test[f"PC{i + 1}"],
                            )
                        )

                elif self.divergence_metric == "intersection":
                    for i in range(self.num_pcs):
                        change_scores.append(
                            self._intersection_divergence(
                                self._density_reference[f"PC{i + 1}"],
                                self._density_test[f"PC{i + 1}"],
                            )
                        )

                change_score = max(change_scores)
                self._change_score.append(change_score)

                self._drift_detection_monitor.update(X=change_score)

                if self._drift_detection_monitor.drift_state is not None:
                    self._build_reference_and_test = True
                    self.drift_state = "drift"

    def reset(self):
        """Initialize the detector's drift state and other relevant attributes.
        Intended for use after ``drift_state == 'drift'``.
        """
        super().reset()

    @classmethod
    def _build_kde(cls, sample):
        """Compute the Kernel Density Estimate for a given 1D data stream

        Args:
            sample: 1D data for which we desire to estimate its density function

        Returns:
            Dict with density estimates for each value and KDE object

        """
        sample_length = len(sample)
        bandwidth = 1.06 * np.std(sample, ddof=1) * (sample_length ** (-1 / 5))
        kde_object = KernelDensity(bandwidth=bandwidth, kernel="epanechnikov").fit(
            sample.values.reshape(-1, 1)
        )
        # score_samples gives log-likelihood for each point, true density values
        # should be > 0 so exponentiate
        density = np.exp(kde_object.score_samples(sample.values.reshape(-1, 1)))

        return {"density": density, "object": kde_object}

    @staticmethod
    def _build_histograms(sample, bins, bin_range):
        """
        Compute the histogram density estimates for a given 1D data stream.
        Density estimates consist of the value of the pdf in each bin,
        normalized s.t. integral over the entire range is 1

        Args:
            sample: 1D array in which we desire to estimate its density function
            bins: number of bins for estimating histograms. Equal to sqrt of
                cardinality of ref window
            bin_range: (float, float) lower and upper bound of histogram bins

        Returns:
            Dict of bin edges and corresponding density values (normalized s.t.
            they sum to 1)

        """

        density = np.histogram(sample, bins=bins, range=bin_range, density=True)
        return {
            "bin_edges": list(density[1]),
            "density": list(density[0] / np.sum(density[0])),
        }

    @classmethod
    def _jensen_shannon_distance(cls, density_reference, density_test):
        """Computes Jensen Shannon between two distributions

        Args:
            density_reference (dict): dictionary of density values and object
                from ref distribution
            density_test (dict): dictionary of density values and object from
                test distribution

        Returns:
            Change Score

        """
        js = jensenshannon(density_reference["density"], density_test["density"])
        return js

    @staticmethod
    def _intersection_divergence(density_reference, density_test):
        """
        Computes Intersection Area similarity between two distributions using
        histogram density estimation method. A value of 0 means the
        distributions are identical, a value of 1 means they are completely
        different

        Args:
            density_reference (dict): dictionary of density values from
                reference distribution
            density_test (dict): dictionary of density values from test
                distribution

        Returns:
            Change score

        """

        intersection = np.sum(
            np.minimum(density_reference["density"], density_test["density"])
        )
        divergence = 1 - intersection

        return divergence
