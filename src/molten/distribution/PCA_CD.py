import statistics
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from molten.drift_detector import DriftDetector
from molten.distribution.kl_divergence import kl_divergence
from molten.other.PageHinkley import PageHinkley


class PCA_CD(DriftDetector):
    """Principal Component Analysis Change Detection (PCA-CD) is a drift
    detection algorithm which checks for change in the distribution of the given
    data using one of several divergence metrics calculated on the data's
    principal components.

    First, principal components are built from the reference window - the
    initial window_size samples. New samples from the test window, of the same
    width, are projected onto these principal components. The divergence metric
    is calculated on these scores for the reference and test windows; if this
    metric diverges enough, then we consider drift to have occurred. This
    threshold is determined dynamically through the use of the Page-Hinkley test.

    Once drift is detected, the reference window is replaced with the current
    test window, and the test window is initialized.

    Ref. Qahtan, A., Wang, S. A PCA-Based Change Detection Framework for
    Multidimensional Data Streams Categories and Subject Descriptors. KDD '15:
    The 21st ACM SIGKDD International Conference on Knowledge Discovery and Data
    Mining, 935-44. https://doi.org/10.1145/2783258.2783359
    """

    def __init__(
        self,
        window_size,
        ev_threshold=0.99,
        delta=0.1,
        divergence_metric="kl",
        sample_period=0.05,
        online_scaling=False,
        track_state=False,
    ):
        """
        Args:
            window_size (int): size of the reference window. Note that PCA_CD
                will only try to detect drift periodically, either every 100
                observations or 5% of the window_size, whichever is smaller.
            ev_threshold (float, optional): Threshold for percent explained
                variance required when selecting number of principal components.
                Defaults to 0.99.
            delta (float, optional): Parameter for Page Hinkley test. Minimum
                amplitude of change in data needed to sound alarm. Defaults to 0.1.
            divergence_metric (str, optional): divergence metric when comparing
                the two distributions when detecting drift. Defaults to "kl".
                    "kl" - symmetric Kullback-Leibler divergence
                    "llh" - log-likelihood
                    "intersection" - intersection area under the curves for the
                    estimated density functions.
            sample_period (float, optional): how often to check for drift. This
                is 100 samples or sample_period * window_size, whichever is
                smaller. Default .05, or 5% of the window size.
            online_scaling (bool, optional): whether to standardize the data as
                it comes in, using the reference window, before applying PCA.
                Defaults to False.
            track_state (bool, optional): whether to store the status of the
                Page Hinkley detector every time drift is identified.
                Defaults to False.
        """
        super().__init__()
        self.window_size = window_size
        self.ev_threshold = ev_threshold
        self.divergence_metric = divergence_metric
        self.track_state = track_state
        self.sample_period = sample_period

        # Initialize parameters
        self.step = min(100, round(self.sample_period * window_size))
        self.xi = round(0.01 * window_size)

        self.delta = delta

        self._drift_detection_monitor = PageHinkley(
            delta=self.delta, xi=self.xi, burn_in=0
        )
        if self.track_state:
            self._drift_tracker = pd.DataFrame()

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
        self._kde_track_reference = {}
        self._kde_track_test = {}

    def update(self, next_obs):
        """Update the detector with a new observation.

        Args:
            next_obs: next observation, as a pandas Series
        """

        if self._build_reference_and_test:
            if self.drift_state is not None:
                self._reference_window = self._test_window.copy()
                if self.online_scaling is True:
                    # we'll need to refit the scaler. this occurs when both reference and test
                    # windows are full, so, inverse_transform first, here
                    self._reference_window = pd.DataFrame(
                        self._reference_scaler.inverse_transform(self._reference_window)
                    )
                self._test_window = pd.DataFrame()
                self.reset()
                self._drift_detection_monitor.reset()

            elif len(self._reference_window) < self.window_size:
                self._reference_window = self._reference_window.append(next_obs)

            elif len(self._test_window) < self.window_size:
                self._test_window = self._test_window.append(next_obs)

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

                # Project Reference window onto PCs
                self._reference_pca_projection = pd.DataFrame(
                    self._pca.transform(self._reference_window),
                    # columns=[f"PC{i}" for i in list(range(1, self.num_pcs + 1))],
                    # index=self._reference_window.index,
                )

                # Compute reference distribution
                for i in range(self.num_pcs):
                    self._kde_track_reference[f"PC{i+1}"] = self._build_kde_track(
                        self._reference_pca_projection.iloc[:, i]
                    )

                # Project test window onto PCs
                self._test_pca_projection = pd.DataFrame(
                    self._pca.transform(self._test_window),
                    # columns=[f"PC{i}" for i in list(range(1, self.num_pcs + 1))],
                    # index=self._test_window.index,
                )

                # Compute test distribution
                for i in range(self.num_pcs):
                    self._kde_track_test[f"PC{i+1}"] = self._build_kde_track(
                        self._test_pca_projection.iloc[:, i]
                    )

        else:

            # Add new obs to test window
            if self.online_scaling is True:
                next_obs = pd.DataFrame(self._reference_scaler.transform(next_obs))
            self._test_window = self._test_window.iloc[1:, :].append(next_obs)

            # Project new observation onto PCs
            next_proj = pd.DataFrame(
                self._pca.transform(np.array(next_obs).reshape(1, -1)),
                # columns=[f"PC{i}" for i in list(range(1, self.num_pcs + 1))],
                # index=pd.Series(self._test_window.index[-1]),
            )

            # Add projection to test projection data
            self._test_pca_projection = self._test_pca_projection.iloc[1:, :].append(
                next_proj, ignore_index=True
            )

            # Compute test distribution
            self._kde_track_test = {}
            for i in range(self.num_pcs):
                self._kde_track_test[f"PC{i+1}"] = self._build_kde_track(
                    self._test_pca_projection.iloc[:, i]
                )

            # Compute change score
            if (self.total_samples % self.step) == 0 and self.total_samples != 0:

                # Compute current score
                change_scores = []
                if self.divergence_metric == "kl":
                    for i in range(self.num_pcs):
                        change_scores.append(
                            max(
                                kl_divergence(
                                    self._kde_track_reference[f"PC{i+1}"][
                                        "kde_estimate"
                                    ]["density"],
                                    self._kde_track_test[f"PC{i+1}"]["kde_estimate"][
                                        "density"
                                    ],
                                    type="discrete",
                                ),
                                kl_divergence(
                                    self._kde_track_test[f"PC{i+1}"]["kde_estimate"][
                                        "density"
                                    ],
                                    self._kde_track_reference[f"PC{i+1}"][
                                        "kde_estimate"
                                    ]["density"],
                                    type="discrete",
                                ),
                            )
                        )

                elif self.divergence_metric == "intersection":
                    for i in range(self.num_pcs):
                        change_scores.append(
                            self._intersection_area(
                                self._kde_track_reference[f"PC{i+1}"]["kde_estimate"][
                                    "density"
                                ],
                                self._kde_track_test[f"PC{i+1}"]["kde_estimate"][
                                    "density"
                                ],
                            )
                        )

                elif self.divergence_metric == "llh":
                    for i in range(self.num_pcs):
                        change_scores.append(
                            self._log_likelihood(
                                self._kde_track_reference[f"PC{i+1}"]["kde_estimate"][
                                    "point"
                                ],
                                self._kde_track_test[f"PC{i+1}"]["kde_estimate"][
                                    "point"
                                ],
                            )
                        )

                change_score = max(change_scores)

                self._drift_detection_monitor.update(
                    next_obs=change_score, obs_id=next_obs.index.values[0]
                )

                if self._drift_detection_monitor.drift_state is not None:
                    self._build_reference_and_test = True
                    self.drift_state = "drift"
                    if self.track_state:
                        self._drift_tracker = self._drift_tracker.append(
                            self._drift_detection_monitor.to_dataframe()
                        )

        super().update()

    def _epanechnikov_kernel(self, x_j, approx_zero=False):
        """Calculate the Epanechnikov kernel value for a given value x_j, for
        use in kernel density estimation.

        Args:
            x_j: single sample value
            approx_zero: whether or not to approximate zero with a very small value
                may be unnecessary in practice. Default False.

        Returns:
            Epanechnikov kernel value for x_j.

        """
        if approx_zero:
            const = 10 ** (-6)
        else:
            const = 0
        return [const if (x_j < 0 or x_j > 1) else (3 / 4) * (1 - (x_j ** 2))][0]

    def _log_likelihood(self, p, q):
        """Computes Log-Likelihood similarity between two distributions

        Args:
            p (list): List of values from first distribution
            q (list): List of values from second distribution

        Returns:
          Log-likelihood similarity

        """
        m = len(p)
        bandwidth = 1.06 * statistics.stdev(q) * (m ** (-1 / 5))
        LLH_q = sum(
            [
                np.log(
                    sum(
                        [
                            (1 / m) * self._epanechnikov_kernel((y - x) / bandwidth)
                            for x in p
                        ]
                    )
                )
                for y in q
            ]
        )
        LLH_p = sum(
            [
                np.log(
                    sum(
                        [
                            (1 / m) * self._epanechnikov_kernel((y - x) / bandwidth)
                            for x in p
                        ]
                    )
                )
                for y in p
            ]
        )
        divergence = abs((LLH_q / len(q)) - (LLH_p / len(p)))

        return divergence

    def _intersection_area(self, p, q):
        """Computes Intersection Area similarity between two distributions

        Args:
            p (list): List of values from first distribution
            q (list): List of values from second distribution

        Returns:
            Intersection area

        """
        divergence = (1 / 2) * sum([abs(x - y) for x, y in zip(p, q)])

        return divergence

    def _build_kde_track(self, X):
        """Compute the Kernel Density Estimate Track for a given 1D data stream

        Args:
            X: 1D data in which we desire to estimate its density function

        Returns:
            Bandwidth and dictionary of resampling points

        """
        m = len(X)
        bandwidth = 1.06 * statistics.stdev(X) * (m ** (-1 / 5))
        density = [
            (1 / (m * bandwidth))
            * sum([self._epanechnikov_kernel((x - x_j) / bandwidth) for x_j in X])
            for x in X
        ]

        return {
            "bandwidth": bandwidth,
            "kde_estimate": {"point": X, "density": density},
        }
