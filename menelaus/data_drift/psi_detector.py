from menelaus.detector import BatchDetector
import pandas as pd
import numpy as np


class PSI(BatchDetector):
    """
    Parent class for PSI-based drift detector, it serves as a fundamental framework for batch data applications.

    The PSI (Population Stability Index) is employed for detecting distributional shifts between a reference population
    and a comparison population. This detector assesses changes by calculating the PSI, which measures the distributional
    change based on percentiles. The psi function in the detector compares the distributions of scores in reference and
    test populations and calculates the PSI values for different bins.

    In summary, the PSI drift detector provides a robust mechanism for monitoring and detecting distributional changes in
    populations, making it adaptable for various data settings and applications.

    Ref. :cite:t:`Psi2022`
    """

    input_type = "batch"

    def __init__(self, eps=1e-4, threshold=0.1):
        """
        Args:
            eps:The eps parameter in the function represents a small constant (1e-4) introduced to prevent division by zero
                when calculating percentages, ensuring numerical stability.
            threshold: It represents the threshold for detecting drift, if the calculated PSI value for a feature exceeds
                this threshold, it indicates drift in that feature, and the drift_state is set to 'drift'. This threshold is a
                user-defined value, and when crossed, it signifies a significant distributional change between the reference
                and test populations.
        """
        super().__init__()
        self.eps = eps
        self.threshold = threshold

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
        self.reference = X.reshape(
            len(X),
        )

    def reset(self):
        """
        Initialize relevant attributes to original values, to ensure information
        only stored from samples_since_reset onwards. Intended for use
        after ``drift_state == 'drift'``.
        """
        super().reset()

    def update(self, X: np.array, y_true=None, y_pred=None):
        """
        Update the detector with a new test batch. If drift is detected, new
        reference batch becomes most recent test batch.

        Args:
          X (numpy.array): next batch of data to detect drift on.
          y_true (numpy.array): true labels, not used in PSI
          y_pred (numpy.array): predicted labels, not used in PSI
        """
        if self._drift_state == "drift":
            self.reset()

        X, _, _ = super()._validate_input(X, None, None)

        super().update(X=X, y_true=None, y_pred=None)
        test_batch = (np.array(X)).reshape(
            len(X),
        )
        min_val = min(min(self.reference), min(test_batch))
        max_val = max(max(self.reference), max(test_batch))
        bins = self._bin_data(self.reference, min_val, max_val)
        bins_initial = pd.cut(self.reference, bins=bins, labels=range(1, len(bins)))
        df_initial = pd.DataFrame({"initial": self.reference, "bin": bins_initial})
        grp_initial = df_initial.groupby("bin").count()
        grp_initial["percent_initial"] = grp_initial["initial"] / sum(
            grp_initial["initial"]
        )
        bins_new = pd.cut(test_batch, bins=bins, labels=range(1, len(bins)))
        df_new = pd.DataFrame({"new": test_batch, "bin": bins_new})
        grp_new = df_new.groupby("bin").count()
        grp_new["percent_new"] = grp_new["new"] / sum(grp_new["new"])
        psi_value = self._PSI(grp_initial, grp_new)
        if psi_value >= self.threshold:
            self._drift_state = "drift"
            self.set_reference(test_batch)
        return psi_value

    def _bin_data(self, feature, min, max):
        """
        Bin the given feature based on the specified minimum and maximum values.

        Args:
            feature (numpy.array): The feature to be binned.
            min (float): The minimum value for binning.
            max (float): The maximum value for binning.

        Returns:
            list: A list of bin edges for the given feature.
        """
        if len(np.unique(feature)) < 10:
            bins = [
                min + (max - min) * (i) / len(np.unique(feature))
                for i in range(len(np.unique(feature)) + 1)
            ]
            bins[0] = min - self.eps
            bins[-1] = max + self.eps
            return bins
        else:
            bins = [min + (max - min) * (i) / 10 for i in range(10 + 1)]
            bins[0] = min - self.eps
            bins[-1] = max + self.eps
            return bins

    def _PSI(self, reference_feature, test_feature):
        """
        Calculate the Population Stability Index (PSI) between reference and test features.

        Args:
            reference_feature (pandas.DataFrame): Reference feature distribution.
            test_feature (pandas.DataFrame): Test feature distribution.

        Returns:
            float: The calculated PSI value indicating distributional change.
        """
        psi_df = reference_feature.join(test_feature, on="bin", how="inner")
        psi_df["percent_initial"] = psi_df["percent_initial"].apply(
            lambda x: self.eps if x == 0 else x
        )
        psi_df["percent_new"] = psi_df["percent_new"].apply(
            lambda x: self.eps if x == 0 else x
        )
        psi_df["psi"] = (psi_df["percent_initial"] - psi_df["percent_new"]) * np.log(
            psi_df["percent_initial"] / psi_df["percent_new"]
        )
        return np.mean(psi_df["psi"])
