import pandas as pd
import numpy as np
from menelaus.detector import BatchDetector
from scipy.stats import chi2_contingency, kstest, cramervonmises_2samp, fisher_exact
from menelaus.detector import BatchDetector

class GenericDetector(BatchDetector):
    """
    Generic Detector class for detecting batched data drift.

    This class provides a generic framework for batched data drift detection using various representations,
    divergence measures, and critical value functions.

    """

    def __init__(self, representation, divergence, crit_function):
        """
        Initialize the Generic Detector.

        Args:
            representation: The representation transformer for the detector.
            divergence: The divergence measure used to detect concept drift.
            crit_function: The critical value function for determining drift.
        """
        super().__init__(self)
        self.reset()
        self.representation = representation
        self.divergence = divergence
        self.crit_function = crit_function

    def set_reference(self, X, y_true=None, y_pred=None):
        """
        Set the reference data for the detector.

        Args:
            X (numpy.array): updated reference batch.
            y_true (numpy.array): true labels, not used in GenericDetector
            y_pred (numpy.array): predicted labels, not used in GenericDetector
        """
        X, _, _ = super()._validate_input(X, None, None)
        X = X.reshape(len(X),)
        X = self.representation.fit(X)#, y_true)
        self.reference = X

    def update(self, X, y_true=None, y_pred=None, alt=None):
        """
        Update the detector with a new test batch. If drift is detected, new
        reference batch becomes most recent test batch.

        Args:
            X (numpy.array): next batch of data to detect drift on.
            y_true (numpy.array): true labels, not used in GenericDetector
            y_pred (numpy.array): predicted labels, not used in GenericDetector
            alt:{‘two-sided’, ‘less’, ‘greater’}, optional(only used when performing Fisher's Exact Test)
                Defines the alternative hypothesis. The following options are available (default is ‘two-sided’):
                "two-sided": the odds ratio of the underlying population is not one
                "less": the odds ratio of the underlying population is less than one
                "greater": the odds ratio of the underlying population is greater than one
        Returns:
            div: Divergence measure.
        """
        if self.drift_state == "drift":
            self.reset()
            self.reference = self.test

        X, _, _ = super()._validate_input(X, None, None)
        X = X.reshape(len(X),)
        X = self.representation.fit(X)
        self.test = X  # , y_true, y_pred)
        if alt == None:
            self.div = self.divergence(self.reference, self.test)
        else:
            self.div = self.divergence(self.reference, self.test, alt)
        if self.crit_function(self.div):
            self.drift_state = "drift"
            return self.div
        else:
            return self.div

    def reset(self):
        """
        Initialize relevant attributes to original values, to ensure information
        only stored from samples_since_reset onwards. Intended for use
        after ``drift_state == 'drift'``
        """
        super().reset(self)
        self.reference = None
        self.test = None
        self.div = None
        pass

# region Validations

class IdentityValidation:
    """
    Identity transform for numerical features.

    This transform returns the input unchanged if it consists of numerical features.

    """

    def __init__(self):
        pass

    def fit(self, X_ref, y_true=None):
        """
        Fit the data to see if it is numerical only.

        Args:
            X (numpy.array): either reference data or test data.
            y_true (numpy.array): true labels, not used in GenericDetector

        Returns:
            X_ref: Confirmed numerical data.
        """
        numeric_dtypes = [np.number]
        if any(np.issubdtype(X_ref.dtype, dtype) for dtype in numeric_dtypes):
            return X_ref
        else:
            raise ValueError("No numerical data detected. Please pass numerical features.")

class CategoricalValidation:
    """
    This Validation is to check if the input data is categorical or not.
    """

    def __init__(self):
        pass

    def fit(self, X_ref, y_true=None):
        """
        Fit the data to test if it is in correct form or not.

        Parameters:
            X (numpy.array): either reference data or test data.
            y_true (numpy.array): true labels, not used in GenericDetector

        Returns:
            X_ref: Confirmed categorical data.
        """
        categorical_dtypes = [np.object_]
        if (any(np.issubdtype(X_ref.dtype, dtype) for dtype in categorical_dtypes)) or len(np.unique(X_ref)) < 10:
            return X_ref
        else:
            raise ValueError("No categorical columns detected. Please pass categorical features.")

class BinaryValidation:
    """
    This Validation checks if the input data consists of only (0,1)'s or (False,True)'s and ensures
    that the input data is in the Binary form.
    """

    def __init__(self):
        pass

    def fit(self, X_ref, y_true=None):
        """
        Fit the data to test if it is in correct form or not.

        Parameters:
            X (numpy.array): either reference data or test data.
            y_true (numpy.array): true labels, not used in GenericDetector

        Returns:
            X_ref: Confirmed binary data.
        """
        unique_values = np.unique(X_ref)
        if (set(unique_values) == {0, 1}) or (set(unique_values) == {False, True}):
            return X_ref
        else:
            raise ValueError("The X_ref data must consist of only (0,1)'s or (False,True)'s for the FETDrift detector.")

# endregion

# region Hypothesis test wrappers

def chi2Divergence(rep_ref, rep_test):
    """
    Calculate the p-value for the chi-squared test of independence between two categorical distributions.

    Parameters:
        rep_ref: Reference data for the test.
        rep_test: Test data for the test.

    Returns:
        pval: The p-value for the chi-squared test.
    """
    categories = np.unique(np.concatenate([rep_ref, rep_test]))
    array1 = np.concatenate([rep_ref, np.setdiff1d(rep_test, rep_ref)])
    counts_array1 = np.array([np.sum(rep_ref == category) for category in categories])
    counts_array2 = np.array([np.sum(rep_test == category) for category in categories])
    contingency_table = np.vstack([counts_array1, counts_array2])
    _, pval, _, _ = chi2_contingency(contingency_table)
    return pval

def ksDivergence(rep_ref, rep_test):
    """
    Calculate the p-value for the Kolmogorov-Smirnov test between two distributions.

    Parameters:
        rep_ref: Reference data for the test.
        rep_test: Test data for the test.

    Returns:
        pval: The p-value for the Kolmogorov-Smirnov test.
    """
    pval = kstest(rep_ref, rep_test).pvalue
    return pval

def cvmDivergence(rep_ref, rep_test):
    """
    Calculate the p-value for the Cramér-von Mises test between two distributions.

    Parameters:
        rep_ref: Reference data for the test.
        rep_test: Test data for the test.

    Returns:
        pval: The p-value for the Cramér-von Mises test.
    """
    pval = cramervonmises_2samp(rep_ref, rep_test).pvalue
    return pval

def fetDivergence(ref, test, alternative):
    """
    Calculate the p-value for the Fisher's Exact Test between two binary distributions.

    Parameters:
        rep_ref: Reference data for the test.
        rep_test: Test data for the test.
        alternative: {‘two-sided’, ‘less’, ‘greater’},The alternative hypothesis to be tested.
                "two-sided": the odds ratio of the underlying population is not one
                "less": the odds ratio of the underlying population is less than one
                "greater": the odds ratio of the underlying population is greater than one

    Returns:
        pval: The p-value for the Fisher's Exact Test.
    """
    ref, test = ref.astype(np.int64), test.astype(np.int64)
    n_ref, n = ref.shape[0], test.shape[0]
    p_val, odds_ratio = np.empty(1), np.empty(1)

    table = np.array([[np.sum(test), np.sum(ref)], [n - np.sum(test), n_ref - np.sum(ref)]])
    odds_ratio[0], p_val[0] = fisher_exact(table, alternative)

    return p_val

# endregion

# region Critical value functions

def crit(pval, alpha=0.05):
    """
    Compare p-value with alpha to determine statistical significance.

    Parameters:
    - pval: The p-value from the hypothesis test.
    - alpha: The significance level.

    Returns:
    - bool: True if the test is significant(p-value smaller than alpha, False otherwise.)
    """
    if pval <= alpha:
        return True
    else:
        return False

# endregion

# region Implemented classes

class CHIDetector(GenericDetector):
    """
    Chi-squared Test Data drift detector using the chi-squared test for categorical data.

    Serves the same purpose as above, just provide users an easy way to call the detector.

    """

    def __init__(self):
        super().__init__(
            representation=CategoricalValidation(),
            divergence=chi2Divergence,
            crit_function=crit,
        )

class KSDetector(GenericDetector):
    """
    Kolmogorov-Smirnov Test Data drift detector using the Kolmogorov-Smirnov test for numerical data.

    Serves the same purpose as above, just provide users an easy way to call the detector.

    """

    def __init__(self):
        super().__init__(
            representation=IdentityValidation(),
            divergence=ksDivergence,
            crit_function=crit,
        )

class CVMDetector(GenericDetector):
    """
    Cramér-von MisesData Test drift detector using the Cramér-von Mises test for numerical data.

    Serves the same purpose as above, just provide users an easy way to call the detector.

    """

    def __init__(self):
        super().__init__(
            representation=IdentityValidation(),
            divergence=cvmDivergence,
            crit_function=crit,
        )

class FETDetector(GenericDetector):
    """
    Fisher's Exact Test Data drift detector using Fisher's Exact Test for binary data.

    Serves the same purpose as above, just provide users an easy way to call the detector.

    """

    def __init__(self):
        super().__init__(
            representation=BinaryValidation(),
            divergence=fetDivergence,
            crit_function=crit,
        )

# endregion
