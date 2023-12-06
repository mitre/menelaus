"""
For now, I think it's alright just to keep all these in the same module.

TODO (now)
finish the __init__.py import statement
docstring for each class and member
clear in-line TODO items
self.representation needs a new variable name
unit tests for the implemented classes

TODO (later)
categorical handling might need rework
spell-checking and consistency w/ rest of library on error messages and validation calls
internals-as-numpy-arrays:
    implementation as not-an-ensemble is not the most code-conservative. Should fix, or see if there's a big performance advantage for vectorization

"""
from menelaus.detector import BatchDetector

import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency, kstest, cramervonmises_2samp, fisher_exact


class GenericDetector(BatchDetector):
    def __init__(self, representation, divergence, crit_function):
        super().__init__(self)
        self.reset()
        ########################################################################
        self.representation = representation
        self.divergence = divergence
        self.crit_function = crit_function

    def set_reference(self, X, y_true=None, y_pred=None):
        X, _, _ = super()._validate_input(X, None, None)
        self.reset()
        X = pd.DataFrame(X, columns=self._input_cols)
        X = self.representation.fit(X)  # , y_true)
        self.reference = self.representation.transform(X)

    def update(self, X, y_true=None, y_pred=None, alt=None):
        if self.drift_state == "drift":
            self.reset()
            # TODO: do we generally impose self.set_reference after drift?
            # if the transform is a very costly operation, we would want to save
            # self.test in between updates, and skip the validation..
            self.set_reference(X, y_true, y_pred)

        X, _, _ = super()._validate_input(X, None, None)

        # TODO: do not use dataframes
        X = pd.DataFrame(X, columns=self._input_cols)
        X = self.representation.fit(X)
        self.test = self.representation.transform(X)  # , y_true, y_pred)
        if alt == None:
            self.div = self.divergence(self.reference, self.test)
        else:
            self.div = self.divergence(self.reference, self.test, alt)
        if self.crit_function(self.div):
            # if self.div >= self.crit_val:
            self.drift_state = "drift"

    def reset(self):
        super().reset(self)
        self.reference = None
        ########################################################################
        self.test = None
        self.div = None
        pass


# region Transforms
class IdentityTransform:
    def __init__(self):
        ...

    def fit(self, X_ref, y_true=None):
        num_col = []
        for col in X_ref.columns:
            if (X_ref[col].dtype in [np.int64, np.float64]) and len(
                X_ref[col].unique()
            ) > 20:
                num_col.append(col)
        if not num_col:
            raise ValueError(
                "No numerical columns detected.Please Pass Numerical Features"
            )
        return X_ref[num_col]
        ...

    def transform(self, X):
        return X


# TODO: this implements the check for a single feature, not multiple features. Should only need to update this to use list comprehensions for the simplest version.
class CategoricalCountTransform:
    def __init__(self):
        ...

    def fit(self, X_ref, y_true=None):
        cat_col = []
        for col in X_ref.columns:
            if (
                X_ref[col].dtype == "object" or X_ref[col].dtype.name == "category"
            ) or len(X_ref[col].unique()) <= 20:
                cat_col.append(col)
        if not cat_col:
            raise ValueError(
                "No categorical columns detected.Please Pass Categorical Features"
            )
        return X_ref[cat_col]

    def transform(self, X):
        return [np.unique(X[col].values, return_counts=True)[1] for col in X.columns]


class BinaryTransform:
    def __init__(self):
        ...

    def fit(self, X_ref, y_true=None):
        bin_col = []
        for col in X_ref.columns:
            if len(X_ref[col].unique()) == 2:
                bin_col.append(col)
        if not bin_col:
            raise ValueError("No binary columns detected.Please Pass Binary Features")
        return X_ref[bin_col]

    def transform(self, X):
        for col in X.columns:
            values = set(np.unique(X[col]))
            if not set(values).issubset(["0", "1", True, False]):
                raise ValueError(
                    "The `x_ref` data must consist of only (0,1)'s or (False,True)'s for the "
                    "FETDrift detector."
                )
        return X


# endregion


# region Hypothesis test wrappers
def chi2Divergence(rep_ref, rep_test):
    # what happens if there are missing categories? need to fill them in with 0's, probably
    pvals = []
    for ref_vals, test_vals in zip(rep_ref, rep_test):
        tab = np.vstack([ref_vals, test_vals])
        pvals.append(chi2_contingency(tab).pvalue)
    return pvals


def ksDivergence(rep_ref, rep_test):
    if isinstance(rep_ref, pd.DataFrame):
        rep_ref = rep_ref.values
    if isinstance(rep_test, pd.DataFrame):
        rep_test = rep_test.values
    pvals = [
        kstest(ref_vals, test_vals).pvalue
        for ref_vals, test_vals in zip(rep_ref.T, rep_test.T)
    ]
    return pvals


def cvmDivergence(rep_ref, rep_test):
    if isinstance(rep_ref, pd.DataFrame):
        rep_ref = rep_ref.values
    if isinstance(rep_test, pd.DataFrame):
        rep_test = rep_test.values
    pvals = [
        cramervonmises_2samp(ref_vals, test_vals).pvalue
        for ref_vals, test_vals in zip(rep_ref.T, rep_test.T)
    ]
    return pvals


def fetDivergence(ref, test, alternative):
    ref, test = ref.astype(np.int64), test.astype(np.int64)
    n_ref, n = ref.shape[0], test.shape[0]
    p_val, odds_ratio = np.empty(len(ref.columns)), np.empty(len(ref.columns))
    # TODO: where are sum_test, sum_ref defined?
    for f in range(len(ref.columns)):
        table = np.array(
            [[sum_test[f], sum_ref[f]], [n - sum_test[f], n_ref - sum_ref[f]]]
        )
        odds_ratio[f], p_val[f] = fisher_exact(table, alternative)

    return p_val


# endregion


# region Critical value functions
def bonf_crit(pvals, alpha=0.05):
    # handles the Bonf correction and working across multiple features
    # TODO: this bit with the alpha is a little inconvenient
    b_alpha = alpha / len(pvals)
    print(pvals)
    return any([pval <= b_alpha for pval in pvals])


# endregion


# region Implemented classes
class CHIDetector(GenericDetector):
    def __init__(self):
        super().__init__(
            representation=CategoricalCountTransform(),
            divergence=chi2Divergence,
            crit_function=bonf_crit,
        )


class KSDetector(GenericDetector):
    def __init__(self):
        super().__init__(
            representation=IdentityTransform(),
            divergence=ksDivergence,
            crit_function=bonf_crit,
        )


class CVMDetector(GenericDetector):
    def __init__(self):
        super().__init__(
            representation=IdentityTransform(),
            divergence=cvmDivergence,
            crit_function=bonf_crit,
        )


class FETDetector(GenericDetector):
    def __init__(self):
        super().__init__(
            representation=BinaryTransform(),
            divergence=fetDivergence,
            crit_function=bonf_crit,
        )


# endregion
