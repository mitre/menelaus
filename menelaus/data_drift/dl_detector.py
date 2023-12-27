from sklearn.preprocessing import StandardScaler
from scipy.stats import binomtest
from scipy.stats import ks_2samp
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from menelaus.detector import BatchDetector
import copy
import numpy as np
import pandas as pd


class DL_Detector(BatchDetector):
    """
    # TODO fix documentation

    DL Detector trains a Random Forest model to classify whether data belongs to the source or target domain and monitors output from the classifier to
    detect if drift is occurring between reference and test datasets.
    Using a cross-validation scheme, it obtains two sets of probability scores. The first set contains the probabilities that unseen data in the source domain belongs to the target domain. The second set contains the probabilities that unseen data in the target domain belongs to the target domain. If there is a significant difference in these probability scores, as determined by a statistical divergence test, it indicates samples in the target domain differ from samples in the source domain.

    This detector provides two options for drift detection:

        * Binomial Test:
            * Reference data assigned 'target' label of 0, test data assigned 'target' label of 1
            * Partitions shuffled reference and test data into two halves
            * Uses first half of both reference and test to train classifier to predict 'target', latter half to obtain test predictions
            * Conduct a binomial test on the number of successful predictions
            * If accuracy of of model is greater than random chance, indicates patterns in test data differ from patterns in reference data
            * Alert to drift

            * The 'pred' dataframe is accessible. It contains the classifier's predictions for reference and source data. The index from this dataset
              can be used to subset the original datasets to identify data in source that is classified as target.

        * KS test
            * This test considers each feature individually
            * Features in reference data assigned 'target' label of 0, test data assigned 'target' label of 1
            * Uses Stratified K Fold to train a classifier to predict 'target' and obtain probability scores for all samples in joint reference and test dataset
            * Conducts a 2 sample Kolmogorov-Smirnov test on the probability scores of reference data compared to probability scores of test data
            * A small pvalue from KS test indicates the distribution of probability scores are differet, alerts to drift
            * Bonferonni correction used to aggregate results from KS test on each feature. Bonferroni minimizes false positives and is a conservative approach to detecting drift.
            * A shift may occur in individiual features but may not be significant enough to alert for drift in entire dataset

            * If X_by_feauture is specified, the 'pred' dataframe contains the classifier's predictions for reference and source data,
             relative to the specified feature only. The index from this dataset can be used to subset the original datasets to identify data in source that is classified as target.


    Ref. CITE FAILING LOUDLY

    Attributes:

    """

    input_type = "batch"

    def __init__(
        self,
        n_splits=5,
        random_state=33,
        scaling=True,
        alpha=0.05,
        threshold=0.5,
    ):
        super().__init__()

        # Initialize parameters
        self.n_splits = n_splits
        self.random_state = random_state
        self.scaling = scaling
        self.alpha = alpha
        self.threshold = threshold

    def set_reference(self, X, y_true=None, y_pred=None):
        """
        Initialize detector with a reference batch. After every update, reference batch
        is automatically set to most recent test batch.

        Args:
            X (pandas.DataFrame): initial baseline dataset
            y_true (numpy.array): true labels for dataset - not used
            y_pred (numpy.array): predicted labels for dataset - not used
        """

        self.ref_idx = X.index
        X, _, _ = super()._validate_input(X, None, None)
        X = pd.DataFrame(X, columns=self._input_cols)

        # Initialize attributes
        self.reference = copy.deepcopy(X)
        self.reset()

    def reset(self):
        """
        If scaling and first batch, scale reference data. Save original indexes of data. Intended for use
        after reference batch is updated, regardless of ``drift_state``.
        """

        super().reset()

        if self.total_batches == 0 and self.scaling == True:
            scaler = StandardScaler()
            self.reference = pd.DataFrame(
                scaler.fit_transform(self.reference),
                columns=self._input_cols,
                index=self.ref_idx,
            )

    def update(self, X, by_feature=True, X_by_feature=None, y_true=None, y_pred=None):
        """
        Update the detector with a new test batch. After update, new test batch will be set as next reference
        batch.

        Args:
            X (DataFrame): next batch of data to detect drift on.
            y_true (numpy.ndarray): true labels of next batch - not used
            y_pred (numpy.ndarray): predicted labels of next batch - not used
        """

        self.reset()

        X_idx = X.index
        X, _, _ = super()._validate_input(X, None, None)
        X = pd.DataFrame(X, columns=self._input_cols)

        super().update(X, None, None)

        if self.scaling == True:
            scaler = StandardScaler()
            X = pd.DataFrame(
                scaler.fit_transform(X), columns=self._input_cols, index=X_idx
            )

        if by_feature == True:
            self.feature_statistics = {}
            self._pvalues = []

            for f in self._input_cols:
                f_drift_state = None

                X_joined, y_joined = self._join_data(
                    pd.DataFrame(self.reference.loc[:, f]), pd.DataFrame(X.loc[:, f])
                )
                pred = self._obtain_proba(X_joined, y_joined)
                test_statistic, pvalue = self._ks_test(
                    pred
                )  # TODO consider a generic divergence test?

                # TODO add PSI as an option?
                # could try using PSI here for each feature
                # PSI < 0.1 is no shift, between 0.1 and 0.25 moderate shift, over 0.25 significant shift
                # or do not bring it in here and use this in another detector?

                if X_by_feature == f:
                    self.pred = pred

                if pvalue <= self.alpha:
                    f_drift_state = "drift"

                self.feature_statistics[f] = {
                    "Statistic": test_statistic,
                    "Pvalue": pvalue,
                    "Drift": f_drift_state,
                }
                self._pvalues.append(pvalue)

            # Bonferonni correction to detect full dataset shift
            bonferonni_alpha = self.alpha / len(self._pvalues)
            for p in self._pvalues:
                if p <= bonferonni_alpha:
                    self.drift_state = "drift"

            self.drift_statistics = {
                "Statistic": test_statistic,
                "Pvalue": pvalue,
                "Drift": self.drift_state,
            }

        else:
            test_statistic, pvalue, self.pred = self._binomial_test(self.reference, X)

            if pvalue <= self.alpha:
                self.drift_state = "drift"

            self.drift_statistics = {
                "Statistic": test_statistic,
                "Pvalue": pvalue,
                "Drift": self.drift_state,
            }

        self.reference = X

    def _join_data(self, reference, test):
        """
        Joins reference and test data, assigning target variable to data source.

        Args:
            reference (pandas.DataFrame): data belonging to reference domain
            test (pandas.DataFrame): data belonging to test domain

        Returns:
            X_joined (pandas.DataFrame): dataset containing combined reference and test data
            y_joined (numpy.array): true labels for data source
        """

        X_reference = reference.copy()
        X_test = test.copy()
        X_reference.loc[:, "target"] = 0
        X_test.loc[:, "target"] = 1
        X_joined = pd.concat([X_reference, X_test])
        y_joined = X_joined["target"]
        X_joined = X_joined.drop(columns=["target"], axis=1)

        return X_joined, y_joined

    def _ks_test(self, pred):
        """
        Conducts Kolmogorov-Smirnov test on predicted probabilities between reference and test data.
        Obtains test statistic and corresponding pvalue

        Args:
            pred (pandas.DataFrame): dataset containing true labels, predicted probabilities, and predicted labels

        Returns:
            test_statistic (float): KS test statistic
            pvalue (float): KS pvalue

        """

        reference = pred[pred["y_true"] == 0]
        test = pred[pred["y_true"] == 1]

        ks = ks_2samp(reference.proba, test.proba)
        test_statistic = round(ks.statistic, 4)
        pvalue = round(ks.pvalue, 4)

        return test_statistic, pvalue

    def _binomial_test(self, reference, test):
        """
        Assigns 'target' outcome variable. Partitions shuffled data into two halves. Uses first half to train classifier, second
        half to test. Conducts Binomial test on number of correctly assigned labels for source of data.
        Obtains test statistic and corresponding pvalue

        Args:
            pred (pandas.DataFrame): dataset containing true labels, predicted probabilities, and predicted labels

        Returns:
            test_statistic (float): Binomial test statistic
            pvalue (float): Binomial pvalue

        """
        X_reference = reference.copy()
        X_test = test.copy()
        X_reference.loc[:, "target"] = 0
        X_test.loc[:, "target"] = 1

        X_reference = X_reference.sample(frac=1)
        train_source = X_reference.iloc[0 : int(len(X_reference) / 2),]
        test_source = X_reference[~X_reference.index.isin(train_source)]

        X_test = X_test.sample(frac=1)
        train_target = X_test.iloc[0 : int(len(X_test) / 2),]
        test_target = X_test[~X_test.index.isin(train_target)]

        X_train = pd.concat([train_source, train_target], axis=0, ignore_index=False)
        y_train = X_train.target
        X_train = X_train.drop(columns=["target"], axis=1)
        X_test = pd.concat([test_source, test_target], axis=0, ignore_index=False)
        y_test = X_test.target
        X_test = X_test.drop(columns=["target"], axis=1)

        clf = RandomForestClassifier(max_depth=2)
        clf.fit(X_train, y_train)

        pred = pd.DataFrame(
            {"y_true": y_test, "proba": clf.predict_proba(X_test)[:, 1]}
        )

        pred["y_pred"] = np.where(pred["proba"] >= 0.5, 1, 0)

        success_n = len(pred[pred.y_true == pred.y_pred])
        b = binomtest(success_n, n=len(pred), p=0.5, alternative="greater")
        test_statistic = round(b.proportion_estimate, 4)
        pvalue = round(b.pvalue, 4)

        return test_statistic, pvalue, pred

    def _obtain_proba(self, X_joined, y_joined):
        """
        Fits a Random Forest model to classify data source. Uses cross validation
        to obtain and store predicted probabilities of label on all data.

        Args:
            X_joined (pandas.DataFrame): dataset containing combined reference and test data
            y_joined (numpy.array): true labels for data source

        Returns:
            pred (pandas.DataFrame): dataset containing true labels, predicted probabilities, and predicted labels

        """

        pred = pd.DataFrame(columns=["y_true", "proba"])

        CV = StratifiedKFold(
            n_splits=self.n_splits, shuffle=True, random_state=self.random_state
        )
        for i, (train_index, test_index) in enumerate(CV.split(X_joined, y_joined)):
            clf = RandomForestClassifier(max_depth=2)
            clf.fit(X_joined.iloc[train_index], y_joined.iloc[train_index])

            pred_i = pd.DataFrame(
                {
                    "y_true": y_joined.iloc[test_index],
                    "proba": clf.predict_proba(X_joined.iloc[test_index])[:, 1],
                }
            )
            pred = pd.concat([pred, pred_i], axis=0, ignore_index=False)

        pred["y_pred"] = np.where(pred["proba"] >= self.threshold, 1, 0)

        return pred
