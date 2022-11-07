import copy
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.base import clone
from menelaus.detector import DriftDetector


class MD3(DriftDetector):
    """The Margin Density Drift Detection (MD3) method is a drift detection
    algorithm that alarms based on the cumulative tracking of the number of
    samples in the margin, the uncertainty region of a classifier. Tracking
    samples that fall in the margin is an unsupervised task, as no true
    labels are required. However, this can lead to more common false alarms.

    To counter this, MD3 has an initial drift warning step based on Margin
    Density, and then confirms or rules out drift based on accuracy of
    predictions on a labeled dataset that is accumulated from the "Oracle",
    or directly from the data stream.

    Margin Density (MD): "The expected number of data samples that fall within
    a robust classifier's (one that distributes importance weights among its
    features) region of uncertainty, i.e. its margin." :cite:t:`sethi2017reliable`

    Because the MD metric is essentially the total number of samples that fall
    with the margin divided by the total number of samples in the set,
    its value is in the range of [0, 1].

    Ref. :cite:t:`sethi2017reliable`
    """

    input_type = "stream"

    def calculate_margin_inclusion_signal(self, sample, clf):
        """
        Calculate the value of the margin inclusion signal for an incoming sample that
        the detector is being updated with. Uses the classifier passed in for this
        margin calculation. If the sample lies in the margin of the classifier, then
        a value of 1 is returned for the margin inclusion signal. Otherwise, a value
        of 0 is returned.

        Args:
            sample (numpy.array): feature values/sample data for the new incoming sample
            clf (sklearn.svm.SVC): the classifier for which we are calculating margin
                inclusion signal.
        """

        w = np.array(clf.coef_[0])
        intercept = np.array(clf.intercept_)
        b = intercept[0] / w[1]

        mis = np.abs(np.dot(w, sample) + b)

        if mis <= 1:
            return 1
        else:
            return 0

    def __init__(
        self,
        clf,
        margin_calculation_function=calculate_margin_inclusion_signal,
        sensitivity=2,
        k=10,
        oracle_data_length_required=None,
    ):
        """
        Args:
            clf (classifier): the classifier for which we are tracking drift. If
                classifier is not of type sklearn.svm.svc, a ``margin_calculation_function``
                must be passed in for appropriate margin signal tracking.
            margin_calculation_function (function): the appropriate margin signal
                function for the classifier. Takes in two arguments: (1) an incoming
                sample of size 1 as a numpy array and (2) the classifier for this
                detector. Should return 1 if the sample falls in the margin of the
                classifier, 0 if not. Defaults to the
                ``calculate_margin_inclusion_signal`` function, which is designed
                specifically for an sklearn.svm.SVC classifier.
            sensitivity (float): the sensitivity at which a change in margin density
                will be detected. Change is signaled when the margin density at a
                time t, given by MD_t, deviates by more than ``sensitivity``
                standard deviations from the reference margin density value MD_Ref.
                A larger value can be set if frequent signaling is not desired.
                Alternatively, a lower value could be used for applications
                where small changes could be harmful, if undetected. Defaults to 2.
            k (int): the number of folds that will be used in k-fold cross validation
                when measuring the distribution statistics of the reference batch
                of data. Defaults to 10.
            oracle_data_length_required (int): the number of samples that will need to
                be collected by the oracle when drift is suspected, for the purpose of
                either confirming or ruling out drift, and then retraining the classifier
                if drift is confirmed. Defaults to the length of the reference distribution
                (this is set in the set_reference method).
        """

        super().__init__()
        self.classifier = clf
        self.margin_calculation_function = margin_calculation_function
        self.sensitivity = sensitivity
        self.k = k
        self.oracle_data_length_required = oracle_data_length_required
        self.oracle_data = None
        self.waiting_for_oracle = False

    def set_reference(self, X, y_true=None, y_pred=None, target_name=None):
        """
        Initialize detector with a reference batch. Reference batch must be
        manually set and updated by user using this method. Reference batch
        is not automatically updated after a drift is detected.

        Args:
            reference_batch (pandas.DataFrame): initial baseline dataset
            y_true (numpy.array): true labels of dataset - not used in MD3
            y_pred (numpy.array): predicted labels of dataset - not used in MD3
            target_name (string): name of the column in the reference_batch
                dataframe which is the target variable
        """

        self.reference_batch_features = copy.deepcopy(
            X.loc[:, X.columns != target_name]
        )
        self.reference_batch_target = copy.deepcopy(X.loc[:, X.columns == target_name])

        self.reference_distribution = self.calculate_distribution_statistics(X)

        if self.oracle_data_length_required is None:
            self.oracle_data_length_required = self.reference_distribution["len"]

        self.forgetting_factor = (
            self.reference_distribution["len"] - 1
        ) / self.reference_distribution["len"]
        self.curr_margin_density = self.reference_distribution["md"]

    def calculate_distribution_statistics(self, data):
        """
        Calculate the following five statistics for the data distribution
        passed in:

            1. Length/Number of Samples (len)

            2. Margin Density (md)

            3. Standard Deviation of Margin Density (md_std)

            4. Accuracy (acc)

            5. Standard Deivation of Accuracy (acc_std)

        Args:
            data (DataFrame): batch of data to calculate distribution
                statistics for
        """

        duplicate_classifier = clone(self.classifier)

        # prepare the cross-validation procedure
        margin_densities = []
        accuracies = []
        cv = KFold(n_splits=self.k, random_state=42, shuffle=True)

        # perform k-fold cross validation to acquire distribution margin density and acuracy values
        for train_index, test_index in cv.split(self.reference_batch_features):
            X_train, X_test = (
                self.reference_batch_features.iloc[train_index],
                self.reference_batch_features.iloc[test_index],
            )
            y_train, y_test = (
                self.reference_batch_target.iloc[train_index],
                self.reference_batch_target.iloc[test_index],
            )

            duplicate_classifier.fit(X_train, y_train.values.ravel())

            # record margin inclusion signals for all samples in this test band
            signal_func_values = []
            for i in range(len(X_test)):
                sample_np_array = X_test.iloc[i].to_numpy()
                margin_inclusion_signal = self.margin_calculation_function(
                    self, sample_np_array, duplicate_classifier
                )
                signal_func_values.append(margin_inclusion_signal)

            # record margin density over this test band
            margin_density = sum(signal_func_values) / len(signal_func_values)
            margin_densities.append(margin_density)

            # record accuracy of prediction over this test band
            y_pred = duplicate_classifier.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            accuracies.append(accuracy)

        # record performance - margin density
        md = np.mean(margin_densities)
        md_std = np.std(margin_densities)

        # record performance - accuracy
        acc = np.mean(accuracies)
        acc_std = np.std(accuracies)

        # return reference distribution statistics
        return {
            "len": len(data),
            "md": md,
            "md_std": md_std,
            "acc": acc,
            "acc_std": acc_std,
        }

    def update(self, X, y_true=None, y_pred=None):
        """
        Update the detector with a new sample.

        Args:
            X (DataFrame): feature values/sample data for the new incoming sample
            y_true (numpy.array): true label of new sample - not used in MD3
            y_pred (numpy.array): predicted label of new sample - not used in MD3
        """

        if self.waiting_for_oracle == True:
            raise ValueError(
                """give_oracle_label method must be called to provide detector with a
                labeled sample to confirm or rule out drift."""
            )

        if len(X) != 1:
            raise ValueError(
                """This method is only available for data inputs in the form of 
                a Pandas DataFrame with exactly 1 record."""
            )

        if self.drift_state == "drift":
            self.reset()

        super().update(X, y_true, y_pred)

        sample_np_array = X.to_numpy()[0]
        margin_inclusion_signal = self.margin_calculation_function(
            self, sample_np_array, self.classifier
        )
        self.curr_margin_density = (
            self.forgetting_factor * self.curr_margin_density
            + (1 - self.forgetting_factor) * margin_inclusion_signal
        )

        warning_level = np.abs(
            self.curr_margin_density - self.reference_distribution["md"]
        )
        warning_threshold = self.sensitivity * self.reference_distribution["md_std"]

        if warning_level > warning_threshold:
            self.drift_state = "warning"
            self.waiting_for_oracle = True

    def give_oracle_label(self, labeled_sample):
        """
        Provide the detector with a labeled sample to confirm or rule out drift. Once a
        certain number of samples is accumulated, drift can be confirmed or ruled out. If
        drift is confirmed, retraining will be initiated using these samples, and the
        reference distribution will be updated accordingly.

        Args:
            labeled_sample (DataFrame): labeled data sample
        """

        if self.waiting_for_oracle != True:
            raise ValueError(
                """give_oracle_label method can be called only when a drift warning has
                been issued and drift needs to be confirmed or ruled out."""
            )

        if len(labeled_sample) != 1:
            raise ValueError(
                """This method is only available for data inputs in the form of 
                a Pandas DataFrame with exactly 1 record."""
            )

        labeled_columns = list(labeled_sample.columns)
        feature_columns = list(self.reference_batch_features.columns)
        target_column = list(self.reference_batch_target.columns)
        reference_columns = feature_columns + target_column
        if len(labeled_columns) != len(reference_columns) or set(
            labeled_columns
        ) != set(reference_columns):
            raise ValueError(
                """give_oracle_label method can be called only with a sample containing
                the same number and names of columns as the original reference distribution."""
            )

        self.drift_state = None

        if self.oracle_data is None:
            self.oracle_data = labeled_sample
        else:
            self.oracle_data = pd.concat(
                [self.oracle_data, labeled_sample], ignore_index=True
            )

        if len(self.oracle_data) == self.oracle_data_length_required:
            X_test, y_test = (
                self.oracle_data[feature_columns],
                self.oracle_data[target_column],
            )
            y_pred = self.classifier.predict(X_test)
            acc_labeled_samples = accuracy_score(y_test, y_pred)

            drift_level = self.reference_distribution["acc"] - acc_labeled_samples
            drift_threshold = self.sensitivity * self.reference_distribution["acc_std"]

            if drift_level > drift_threshold:
                self.drift_state = "drift"

            # update reference distribution
            self.set_reference(self.oracle_data, target_name=target_column[0])
            self.oracle_data = None
            self.waiting_for_oracle = False

    def reset(self):
        """
        Initialize the detector's drift state and other relevant attributes.
        Intended for use after ``drift_state == 'drift'``.
        """
        super().reset()
        self.curr_margin_density = self.reference_distribution["md"]
