import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from menelaus.drift_detector import DriftDetector

class MD3(DriftDetector):
    """The Margin Density Drift Detection (MD3) method is a drift detection
    algorithm based on the number of samples in the uncertainty region
    of a classifier, intended for an online classifier.

    The MD3 algorithm is a distribution independent, application independent,
    and model independent method for reliably detecting drifts from data
    streams.

    Margin Density (MD): The expected number of data samples that fall within
    a robust classifier's (one that distributes importance weights among its
    features) region of uncertainty, i.e. its margin.

    The MD metric, being a ratio, has its value in the range of [0, 1].

    TODO: This initial implementation assumes that the classifier being used is a
    Support Vector Machine (SVM) because SVMs have explicit margins defined
    by the hyperplane. Classifiers such as decision trees and K-nearest
    neighbors return discrete class labels and do not have any intuitive
    notion of margin. To implement compatibility with MD3 for these
    classifiers, a feater bagging ensemble technique must be used (this will
    be added to the MD3 implementation later on). For now, this algorithm
    is designed specifically for SVMs (and potentially logistic regression) 
    for the sake of simplicity.

    Attributes:
        total_updates (int): number of samples the drift detector has ever
            been updated with
        updates_since_reset (int): number of samples since the last time the
            drift detector was reset
        drift_state (str): detector's current drift state. Can take values
            ``"drift"``, ``"warning"``, or ``None``.
    """

    input_type = "stream"

    def __init__(self, clf, sensitivity=2, k=10):
        """
        Args:
            clf (sklearn.svm.SVC): the classifier for which we are tracking drift.
                For now, assumed to be a Support Vector Machine (SVM). TODO: change
                this after adding capability for other models.
            sensitivity (float): the sensitivity at which a change in margin density
                will be detected. Change is signaled when the margin density at a
                time t, given by MD sub t, deviates by more than sensitivity 
                standard deviations from the reference margin density value MD sub Ref.
                A larger value can be set if frequent signaling is not desired, 
                alternatively a lower value could be used for critical applications
                where small changes could be harmful, if undetected. Defaults to 2. 
                TODO: suggested by paper to be picked in the range of [0, 3]
            k (int): the number of folds that will be used in k-fold cross validation
                when measuring the distribution statistics of the reference batch
                of data. Defaults to 10.
        """

        super().__init__()
        self.classifier = clf
        self.sensitivity = sensitivity
        self.k = k
        self.process_svm()

    def process_svm(self):
        # TODO: make sure these calculations are correct
        # get the separating hyperplane
        # The decision boundary is the line y = a*x - b
        # Sources:
        #    (1) MD3 Paper
        #    (2) https://scikit-learn.org/stable/auto_examples/svm/plot_svm_margin.html#sphx-glr-download-auto-examples-svm-plot-svm-margin-py
        #    (3) https://scikit-learn.org/stable/auto_examples/svm/plot_separating_hyperplane.html#example-svm-plot-separating-hyperplane-py
        #    (4) https://stackoverflow.com/questions/23794277/extract-decision-boundary-with-scikit-learn-linear-svm
        self.w = np.array(self.classifier.coef_[0])
        self.intercept = np.array(self.classifier.intercept_)
        self.a = -self.w[0] / self.w[1]
        self.b = self.intercept[0] / self.w[1]

        # calculate the magnitude of the margin
        self.margin = 1 / np.sqrt(np.sum(self.classifier.coef_**2))

    # TODO: two ways we can handle the update/drift detection:
    #       (1) after the detector has been updated with some number of samples
    #           that the user can pass in (maybe 30), then we start looking for drift
    #       (2) the user sets the detector to have a reference batch, and then passes
    #           in samples one at a time to update the detector
    def set_reference(self, reference_batch, target_name):
        """
        Initialize detector with a reference batch. Reference batch must be
        manually set and updated by user using this method. Reference batch
        is not automatically updated after a drift is detected.

        Args:
            reference_batch (DataFrame): initial baseline dataset
            target_name (string): name of the column in the reference_batch
                dataframe which is the target variable
        """

        self.reference_batch_features = reference_batch.loc[:, reference_batch.columns != target_name]
        self.reference_batch_target = reference_batch[target_name]
        # TODO: in the formula for the forgetting factor in the paper, is N
        # the total number of samples so far, or the size of the reference batch?
        self.forgetting_factor = (len(reference_batch) - 1) / len(reference_batch)
        self.reference_distribution = self.calculate_distribution_statistics(reference_batch)
        self.curr_margin_density = self.reference_distribution["md"]

    def calculate_distribution_statistics(self, data):
        """
        Calculate the following four statistics for the data distribution
        passed in:
            1. Margin Density (md)
            2. Standard Deviation of Margin Density (md_std)
            3. Accuracy (acc)
            4. Standard Deivation of Accuracy (acc_std)

        Args:
            data (DataFrame): batch of data to calculate distribution
                statistics for
        """

        duplicate_classifier = self.classifier

        # prepare the cross-validation procedure
        margin_densities = []
        accuracies = []
        # TODO: set random state here or no?
        cv = KFold(n_splits=self.k, random_state=1, shuffle=True)
        # perform k-fold cross validation to acquire distribution margin density and acuracy values
        for train_index, test_index in cv.split(self.reference_batch_features):
            X_train, X_test = self.reference_batch_features[train_index], self.reference_batch_features[test_index]
            y_train, y_test = self.reference_batch_target[train_index], self.reference_batch_target[test_index]
            duplicate_classifier.fit(X_train, y_train)

            # record margin inclusion signals for all samples in this test band
            signal_func_values = []
            for i in range(len(X_test)):
                sample_np_array = X_test.loc[i, :].to_numpy()
                margin_inclusion_signal = self.calculate_margin_inclusion_signal_other_classifier(sample_np_array, duplicate_classifier)
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
            "md": md,
            "md_std": md_std,
            "acc": acc,
            "acc_std": acc_std
        }

    def calculate_margin_density(self, data):
        """
        Calculate the total margin density of the batch of data passed in.

        Args:
            data (DataFrame): batch of data to calculate margin density for
        """

        signal_func = 0
        for i in range(len(data)):
            sample_np_array = data.loc[i, :].to_numpy()
            margin_inclusion_signal = self.calculate_margin_inclusion_signal(sample_np_array)
            signal_func += margin_inclusion_signal
        
        return signal_func / len(data)

    def update(self, new_sample):
        """
        Update the detector with a new sample.

        Args:
            new_sample (DataFrame): feature values/sample data for the new incoming sample
        """

        if len(new_sample) != 1:
            raise ValueError(
                """This method is only available for data inputs in the form of 
                a Pandas DataFrame with exactly 1 record."""
            )

        sample_np_array = new_sample.loc[0, :].to_numpy()
        margin_inclusion_signal = self.calculate_margin_inclusion_signal(sample_np_array)
        self.curr_margin_density = self.forgetting_factor * self.curr_margin_density + (1 - self.forgetting_factor) * margin_inclusion_signal
        
        if np.abs(self.curr_margin_density - self.reference_distribution["md"]) > self.sensitivity * self.reference_distribution["md_std"]:
            self.drift_state = "warning"

        # TODO: continue implementing algorithm from here
        # Next step is to collect labeled samples to confirm that drift is occurring

    def reset(self):
        """
        Initialize the detector's drift state and other relevant attributes.
        Intended for use after ``drift_state == 'drift'``.
        """
        super().reset()
        self.curr_margin_density = self.reference_margin_density

    def calculate_margin_inclusion_signal(self, sample):
        """
        Calculate the value of the margin inclusion signal for an incoming sample that
        the detector is being updated with. If the sample lies in the margin of the
        classifier, then a value of 1 is returned for the margin inclusion signal.
        Otherwise, a value of 0 is returned.

        Args:
            sample (numpy.array): feature values/sample data for the new incoming sample
        """

        if np.abs(np.dot(self.w, sample) + self.b) <= 1:
            return 1
        else:
            return 0

    def calculate_margin_inclusion_signal_other_classifier(self, sample, clf):
        """
        Calculate the value of the margin inclusion signal for an incoming sample that
        the detector is being updated with. Uses the classifier passed in rather than
        the classifier stored in the detector. If the sample lies in the margin of the
        classifier, then a value of 1 is returned for the margin inclusion signal.
        Otherwise, a value of 0 is returned.

        Args:
            sample (numpy.array): feature values/sample data for the new incoming sample
            clf (sklearn.svm.SVC): the classifier for which we are calculating margin
                inclusion signal. TODO: add compatibility with other types of models.
        """

        # TODO: make sure calculations for decision boundary here are correct (like above)
        w = np.array(clf.coef_[0])
        intercept = np.array(clf.intercept_)
        b = intercept[0] / w[1]

        if np.abs(np.dot(w, sample) + b) <= 1:
            return 1
        else:
            return 0
