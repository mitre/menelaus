import numpy as np
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

    def __init__(self, clf):
        """
        Args:
            clf (sklearn.svm.SVC): the classifier for which we are tracking drift.
                For now, assumed to be a Support Vector Machine (SVM). TODO: change
                this after adding capability for other models.
        """

        super().__init__()
        self.classifier = clf
        self.process_svm()

    def process_svm(self):
        # get the separating hyperplane
        self.w = np.array(self.clf.coef_[0])
        self.b = -self.w[0] / self.w[1]

        # calculate the magnitude of the margin
        self.margin = 1 / np.sqrt(np.sum(self.clf.coef_**2))

    # TODO: two ways we can handle the update/drift detection:
    #       (1) after the detector has been updated with some number of samples
    #           that the user can pass in (maybe 30), then we start looking for drift
    #       (2) the user sets the detector to have a reference batch, and then passes
    #           in samples one at a time to update the detector
    def set_reference(self, reference_batch):
        """
        Initialize detector with a reference batch. Reference batch must be
        manually set and updated by user using this method. Reference batch
        is not automatically updated after a drift is detected.

        Args:
            reference_batch (DataFrame): initial baseline dataset
        """

        self.reference_batch = reference_batch
        # TODO: in the formula for the forgetting factor in the paper, is N
        # the total number of samples so far, or the size of the reference batch?
        self.forgetting_factor = (len(reference_batch) - 1) / len(reference_batch)
        self.reference_margin_density = self.calculate_margin_density(reference_batch)
        self.curr_margin_density = self.reference_margin_density

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
        # TODO: keep implementing the algorithm from here

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
    