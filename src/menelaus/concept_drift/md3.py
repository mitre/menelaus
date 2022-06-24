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
        super().__init__()
        self.classifier = clf

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

    def update(self, new_sample):
        """
        Update the detector with a new sample.

        Args:
            new_sample (DataFrame): feature values/sample data for the new incoming sample
        """

        margin_inclusion_signal = self.calculate_margin_inclusion_signal(new_sample)

    def calculate_margin_inclusion_signal(self, sample):
        """
        Calculate the value of the margin inclusion signal for an incoming sample that
        the detector is being updated with. If the sample lies in the margin of the
        classifier, then a value of 1 is returned for the margin inclusion signal.
        Otherwise, a value of 0 is returned.

        Args:
            sample (DataFrame): feature values/sample data for the new incoming sample
        """

        # TODO: Write this function
        return 1
