from menelaus.change_detection import adwin


class ADWIN(adwin.ADWIN):
    def update(self, y_true, y_pred, X=None):
        """Update the detector with a new sample.

        Args:
          y_true: actual class of next sample
          y_pred: predicted class of next sample
          X: next sample in the stream of data - not used in ADWIN (concept drift)
        """

        new_value = int(y_true == y_pred)

        # This class is here to avoid asking the user to provide such a direct
        # function of (y_true, y_pred) in the X argument, which is unintuitive.
        super().update(new_value, y_true=None, y_pred=None)
