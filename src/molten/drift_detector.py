class DriftDetector:
    """Base class for MOLTEN drift detectors."""

    def __init__(self, *args, **kwargs):
        """
        Attributes:
            total_samples (int): number of samples the drift detector has ever
                been updated with
            samples_since_reset (int): number of samples since the last time the
                drift detector was reset
            drift_state (str): detector's current drift state. Can take values
                "drift", "warning", or None.
        """
        super().__init__()
        self.total_samples = 0
        self.samples_since_reset = 0
        self._drift_state = None

    def update(self, *args, **kwargs):
        """Update the detector with a new sample."""
        self.total_samples += 1
        self.samples_since_reset += 1

    def reset(self, *args, **kwargs):
        """Initialize the detector's drift state and other relevant attributes.
        Intended for use after drift_state == 'drift'."""
        self.samples_since_reset = (
            0  # number of elements the detector has been updated with since last reset
        )
        self._drift_state = None

    @property
    def drift_state(self):
        """
        Returns:
            str: detector's current drift state, with values "drift", "warning",
                or None.
        """
        return self._drift_state

    @drift_state.setter
    def drift_state(self, value):
        """Set detector's drift state to "drift", "warning", or None.

        Args:
            value (str): "drift", "warning", or None

        Raises:
            ValueError: raised if disallowed value is given
        """
        if not value in ("drift", "warning", None):
            raise ValueError(
                'DriftDetector._drift_state must be "drift", "warning", or None.'
            )
        else:
            self._drift_state = value
