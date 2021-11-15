class DriftDetector:
    """Base class for MOLTEN drift detectors."""

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.total_samples = (
            0  # number of samples the drift detector has ever been updated with
        )
        self.n = 0  # number of samples since the last time the drift detector was reset
        self._drift_state = None

    def update(self, *args, **kwargs):
        """Update the detector with a new sample."""
        self.total_samples += 1
        self.n += 1

    def reset(self, *args, **kwargs):
        """Initialize the detector's drift state and other relevant attributes. Intended for use after drift_state == 'drift'."""
        self.n = (
            0  # number of elements the detector has been updated with since last reset
        )
        self._drift_state = None  # detector's current drift state, with values "drift", "warning", or None.

    @property
    def drift_state(self):
        """ """
        return self._drift_state

    @drift_state.setter
    def drift_state(self, value):
        """"""
        if not value in ("drift", "warning", None):
            raise ValueError(
                'DriftDetector._drift_state must be "drift", "warning", or None.'
            )
        else:
            self._drift_state = value
