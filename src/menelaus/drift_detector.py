from abc import ABC, abstractmethod


class DriftDetector(ABC):
    """Base class for Menelaus drift detectors.
    A DriftDetector object implements the ``update`` and ``reset`` methods and
    calls the ``super`` methods to initialize and update the attributes below.

    Generally, a DriftDetector is instantiated, then repeatedly passed new data
    via ``update``. At each ``update`` step, its ``drift_state`` will reflect
    whether drift has been detected or almost been detected. After the
    detector's state is set to ``"drift"``, ``update`` calls ``reset`` to
    re-initialize the relevant attributes.

    A "batch" detector will compare a new dataset, passed via ``update``, to a
    reference dataset, usually the original reference dataset. A "stream"
    detector compares only one new sample at a time, also passed via ``update``.

    Attributes:
        total_samples (int): number of samples/batches the drift detector
            has ever been updated with
        samples_since_reset (int): number of samples/batches since the last
            time the drift detector was reset

    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.total_samples = 0
        self.samples_since_reset = 0
        self._drift_state = None
        self._input_type = None

    @abstractmethod
    def update(self, *args, **kwargs):
        """Update the detector with a new sample or batch."""
        self.total_samples += 1
        self.samples_since_reset += 1

    @abstractmethod
    def reset(self, *args, **kwargs):
        """Initialize the detector's drift state and other relevant attributes.
        Intended for use after ``drift_state == 'drift'``."""
        self.samples_since_reset = (
            0  # number of elements the detector has been updated with since last reset
        )
        self.drift_state = None

    @property
    def drift_state(self):
        """Detector's current drift state, with values ``"drift"``, ``"warning"``,or
        ``None``.
        """
        return self._drift_state

    @drift_state.setter
    def drift_state(self, value):
        """Set detector's drift state to ``"drift"``, ``"warning"``, or ``None``.

        Args:
            value (str): ``"drift"``, ``"warning"``, or ``None``

        Raises:
            ValueError: raised if disallowed value is given
        """
        if not value in ("drift", "warning", None):
            raise ValueError(
                'DriftDetector._drift_state must be ``"drift"``, ``"warning"``, or ``None``.'
            )
        else:
            self._drift_state = value

    @property
    @abstractmethod
    def _input_type(self):
        """The type of input the detector accepts, either ``"batch"``, with multiple
        samples in one call to update(), or ``"stream"``, with one sample per call
        to update().
        """
        return self._input_type
