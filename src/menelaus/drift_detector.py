from abc import ABC, abstractmethod


class StreamingDetector(ABC):
    """
    Abstract base class for all streaming data-based detectors.
    Minimally implements abstract methods common to all stream
    based detection algorithms.

    Attributes:
        total_samples (int): total number of samples the drift
            detector has been updated with
        samples_since_reset (int): number of samples since last
            drift detection
        drift_state (str): detector's current drift state, can
            take ``str`` values ``"drift"``, ``"warning"``, or ``None``.
    """

    def __init__(self, *args, **kwargs):
        self.total_samples = 0
        self.samples_since_reset = 0
        self._drift_state = None

    @abstractmethod
    def update(self, *args, **kwargs):
        """Update detector with new sample (data point)."""
        self.total_samples += 1
        self.samples_since_reset += 1

    @abstractmethod
    def reset(self, *args, **kwargs):
        """
        Initialize the detector's drift state and other relevant attributes.
        Intended for use after ``drift_state == "drift"``.
        """
        self.samples_since_reset = 0
        self.drift_state = None

    @property
    def drift_state(self):
        """Set detector's drift state to ``"drift"``, ``"warning"``, or ``None``."""
        return self._drift_state

    @drift_state.setter
    def drift_state(self, value):
        """Set detector's drift state to ``"drift"``, ``"warning"``, or ``None``.

        Args:
            value (str): ``"drift"``, ``"warning"``, or ``None``

        Raises:
            ValueError: raised if disallowed value is given
        """
        if value not in ("drift", "warning", None):
            raise ValueError("tbd")
        else:
            self._drift_state = value


class BatchDetector(ABC):
    """
    Abstract base class for all batch data-based detectors.
    Minimally implements abstract methods common to all batch
    based detection algorithms.

    Attributes:
        total_batches (int): total number of batches the drift
            detector has been updated with
        batches_since_reset (int): number of batches since last
            drift detection
        drift_state (str): detector's current drift state, can
            take ``str`` values ``"drift"``, ``"warning"``, or ``None``.
    """

    def __init__(self, *args, **kwargs):
        self.total_batches = 0
        self.batches_since_reset = 0
        self._drift_state = None

    @abstractmethod
    def update(self, *args, **kwargs):
        """Update detector with new batch of data"""
        self.total_batches += 1
        self.batches_since_reset += 1

    @abstractmethod
    def reset(self, *args, **kwargs):
        """
        Initialize the detector's drift state and other relevant attributes.
        Intended for use after ``drift_state == 'drift'``.
        """
        self.batches_since_reset = 0
        self.drift_state = None

    @property
    def drift_state(self):
        """Set detector's drift state to ``"drift"``, ``"warning"``, or ``None``."""
        return self._drift_state

    @drift_state.setter
    def drift_state(self, value):
        """Set detector's drift state to ``"drift"``, ``"warning"``, or ``None``.

        Args:
            value (str): ``"drift"``, ``"warning"``, or ``None``

        Raises:
            ValueError: raised if disallowed value is given
        """
        if value not in ("drift", "warning", None):
            raise ValueError("tbd")
        else:
            self._drift_state = value


#######################
# Deprecated in 0.2.0
#######################
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
        total_updates (int): number of samples/batches the drift detector
            has ever been updated with
        updates_since_reset (int): number of samples/batches since the last
            time the drift detector was reset
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.total_updates = 0
        self.updates_since_reset = 0
        self._drift_state = None
        self._input_type = None

    @abstractmethod
    def update(self, *args, **kwargs):
        """Update the detector with a new sample or batch."""
        self.total_updates += 1
        self.updates_since_reset += 1

    @abstractmethod
    def reset(self, *args, **kwargs):
        """Initialize the detector's drift state and other relevant attributes.
        Intended for use after ``drift_state == 'drift'``."""
        self.updates_since_reset = (
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
        if value not in ("drift", "warning", None):
            raise ValueError(
                """DriftDetector._drift_state must be ``"drift"``, 
                ``"warning"``, or ``None``."""
            )
        else:
            self._drift_state = value

    @property
    @abstractmethod
    def input_type(self):
        """The type of input the detector accepts, either ``"batch"``, with multiple
        samples in one call to update(), or ``"stream"``, with one sample per call
        to update().
        """
        return self._input_type
