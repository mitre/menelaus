from abc import ABC, abstractmethod
from typing import List
from toolz import pipe


class Ensembler(ABC):
    def __init__(self, detectors: List, pipeline: List):
        self.detectors = detectors
        self.pipeline = pipeline


class StreamingDetector(ABC):
    """
    Abstract base class for all streaming data-based detectors.
    Minimally implements abstract methods common to all stream
    based detection algorithms.
    """

    def __init__(self, *args, **kwargs):
        self._total_samples = 0
        self._samples_since_reset = 0
        self._drift_state = None

    @abstractmethod
    def update(self, X, y_true, y_pred):
        """
        Update detector with new sample (data point).

        Args:
            X (numpy.ndarray): input data
            y_true (numpy.ndarray): if applicable, true labels of input data
            y_pred (numpy.ndarray): if applicable, predicted labels of input data
        """
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
    def total_samples(self):
        """Total number of samples the drift detector has been updated with.

        Returns:
            int
        """
        return self._total_samples

    @total_samples.setter
    def total_samples(self, value):
        self._total_samples = value

    @property
    def samples_since_reset(self):
        """Number of samples since last drift detection.

        Returns:
            int
        """
        return self._samples_since_reset

    @samples_since_reset.setter
    def samples_since_reset(self, value):
        self._samples_since_reset = value

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
    """

    def __init__(self, *args, **kwargs):
        self._total_batches = 0
        self._batches_since_reset = 0
        self._drift_state = None

    @abstractmethod
    def update(self, X, y_true, y_pred):
        """
        Update detector with new batch of data

        Args:
            X (numpy.ndarray): input data
            y_true (numpy.ndarray): if applicable, true labels of input data
            y_pred (numpy.ndarray): if applicable, predicted labels of input data
        """
        self.total_batches += 1
        self.batches_since_reset += 1

    @abstractmethod
    def set_reference(self, X, y_true, y_pred):
        """
        Initialize detector with a reference batch.

        Args:
            X (pandas.DataFrame or numpy.array): baseline dataset
            y_true (numpy.array): actual labels of dataset
            y_pred (numpy.array): predicted labels of dataset
        """
        raise NotImplementedError

    @abstractmethod
    def reset(self, *args, **kwargs):
        """
        Initialize the detector's drift state and other relevant attributes.
        Intended for use after ``drift_state == 'drift'``.
        """
        self.batches_since_reset = 0
        self.drift_state = None

    @property
    def total_batches(self):
        """Total number of batches the drift detector has been updated with.

        Returns:
            int
        """
        return self._total_batches

    @total_batches.setter
    def total_batches(self, value):
        self._total_batches = value

    @property
    def batches_since_reset(self):
        """Number of batches since last drift detection.

        Returns:
            int
        """
        return self._batches_since_reset

    @batches_since_reset.setter
    def batches_since_reset(self, value):
        self._batches_since_reset = value

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


class BatchEnsembler(BatchDetector, Ensembler):
    def __init__(self, detectors: List, pipeline: List):
        BatchDetector.__init__(self)
        Ensembler.__init__(self, detectors, pipeline)

    def update(self, X, y_true, y_pred):
        self = pipe(self, *self.pipeline) # how does it use extra args? read docs
        return self

    def reset(self):
        BatchDetector.reset(self)

    def set_reference(self):
        # really not sure what to do here
        pass

    # other properties? inherited automatically?
    # I think "operator" functions should go in class files in detector code


##############################
# To Be Deprecated in 0.2.0+
##############################
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
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        self._total_updates = 0
        self._updates_since_reset = 0
        self._drift_state = None
        self._input_type = None

    @abstractmethod
    def update(self, X, y_true, y_pred):
        """
        Update the detector with a new sample or batch.

        Args:
            X (numpy.ndarray): input data
            y_true (numpy.ndarray): if applicable, true labels of input data
            y_pred (numpy.ndarray): if applicable, predicted labels of input data
        """
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
    def total_updates(self):
        """Number of samples/batches the drift detector has ever been updated
        with.

        Returns:
            int
        """
        return self._total_updates

    @total_updates.setter
    def total_updates(self, value):
        self._total_updates = value

    @property
    def updates_since_reset(self):
        """Number of samples/batches since the last time the drift detector was
        reset.

        Returns:
            int
        """
        return self._updates_since_reset

    @updates_since_reset.setter
    def updates_since_reset(self, value):
        self._updates_since_reset = value

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
