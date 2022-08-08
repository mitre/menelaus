from abc import ABC, abstractmethod
from pandas import DataFrame
import numpy as np
import copy


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
        self.input_cols = None
        self.input_col_dim = None

    @abstractmethod
    def update(self, X, y_true, y_pred):
        """
        Update detector with new sample (data point).

        Args:
            X (numpy.ndarray): input data
            y_true (numpy.ndarray): if applicable, true labels of input data
            y_pred (numpy.ndarray): if applicable, predicted labels of input data
        """
        self._validate_input(X, y_true, y_pred)
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

    def _validate_X(self, X):
        """Validate that the input only contains one observation, and that its
        dimensions/column names match earlier input. If there is no
        earlier input, store the dimension/column names.

        Args:
            X (array-like or numeric): Input features.

        Raises:
            ValueError: if a dataframe has ever been passed, raised if X's
                column names don't match
            ValueError: if an array has ever been passed, raised if X's number
                of columns don't match
            ValueError: raised if X contains more than one observation
        """
        ary = copy.copy(X)
        if isinstance(ary, DataFrame):
            # The first update with a dataframe will constrain subsequent input.
            if self.input_cols is None:
                self.input_cols = ary.columns
                self.input_col_dim = len(self.input_cols)
            elif self.input_cols is not None:
                if not ary.columns.equals(self.input_cols):
                    raise ValueError(
                        "Columns of new data must match with columns of prior data."
                    )
        else:
            ary = np.array(ary)
            if len(ary.shape) == 0:
                ary = ary.reshape(-1, 1)
            elif len(ary.shape) == 1:
                ary = ary.reshape(1, -1)
            if self.input_col_dim is None:
                # This allows starting with a dataframe, then later passing bare
                # numpy arrays. For now, assume users are not miscreants.
                self.input_col_dim = ary.shape[1]
            elif self.input_col_dim is not None:
                if ary.shape[1] != self.input_col_dim:
                    raise ValueError(
                        "Column-dimension of new data must match prior data."
                    )

        if ary.shape[0] != 1:
            raise ValueError(
                "Input for streaming detectors should contain only one observation."
            )

    def _validate_y(self, y):
        """Validate that input contains only one observation.

        Args:
            y (numeric): the current value for `y_true` or `y_pred`, given to
                `update`.

        Raises:
            ValueError: raised if more than one observation is passed.
        """
        ary = np.array(y).ravel()
        if ary.shape != (1,):
            raise ValueError(
                "Input for streaming detectors should contain only one observation."
            )

    def _validate_input(self, X, y_true, y_pred):
        """Helper method for `update`. Validates whether the input is appropriate
        for a streaming detector. Errors will be raised if the input is more
        than one observation, or if its dimensions don't match subsequent input.

        Args:
            X (numpy.ndarray): input data
            y_true (numpy.ndarray): if applicable, true labels of input data
            y_pred (numpy.ndarray): if applicable, predicted labels of input data
        """
        if X is not None:
            self._validate_X(X)
        if y_true is not None:
            self._validate_y(y_true)
        if y_pred is not None:
            self._validate_y(y_pred)

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
