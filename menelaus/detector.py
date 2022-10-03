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
        self._input_cols = None
        self._input_col_dim = None

    @abstractmethod
    def update(self, X, y_true, y_pred):
        """
        Update detector with new sample (data point).

        Args:
            X (numpy.ndarray): if applicable, one row of features from input data
            y_true (numpy.ndarray): if applicable, one true label from input data
            y_pred (numpy.ndarray): if applicable, one predicted label from input data
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

    def _validate_X(self, X):
        """Validate that the input only contains one observation, and that its
        dimensions/column names match earlier input. If there is no
        earlier input, store the dimension/column names.

        Args:
            X (array-like or numeric): One row from input features.

        Raises:
            ValueError: if a dataframe has ever been passed, raised if X's
                column names don't match
            ValueError: if an array has ever been passed, raised if X's number
                of columns don't match
            ValueError: raised if X contains more than one observation after coercion
        """
        if isinstance(X, DataFrame):
            # The first update with a dataframe will constrain subsequent input.
            if self._input_cols is None:
                self._input_cols = X.columns
                self._input_col_dim = len(self._input_cols)
            elif self._input_cols is not None:
                if not X.columns.equals(self._input_cols):
                    raise ValueError(
                        "Columns of new data must match with columns of prior data."
                    )
            ary = X.values
        else:
            ary = copy.copy(X)
            ary = np.array(ary)
            if len(ary.shape) <= 1:
                # only one sample should be passed, so coerce column vectors (e.g. pd.Series) to rows
                ary = ary.reshape(1, -1)
            if self._input_col_dim is None:
                # This allows starting with a dataframe, then later passing bare
                # numpy arrays. For now, assume users are not miscreants.
                self._input_col_dim = ary.shape[1]
            elif self._input_col_dim is not None:
                if ary.shape[1] != self._input_col_dim:
                    raise ValueError(
                        "Column-dimension of new data must match prior data."
                    )

        if ary.shape[0] != 1:
            raise ValueError(
                "Input for streaming detectors should contain only one observation."
            )
        return ary

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
        return ary

    def _validate_input(self, X, y_true, y_pred):
        """Helper method for `update`. Validates whether the input is appropriate
        for a streaming detector. Errors will be raised if the input is more
        than one observation, or if X's dimensions don't match prior input.

        Args:
            X (numpy.ndarray): input data
            y_true (numpy.ndarray): if applicable, one true label from input data
            y_pred (numpy.ndarray): if applicable, one predicted label from input data
        """
        if X is not None:
            X = self._validate_X(X)
        if y_true is not None:
            y_true = self._validate_y(y_true)
        if y_pred is not None:
            y_pred = self._validate_y(y_pred)
        return X, y_true, y_pred

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
        self._input_cols = None
        self._input_col_dim = None

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
            ValueError: if only one sample has been passed
        """
        if isinstance(X, DataFrame):
            # The first update with a dataframe will constrain subsequent input.
            if self._input_cols is None:
                self._input_cols = X.columns
                self._input_col_dim = len(self._input_cols)
            elif self._input_cols is not None:
                if not X.columns.equals(self._input_cols):
                    raise ValueError(
                        "Columns of new data must match with columns of prior data."
                    )
            ary = X.values
        else:
            ary = copy.copy(X)
            ary = np.array(ary)
            if len(ary.shape) <= 1:
                # Batch size of 1 will break downstream - don't allow it.
                # Attempts to coerce a row vector into a column vector.
                ary = ary.reshape(-1, 1)
            if self._input_col_dim is None:
                # This allows starting with a dataframe, then later passing bare
                # numpy arrays. For now, assume users are not miscreants.
                self._input_col_dim = ary.shape[1]
            elif self._input_col_dim is not None:
                if ary.shape[1] != self._input_col_dim:
                    raise ValueError(
                        "Column-dimension of new data must match prior data."
                    )
        if ary.shape[0] <= 1:
            raise ValueError(
                "Input for batch detectors should contain more than one observation."
            )
        return ary

    def _validate_y(self, y):
        """Validate that input contains only one column.

        Args:
            y (numeric): the current value for `y_true` or `y_pred`, given to
                `update`.

        Raises:
            ValueError: if an array has been passed that has more than one column
        """
        ary = np.array(y)
        if len(ary.shape) <= 1:
            ary = ary.reshape(1, -1)
        if ary.shape[0] == 1:
            raise ValueError(
                "Input for batch detectors should contain more than one obsevation."
            )
        if ary.shape[1] != 1:
            raise ValueError("y input for detectors should contain only one column.")
        return ary

    def _validate_input(self, X, y_true, y_pred):
        """Helper method for `update` and `set_reference`. Validates whether the
        input is appropriate for a batch detector. Errors will be raised if X's
        dimensions don't match prior input, or a y input has more than one
        column.

        Args:
            X (numpy.ndarray): input data
            y_true (numpy.ndarray): if applicable, true labels of input data
            y_pred (numpy.ndarray): if applicable, predicted labels of input data
        """
        if X is not None:
            X = self._validate_X(X)
        if y_true is not None:
            y_true = self._validate_y(y_true)
        if y_pred is not None:
            y_pred = self._validate_y(y_pred)
        return X, y_true, y_pred

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


class DriftDetector(ABC):
    """
    This class is deprecated in 0.2.0+.

    Base class for Menelaus drift detectors.

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
