"""Methods for testing <logical flow> of PCA-CD."""
import pandas as pd
import numpy as np
from menelaus.data_drift.pca_cd import PCACD


def test_build_kl():
    """
    For KL: tests setup construction of reference and test windows, scaling of data, PC projection, and density estimation
    """

    # Setup data
    np.random.seed(1)
    size = 150
    col1 = np.random.randint(1, 10, size)
    col2 = np.random.uniform(1, 2, size)
    col3 = np.random.normal(0, 1, size)
    reference = pd.DataFrame(data=[col1, col2, col3]).T

    # Setup detector

    window_size = 50
    det = PCACD(window_size=window_size, divergence_metric="kl")

    # Run Detector

    # Construct reference window
    reference_size = 0
    for i in range(window_size):
        det.update(reference.iloc[[i]])
        reference_size += 1
        assert len(det._reference_window) == reference_size
        assert det.drift_state is None
    assert len(det._test_window) == 0
    assert det._build_reference_and_test is True
    assert det._density_reference == {}

    # Construct test window
    test_size = 0
    for i in range(window_size, window_size * 2):
        det.update(reference.iloc[[i]])
        test_size += 1
        assert len(det._test_window) == test_size
        assert det.drift_state is None

    # Test projection of PCs
    assert len(det._reference_pca_projection) == reference_size
    assert len(det._test_pca_projection) == test_size

    # Test estimation of densities
    assert det._build_reference_and_test is False
    assert det._density_reference != {}


def test_reset():
    """
    For drift: tests reset() functionality
    """

    # Setup data
    np.random.seed(1)
    size = 150
    col1 = np.random.randint(1, 10, size)
    col2 = np.random.uniform(1, 2, size)
    col3 = np.random.normal(0, 1, size)
    reference = pd.DataFrame(data=[col1, col2, col3]).T

    # adding drift
    col1_ = np.random.uniform(9, 10, size)
    col2_ = np.random.normal(1, 3, size)
    col3_ = np.random.randint(20, 30, size)
    drift = pd.DataFrame(data=[col1_, col2_, col3_]).T

    # Setup detector

    window_size = 50
    det = PCACD(window_size=window_size, divergence_metric="kl", delta=0.05)

    # Run Detector

    # Update with reference distribution
    for i in range(len(reference)):
        det.update(reference.iloc[[i]])
    assert det._drift_state is None

    # Update with drifted distribution
    for i in range(len(drift)):
        det.update(drift.iloc[[i]])
        if det.drift_state is not None:
            break

    # Update with additional sample, test reset
    det.update(drift.iloc[[i]])
    assert det._build_reference_and_test is True
    assert det._drift_state is None
    assert det.samples_since_reset == 0


def test_kl_no_drift():
    """
    For KL with no drift: tests update and detection of drift
    """

    # Setup data
    np.random.seed(1)
    size = 150
    col1 = np.random.randint(1, 10, size)
    col2 = np.random.uniform(1, 2, size)
    col3 = np.random.normal(0, 1, size)
    reference = pd.DataFrame(data=[col1, col2, col3]).T

    # Setup detector

    window_size = 50
    det = PCACD(window_size=window_size, divergence_metric="kl")

    # Run Detector

    # Construct reference and test windows
    for i in range(window_size * 2):
        det.update(reference.iloc[[i]])
    assert det._drift_state is None

    # Add test samples, still from reference distribution
    step = 0.05 * window_size
    update_size = 0
    for i in range(window_size * 2, len(reference)):
        det.update(reference.iloc[[i]])

        assert len(det._test_window) == window_size
        assert len(det._test_pca_projection) == window_size

        update_size += 1
        if (window_size * 2 + update_size) % step == 0:
            assert det._density_test != {}
            assert len(det._change_score) != 0

    assert det.drift_state is None


def test_kl_drift():
    """
    For KL with drift: tests update and detection of drift
    """

    # Setup data
    np.random.seed(1)
    size = 150
    col1 = np.random.randint(1, 10, size)
    col2 = np.random.uniform(1, 2, size)
    col3 = np.random.normal(0, 1, size)
    reference = pd.DataFrame(data=[col1, col2, col3]).T

    # adding drift
    col1_ = np.random.uniform(9, 10, size)
    col2_ = np.random.normal(1, 3, size)
    col3_ = np.random.randint(
        20, 30, size
    )  # categorical data may not be appropriate for PCA-CD
    drift = pd.DataFrame(data=[col1_, col2_, col3_]).T

    # Setup detector

    window_size = 50
    det = PCACD(window_size=window_size, divergence_metric="kl", delta=0.05)

    # Run Detector

    # Update with reference distribution
    for i in range(len(reference)):
        det.update(reference.iloc[[i]])
    assert det._drift_state is None

    # Update with drifted distribution
    step = 0.05 * window_size
    update_size = 0
    for i in range(len(drift)):
        det.update(drift.iloc[[i]])

        assert len(det._test_window) == window_size
        assert len(det._test_pca_projection) == window_size

        update_size += 1
        if (window_size * 2 + update_size) % step == 0:
            assert det._density_test != {}
            assert len(det._change_score) != 0

        if det.drift_state is not None:
            break

    assert det.drift_state is not None


def test_build_intersection():
    """
    For intersection: tests setup and density estimation
    """

    # Setup data
    np.random.seed(1)
    size = 150
    col1 = np.random.randint(1, 10, size)
    col2 = np.random.uniform(1, 2, size)
    col3 = np.random.normal(0, 1, size)
    reference = pd.DataFrame(data=[col1, col2, col3]).T

    # Setup detector

    window_size = 50
    det = PCACD(window_size=window_size, divergence_metric="intersection")

    # Run Detector

    # Construct reference window
    for i in range(window_size):
        det.update(reference.iloc[[i]])
    assert det._build_reference_and_test is True
    assert det._density_reference == {}

    # Construct test window
    for i in range(window_size, window_size * 2):
        det.update(reference.iloc[[i]])

    # Test estimation of densities
    assert det._build_reference_and_test is False
    assert det._density_reference != {}


def test_intersection_no_drift():
    """
    For intersection with no drift: tests update and detection of drift
    """

    # Setup data
    np.random.seed(1)
    size = 150
    col1 = np.random.randint(1, 10, size)
    col2 = np.random.uniform(1, 2, size)
    col3 = np.random.normal(0, 1, size)
    reference = pd.DataFrame(data=[col1, col2, col3]).T

    # Setup detector

    window_size = 50
    det = PCACD(window_size=window_size, divergence_metric="intersection")

    # Run Detector

    # Construct reference and test windows
    for i in range(window_size * 2):
        det.update(reference.iloc[[i]])
    assert det._drift_state is None

    # Add test samples, still from reference distribution
    step = 0.05 * window_size
    update_size = 0
    for i in range(window_size * 2, len(reference)):
        det.update(reference.iloc[[i]])

        assert len(det._test_window) == window_size
        assert len(det._test_pca_projection) == window_size

        update_size += 1
        if (window_size * 2 + update_size) % step == 0:
            assert det._density_test != {}
            assert len(det._change_score) != 0

    assert det.drift_state is None


def test_intersection_drift():
    """
    For intersection with drift: tests update and detection of drift
    """

    # Setup data
    np.random.seed(1)
    size = 150
    col1 = np.random.normal(1, 10, size)
    col2 = np.random.uniform(1, 2, size)
    col3 = np.random.normal(0, 1, size)
    reference = pd.DataFrame(data=[col1, col2, col3]).T

    # adding drift
    col1_ = np.random.normal(9, 10, size)
    col2_ = np.random.normal(1, 3, size)
    col3_ = np.random.randint(20, 30, size)
    drift = pd.DataFrame(data=[col1_, col2_, col3_]).T

    # Setup detector

    window_size = 50
    det = PCACD(window_size=window_size, divergence_metric="intersection")

    # Run Detector

    # Update with reference distribution
    for i in range(len(reference)):
        det.update(reference.iloc[[i]])
    assert det._drift_state is None

    # Add test samples, from drifted data
    step = 0.05 * window_size
    update_size = 0
    for i in range(len(drift)):
        det.update(drift.iloc[[i]])

        assert len(det._test_window) == window_size
        assert len(det._test_pca_projection) == window_size

        update_size += 1
        if (window_size * 2 + update_size) % step == 0:
            assert det._density_test != {}
            assert len(det._change_score) != 0

        if det.drift_state is not None:
            break

    assert det.drift_state is not None
