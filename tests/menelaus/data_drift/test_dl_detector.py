import numpy as np
import pandas as pd
import pytest
from menelaus.data_drift.dl_detector import DL_Detector
from sklearn.preprocessing import StandardScaler

rows = 10
columns = 5
def test_dl_detector_init():
    """Test correct default initialization for dl_detector"""
    det = DL_Detector()
    assert det.n_splits == 5
    assert det.random_state == 33
    assert det.scaling == True
    assert det.alpha == 0.05
    assert det.threshold == 0.5


def test_dl_detector_set_reference():
    """Assert dl_detector.set_reference works as intended"""
    det = DL_Detector()
    scaler = StandardScaler()
    ref = pd.DataFrame(
        np.ones((rows, columns)), columns=[f"Column_{i}" for i in range(columns)]
    )
    det.set_reference(ref)
    scale_ref = pd.DataFrame(scaler.fit_transform(ref), columns=ref.columns, index=ref.index)
    assert det.reference == scale_ref


def test_dl_detector_update_1():
    """Ensure dl_detector can return no drift accurately"""
    det = DL_Detector()
    ref = pd.DataFrame(
        np.ones((rows, columns)), columns=[f"Column_{i}" for i in range(columns)]
    )
    det.set_reference(ref)
    det.update(ref.replace(1, 2))
    assert det.drift_state is None


def test_dl_detector_update_2():
    """Ensure dl_detector can update with drift actions triggered"""
    det = DL_Detector()
    ref = pd.DataFrame(
        np.ones((rows, columns)), columns=[f"Column_{i}" for i in range(columns)]
    )
    test = pd.DataFrame(
        np.random.randint(0, 100, size=(rows, columns)),
        columns=[f"Column_{i}" for i in range(columns)],
    )
    det.set_reference(ref)
    det.update(test)
    assert det.drift_state is not None


def test_dl_detector_update_3():
    """Ensure dl_detector can update with drift actions triggered when by_feature equal False"""
    det = DL_Detector()
    ref = pd.DataFrame(
        np.ones((rows, columns)), columns=[f"Column_{i}" for i in range(columns)]
    )
    test = pd.DataFrame(
        np.random.randint(0, 100, size=(rows, columns)),
        columns=[f"Column_{i}" for i in range(columns)],
    )
    det.set_reference(ref)
    det.update(test, by_feature=False)
    assert det.drift_state is not None


def test_dl_detector_update_4():
    """Check dl_detector.update behavior after drift alarm"""
    det = DL_Detector()
    ref = pd.DataFrame(
        np.ones((rows, columns)), columns=[f"Column_{i}" for i in range(columns)]
    )
    det.set_reference(ref)
    det._drift_state = "drift"
    det.update(ref.replace(1, 2))
    assert det.drift_state is None
