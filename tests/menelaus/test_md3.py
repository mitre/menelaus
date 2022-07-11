"""Methods for checking simple behaviors of MD3."""
import pytest
import numpy as np
import pandas as pd
from sklearn import svm
from menelaus.concept_drift.md3 import MD3


def test_init():
    """Confirm MD3 parameters are initialized as expected"""
    clf = svm.SVC(kernel='linear')
    det = MD3(clf=clf)
    assert det.classifier == clf
    assert det.sensitivity == 2
    assert det.k == 10
    assert det.oracle_data_length_required is None
    assert det.oracle_data is None
    assert det.waiting_for_oracle == False
    
    
def test_set_reference():
    """Confirm that the set_reference method behaves as expected,
    calculating the correct distribution statistics"""
    np.random.seed(123)
    data = [[1, 2, 0], [1, 1, 0], [5, 6, 1], [6, 7, 1], [3, 4, 1]]
    df = pd.DataFrame(data, columns=['var1', 'var2', 'y'])
    X_train = df.loc[:, ['var1', 'var2']]
    y_train = df.loc[:, 'y']
    clf = svm.SVC(kernel='linear')
    clf.fit(X_train, y_train)
    det = MD3(clf=clf, k=2)
    det.set_reference(df, 'y')
    assert X_train.equals(det.reference_batch_features)
    assert df.loc[:, df.columns == 'y'].equals(det.reference_batch_target)
    assert det.reference_distribution["len"] == len(data)
    assert det.reference_distribution["md"] == 0.0
    assert det.reference_distribution["md_std"] == 0.0
    assert np.abs(det.reference_distribution["acc"] - 0.833) < 0.1
    assert np.abs(det.reference_distribution["acc_std"] - 0.166) < 0.1
    assert det.oracle_data_length_required == len(data)
    assert det.forgetting_factor == (len(data) - 1) / len(data)
    
    
def test_update():
    """Confirm that the update method, including validation, behaves
    as expected"""
    np.random.seed(123)
    data = [[1, 1, 0], [2, 2, 0], [3, 4, 1], [5, 6, 1], [6, 7, 1]]
    df = pd.DataFrame(data, columns=['var1', 'var2', 'y'])
    X_train = df.loc[:, ['var1', 'var2']]
    y_train = df.loc[:, 'y']
    clf = svm.SVC(kernel='linear')
    clf.fit(X_train, y_train)
    det = MD3(clf=clf, k=2)
    det.set_reference(df, 'y')
    
    data_update1 = [[-4, 8], [3, 4]]
    df_update1 = pd.DataFrame(data_update1, columns=['var1', 'var2'])
    with pytest.raises(ValueError) as _:
        det.update(df_update1)
    
    data_update2 = [[2.5, 3.5]]
    df_update2 = pd.DataFrame(data_update2, columns=['var1', 'var2'])
    det.update(df_update2)
    assert det.drift_state == "warning"
    assert det.waiting_for_oracle == True
    
    with pytest.raises(ValueError) as _:
        det.update(df_update2)
    

def test_give_oracle_label():
    """Confirm that providing a stream of labeled samples to the detector
    with give_oracle_label behaves as expected"""
    np.random.seed(123)
    data = [[1, 1, 0], [2, 2, 0], [3, 4, 1], [5, 6, 1], [6, 7, 1]]
    df = pd.DataFrame(data, columns=['var1', 'var2', 'y'])
    X_train = df.loc[:, ['var1', 'var2']]
    y_train = df.loc[:, 'y']
    clf = svm.SVC(kernel='linear')
    clf.fit(X_train, y_train)
    det = MD3(clf=clf, k=2)
    det.set_reference(df, 'y')
    
    labeled_sample = [[1, 2, 0]]
    labeled_df = pd.DataFrame(labeled_sample, columns=['var1', 'var2', 'y'])
    with pytest.raises(ValueError) as _:
        det.give_oracle_label(labeled_df)
    
    data_update1 = [[-4, 8], [3, 4]]
    df_update1 = pd.DataFrame(data_update1, columns=['var1', 'var2'])
    with pytest.raises(ValueError) as _:
        det.update(df_update1)
    
    data_update2 = [[2.5, 3.5]]
    df_update2 = pd.DataFrame(data_update2, columns=['var1', 'var2'])
    det.update(df_update2)
    assert det.drift_state == "warning"
    assert det.waiting_for_oracle == True
    
    with pytest.raises(ValueError) as _:
        det.update(df_update2)
        
    double_labeled_sample = [[1, 2, 0], [7, 8, 1]]
    double_labeled_df = pd.DataFrame(double_labeled_sample, columns=['var1', 'var2', 'y'])
    with pytest.raises(ValueError) as _:
        det.give_oracle_label(double_labeled_df)
        
    wrong_labeled_sample = [[1, 2, 0], [7, 8, 1]]
    wrong_labeled_df = pd.DataFrame(wrong_labeled_sample, columns=['var1', 'var2', 'target'])
    with pytest.raises(ValueError) as _:
        det.give_oracle_label(wrong_labeled_df)
        
    wrong_len_sample = [[1, 0], [7, 1]]
    wrong_len_df = pd.DataFrame(wrong_len_sample, columns=['var1', 'target'])
    with pytest.raises(ValueError) as _:
        det.give_oracle_label(wrong_len_df)
        
    labeled_sample = [[1, 2, 0]]
    labeled_df = pd.DataFrame(labeled_sample, columns=['var1', 'var2', 'y'])
    det.give_oracle_label(labeled_df)
    assert det.drift_state is None
    assert labeled_df.equals(det.oracle_data)
        
    for i in range(len(data) - 1):
        if i % 2 == 0:
            labeled_sample = [[6.5, 7.5, 1]]
            labeled_df = pd.DataFrame(labeled_sample, columns=['var1', 'var2', 'y'])
            det.give_oracle_label(labeled_df)
            assert det.drift_state is None
            if i == len(data) - 2:
                assert det.oracle_data is None
            else:
                assert len(det.oracle_data) == i + 2
        else:
            labeled_sample = [[1.5, 2.5, 0]]
            labeled_df = pd.DataFrame(labeled_sample, columns=['var1', 'var2', 'y'])
            det.give_oracle_label(labeled_df)
            assert det.drift_state is None
            if i == len(data) - 2:
                assert det.oracle_data is None
            else:
                assert len(det.oracle_data) == i + 2
