"""All methods for testing DetectA implementation."""
import numpy as np
import pandas as pd
import pytest
from menelaus.data_drift.detectA import DetectA


def test_update_labels_validation():
    """Test ability to raise error if labels are passed in containing new labels
    """
    np.random.seed(0)
    ref = pd.DataFrame({'a': np.random.normal(3, 2.5, size=500),
                         'b': np.random.normal(6, 4, size=500)})
    ref_label = pd.DataFrame({'label': np.random.randint(0, 2, size = 500)})
    test = pd.DataFrame({'a': np.random.normal(3, 2.5, size=500),
                            'b': np.random.normal(6, 4, size=500)})
    test_label = pd.DataFrame({'label': np.random.randint(3,5, size = 500)})
                         
    det = DetectA()
    det.set_reference(ref, ref_label)

    with pytest.raises(ValueError) as _:
        _ = det.update(test, test_label)

def test_set_reference_validation():
    """Test ability to raise error if y_true is None and k is not specified 
    """
    np.random.seed(0)
    ref = pd.DataFrame({'a': np.random.normal(3, 2.5, size=500),
                        'b': np.random.normal(6, 4, size=500)})
                     
    det = DetectA()
    with pytest.raises(ValueError) as _:
        _ = det.set_reference(ref, y_true = None)

def test_set_reference_no_labels():
    """Test setting reference with no labels passed in
    """
    np.random.seed(0)
    ref = pd.DataFrame({'a': np.random.normal(3, 2.5, size=500),
                         'b': np.random.normal(6, 4, size=500)})
                 
    det = DetectA()
    det.set_reference(ref, y_true = None, k = 2)

    assert det.init_labels is None
    assert det.y_pred is not None 
    assert len(np.unique(det.y_pred)) == 2
  

def test_set_reference_labels():
    """Test setting reference with labels passed in
    """
    np.random.seed(0)
    ref = pd.DataFrame({'a': np.random.normal(3, 2.5, size=500),
                         'b': np.random.normal(6, 4, size=500)})
    ref_label = pd.DataFrame({'label': np.random.randint(0, 2, size = 500)})
    det = DetectA()
    det.set_reference(ref, y_true = ref_label)
    num_labels = len(np.unique(ref_label))

    assert det.init_labels is not None
    assert len(np.unique(det.init_labels)) == num_labels
    assert det.y_pred is not None 
    assert len(np.unique(det.y_pred))  == num_labels
    

def test_update_no_labels_no_drift():
    """Test update with no labels passed in and drift is not detected
    """
    np.random.seed(0)
    ref = pd.DataFrame({'a': np.random.normal(3, 2.5, size=500),
                         'b': np.random.normal(6, 4, size=500)})
    ref_label = pd.DataFrame({'label': np.random.randint(0, 2, size = 500)})
    
    det = DetectA()
    det.set_reference(ref, ref_label)
    det.update(ref)
    assert det.drift_state is None

    det.update(ref)
    assert det.drift_state is None
  

def test_update_labels_no_drift():
    """Test update with labels passed in and drift is not detected
    """
    np.random.seed(0)
    ref = pd.DataFrame({'a': np.random.normal(3, 2.5, size=500),
                         'b': np.random.normal(6, 4, size=500)})
    ref_label = pd.DataFrame({'label': np.random.randint(0, 2, size = 500)})
    
    det = DetectA()
    det.set_reference(ref, y_true = ref_label)
    det.update(ref, ref_label)
    assert det.drift_state is None
    
    det.update(ref, ref_label)
    assert det.drift_state is None
    
    

def test_update_no_labels_drift():
    """Test update with no labels passed in and drift is detected
    """
    np.random.seed(0)
    ref = pd.DataFrame({'a': np.random.normal(3, 2.5, size=500),
                         'b': np.random.normal(6, 4, size=500)})
    ref_label = pd.DataFrame({'label': np.random.randint(0, 2, size = 500)})
    test = pd.DataFrame({'a': np.random.normal(4, 2.5, size=500),
                         'b': np.random.normal(7, 4, size=500)})
                         
    det = DetectA()
    det.set_reference(ref, ref_label)
    det.update(test)
    assert det.drift_state == 'drift'
   

def test_update_labels_drift():
    """Test update with labels passed in and drift is detected
    """
    np.random.seed(0)
    ref = pd.DataFrame({'a': np.random.normal(3, 2.5, size=500),
                         'b': np.random.normal(6, 4, size=500)})
    ref_label = pd.DataFrame({'label': np.random.randint(0, 2, size = 500)})
    test = pd.DataFrame({'a': np.random.normal(4, 2.5, size=500),
                         'b': np.random.normal(7, 4, size=500)})
    test_label = pd.DataFrame({'label': np.random.randint(0,2, size = 500)})
                         
    det = DetectA()
    det.set_reference(ref, ref_label)
    det.update(test, test_label)
    assert det.drift_state == 'drift'

    det.update(test, test_label)
    assert det.total_batches == 2 
    assert det.batches_since_reset == 1
    assert det.drift_state == None

def test_reset_after_drift():
    """Test reset functionality after drift is detected
    """
    np.random.seed(0)
    ref = pd.DataFrame({'a': np.random.normal(3, 2.5, size=500),
                         'b': np.random.normal(6, 4, size=500)})
    ref_label = pd.DataFrame({'label': np.random.randint(0, 2, size = 500)})
    test = pd.DataFrame({'a': np.random.normal(10, 10, size=500),
                         'b': np.random.normal(17, 8, size=500)})
                         
    det = DetectA()
    det.set_reference(ref, ref_label)
    det.update(test)
    assert det.drift_state == 'drift'

    det.update(test)
    assert det.total_batches == 2 
    assert det.batches_since_reset == 1
    assert det.drift_state == None
