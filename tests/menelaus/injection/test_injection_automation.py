import numpy as np
import pandas as pd
from menelaus.injection.injection_automation import InjectionTesting


def test_brownian_noise():
    df = pd.DataFrame(np.random.rand(100, 5), columns=["a", "b", "c", "d", "e"])
    tester = InjectionTesting(df)
    start = 0.75
    end = 1

    col = tester.inject_random_brownian_noise(50, start=start, end=end, num_drift_cols=1)
    std_normal = (tester.df.iloc[0 : int(start * len(df)), ][col].std().iloc[0, ])
    std_drift = (tester.df.iloc[int(start * len(df)) + 1:int(end * len(df)), ][col].std().iloc[0, ])

    assert std_drift > std_normal


def test_class_manipulation():
    df = pd.DataFrame(np.random.choice(a=["x", "y", "z"], size=100, p=[0.4, 0.3, 0.3]))
    swap_tester = InjectionTesting(df)
    join_tester = InjectionTesting(df)
    start = 0
    end = 1

    cols, all_swap_classes = swap_tester.inject_random_class_manipulation(
        manipulation_type="class_swap", start=start, end=end
    )
    col = cols[0]
    swap_classes = all_swap_classes[0]

    assert len(df[df[col] == swap_classes[0]]) == len(swap_tester.df[swap_tester.df[col] == swap_classes[1]])
    assert len(df[df[col] == swap_classes[1]]) == len(swap_tester.df[swap_tester.df[col] == swap_classes[0]])

    cols, all_join_classes = join_tester.inject_random_class_manipulation(manipulation_type="class_join", start=start, end=end)
    col = cols[0]
    join_classes = all_join_classes[0]

    assert len(join_tester.df[join_tester.df[col] == join_classes[0]]) == 0
    assert len(join_tester.df[join_tester.df[col] == join_classes[1]]) == 0


def test_feature_swap():
    df = pd.DataFrame()
    df['a'] = [0] * 100
    df['b'] = [1] * 100
    tester = InjectionTesting(df)
    start = 0.75
    end = 1

    tester.inject_random_feature_swap(start=start, end=end)
    assert(tester.df['a'].sum() == 25)
    assert(tester.df['b'].sum() == 75)


def test_feature_hide_and_sample():
    df = pd.DataFrame()
    df['a'] = np.random.choice(a=["x", "y", "z"], size=100, p=[0.4, 0.3, 0.3])
    df['b'] = np.random.rand(100, 1)
    tester = InjectionTesting(df)

    tester.inject_random_feature_hide_and_sample()
    assert(len(tester.df) < len(df))


def test_detectors():
    df = pd.DataFrame()
    df['a'] = np.random.choice(a=["x", "y", "z"], size=100, p=[0.4, 0.3, 0.3])
    df['b'] = np.random.rand(100, 1)
    df['c'] = np.random.rand(100, 1)
    tester = InjectionTesting(df)

    tester.test_adwin_detector()
    tester.test_cbdb_detector(cols=['b'])
    tester.test_ddm_detector()
    tester.test_eddm_detector()
    tester.test_hdddm_detector(cols=['b'])
    tester.test_kdq_tree_batch_detector(cols=['b'])
    tester.test_kdq_tree_streaming_detector(cols=['b'])
    tester.test_lfr_detector()
    tester.test_md3_detector()
    tester.test_nndvi_detector()
    tester.test_pcacd_detector()
    tester.test_stepd_detector()
