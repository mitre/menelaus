import numpy as np
import pandas as pd
from menelaus.datasets import fetch_circle_data, fetch_rainfall_data
from menelaus.injection.injection_automation import InjectionTesting


def test_brownian_injection():
    df = pd.DataFrame(np.random.rand(100, 5), columns=['a', 'b', 'c', 'd', 'e'])
    tester = InjectionTesting(df)
    start = 0.75
    end = 1

    col = tester.inject_random_brownian_noise(50, start=start, end=end, num_drift_cols=1)
    std_normal = tester.df.iloc[0 : int(start * len(df)), ][col].std().iloc[0, ]
    std_drift = tester.df.iloc[int(start * len(df)) + 1 : int(end * len(df)), ][col].std().iloc[0, ]

    assert(std_drift > std_normal)
