"""All methods for testing correctness of CUSUM implementation."""

import os
import pandas as pd
import numpy as np

from molten.other.cusum import CUSUM


def test_repeated_cusum():
    """Test that CUSUM detects drift at same locations, multiple times."""
    # TODO - sort out this hack for loading data
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    full_path = os.path.join(os.getcwd(), 'artifacts', 'dataCircleGSev3Sp3Train.csv')
    df = pd.read_csv(full_path, usecols=[0,1,2], names=['var1', 'var2', 'y'])
    # set up test (mean / std of 'Var 2' pre-drift) 
    mean = np.mean(df.iloc[0:1000,1])
    std = np.std(df.iloc[0:1000,1])
    locations = None
    # over r repetitions, results should be the same
    for _ in range(3):
        status = pd.DataFrame(columns=['index', 'drift'])
        cusum = CUSUM(
            target=mean,
            sd_hat=std,
            burn_in=50,
            delta=0.005,
            threshold=40,
            direction=None
        )
        for i in range(len(df)):
            obs = df['var2'][i]
            cusum.update(obs)
            status.loc[i] = [i, cusum.drift_state]
        # test locations same
        if not locations:
            locations = set(status[status.drift == 'drift'].index.tolist())
        else:
            new_locations = set(status[status.drift == 'drift'].index.tolist())
            assert locations == new_locations


def test_():
    """TODO - test some other aspect of CUSUM."""
    pass

