"""All methods for testing correctness of Page Hinkley implementation."""

import os
import pandas as pd

from molten.other.PageHinkley import PageHinkley


def test_repeated_pagehinkley():
    """Test that PH detects drift at same locations, multiple times."""
    # really need to sort out this hack for loading data
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    full_path = os.path.join(os.getcwd(), 'artifacts', 'dataCircleGSev3Sp3Train.csv')
    df = pd.read_csv(full_path, usecols=[0,1,2], names=['var1', 'var2', 'y'])
    # set up test
    delta = 0.01
    xi = 15
    locations = None
    # over r repetitions, results should be the same
    for _ in range(3):
        status = pd.DataFrame(columns=['index', 'drift'])
        ph = PageHinkley(delta=delta, xi=xi, direction='negative', burn_in=30)
        for i in range(len(df)):
            obs = df['var2'][i]
            ph.update(next_obs=obs, obs_id=i)
            status.loc[i] = [i, ph.drift_state]
        # test locations same
        if not locations:
            locations = set(status[status.drift == 'drift'].index.tolist())
        else:
            new_locations = set(status[status.drift == 'drift'].index.tolist())
            assert locations == new_locations


def test_():
    """TODO - test some other aspect of Page Hinkley."""
    pass

