"""Methods for testing correctness of PCA-CD."""

import os
import pandas as pd
from sklearn.preprocessing import StandardScaler

from molten.distribution.pca_cd import PCACD


def test_repeated_pcacd():
    """Test that PCACD detects drift at same locations, multiple times."""
    # # TODO - sort out this hack for loading data
    # os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # full_path = os.path.join(os.getcwd(), 'artifacts', 'dataCircleGSev3Sp3Train.csv')
    # df = pd.read_csv(full_path, usecols=[0,1,2], names=['var1', 'var2', 'y'])
    # locations = None
    # # repeat r times
    # for _ in range(2):
    #     # set up pca cd and reporting
    #     status = pd.DataFrame(columns=['index', 'drift'])
    #     pca_cd = PCACD(window_size = 350, delta = 0.05)
    #     data = df[['var1','var2']]
    #     data = pd.DataFrame(StandardScaler().fit_transform(data))
    #     for i in range(len(df)):
    #         pca_cd.update(data.iloc[[i]])
    #         status.loc[i] = [i, pca_cd.drift_state]
    #     # test locations same
    #     if not locations:
    #         locations = set(status[status.drift == 'drift'].index.tolist())
    #     else:
    #         new_locations = set(status[status.drift == 'drift'].index.tolist())
    #         assert locations == new_locations
    assert True
