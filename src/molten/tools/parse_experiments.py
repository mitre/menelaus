""" Helper functions: once trials have been run via run_experiments.py,
    these can read/label results
"""

import os
import glob
import pandas as pd


def read_dir(detector, data_id, dm, ws, ev):
    """ Given parameters of a configuration with existing trials,
        read results generated for the same. Returns list of dataframes,
        each of which identifies drift locations identified in a unique trial.

    Args:
        detector (str): Identifies detector used in trials.
        data_id (str): Identifies data used in trials.
        dm (str): Identifies divergence metric used in trials
        ws (int): Identifies window size used in trials.
        ev (float): Identifies EV threshold used in trials.

    Returns:
        list(pandas.DataFrame): For trials with saved data found under the
        run_experiments.py file convention, read results and return list
        of DataFrames (each DataFrame identifies drift indexes for its trial).
    """
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    dir_name = './artifacts/%s_%s_%s_%s_%s/*' % (detector, data_id, dm, str(ws), str(int(100*ev)))
    fs = glob.glob(dir_name)
    dfs = [pd.read_csv(f, index_col=0) for f in fs]
    return dfs


def process_dfs(dfs_list):
    """ Given a list of DataFrames (typically the output of read_dir()
        above), return all drift locations identifies recorded in any of
        the DataFrames (i.e., trials).

    Args:
        dfs_list (list(pandas.DataFrame)): List of DataFrames, each containing
        drift locations identified during a trial. 

    Returns:
        list: Flattened list containing all drift indexes obtained from input
        DataFrames.
    """
    ret = {}
    drifts_lists = [df.drift_loc.tolist() for df in dfs_list]
    ret['trials'] = len(dfs_list)
    ret['drift_locs'] = [drift_loc for drift_list in drifts_lists for drift_loc in drift_list]
    return ret
