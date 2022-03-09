""" Use these functions to run experiments with various configurations """

import os
import time
import pandas as pd

from sklearn.preprocessing import StandardScaler

from molten.data_drift.pca_cd import PCACD


DATA_PATHS = {
    "circle33": "dataCircleGSev3Sp3Train.csv",
    "fake_wls": "fake_wls_eligibility.csv",
}


def get_data(data_id, sample_by=None, sample_size=None, drop=None, randomize=False):
    """Get dataframe.

    Args:
        data_id (str): Dictionary key for dataset.
        sample_by (str, optional): If not None, variable to batch/group by (e.g., year). Defaults to None.
        sample_size (int, optional): If not None, specifies batch size. Defaults to None.
        drop (list, optional): Attribute names (str) to remove from dataframe. Defaults to None.
        randomize (bool, optional): True if seed must be fixed (e.g., no bootstrapping). Defaults to False.

    Returns:
        pandas.DataFrame: Loaded dataset.
    """
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    full_path = os.path.join(os.getcwd(), "artifacts", DATA_PATHS[data_id])
    if data_id == 'circle33':
        df = pd.read_csv(full_path, usecols = [0,1,2],names = ['var1','var2','y'])
    else:
        df = pd.read_csv(full_path)
    if sample_by:
        df_list = [d for _, d in df.groupby([sample_by])]
        if not randomize:
            df_list = [d.sample(n=sample_size, random_state=0) for d in df_list]
        else:
            df_list = [d.sample(n=sample_size) for d in df_list]
        df = pd.concat(df_list).reset_index(drop=True)
    if drop:
        df = df.drop(drop, axis=1)
    return df


def run_pca_cd(dir_path, df, divergence_metric, window_size, ev_threshold):
    """Helper function to run single PCA_CD trial, write to status files as needed.

    Args:
        dir_path (str): Path to common directory for this configuration.
        df (pandas.DataFrame): Prepared dataset.
        divergence_metric (str): Divergence metric to use.
        window_size (int): Window size to use within PCA_CD.
        ev_threshold (float): Explained variance threshold for PCA_CD.

    Returns:
        dict: Information about the trial run (time, locations, and PCs used).
    """
    # create empty status file for this trial, save inside configuration directory
    timestamp = int(time.time())
    status_path = "%s.csv" % os.path.join(dir_path, str(timestamp))
    status = pd.DataFrame(columns=["drift_loc"])

    # begin timing prior to PCA_CD creation and scaling
    status.to_csv(status_path)
    start = time.time()
    pca_cd = PCACD(
        divergence_metric=divergence_metric,
        window_size=window_size,
        ev_threshold=ev_threshold,
        online_scaling=True
    )

    # update over rows, appending any observed drift to status file
    for k in range(len(df)):
        pca_cd.update(df.iloc[[k]])
        if pca_cd.drift_state:
            new_record = pd.DataFrame({"drift_loc": [k]})
            new_record.to_csv(status_path, header=None, mode='a')

    # end time, return additional information about trial run
    diff = time.time() - start
    return {"runtime": diff, "num_pcs": pca_cd.num_pcs, "trial_path": status_path}


def run_trials(
    df,
    data_id,
    times=1,
    dms=["kl", "llh", "int"],
    wss=[300, 350],
    evs=[0.9, 0.99],
    sample_size=1000,
):
    """Run a set of trials, potentially multiple repetitions + combinations of configurations.
        Save each trial result to master .csv file as needed.

    Args:
        df (pandas.DataFrame): Prepared dataset.
        data_id (str): Specify dataset to use for documenting purposes.
        times (int, optional): Number of repetitions (e.g., bootstrapping). Defaults to 1.
        dms (list[str], optional): Set of divergence metrics for PCA_CD. Defaults to ['kl', 'llh', 'int'].
        wss (list[int], optional): Set of window sizes for PCA_CD. Defaults to [300,350].
        evs (list[float], optional): Set of explained variance thresholds for PCA_CD. Defaults to [0.9, 0.99].
        sample_size (int, optional): For documenting batch size to master file, currently bugged. Defaults to 1000.
    """
    # create master dataframe, each row is a single trial
    df_experiments = pd.DataFrame(
        columns=[
            "n",
            "sample_size_per_window",
            "trial_path",
            "dm",
            "ws",
            "ev",
            "num_pcs",
            "runtime",
        ]
    )
    log_path = os.path.join(os.getcwd(), "artifacts", "master.csv")
    if not os.path.exists(log_path):
        df_experiments.to_csv(log_path)

    # iterate over reps, parameter grid
    for i in range(times):
        for dm in dms:
            for ws in wss:
                for ev in evs:

                    # setup and create configuration directory (PCA_CD only)
                    dir_path = "p_%s_%s_%s_%s" % (data_id, dm, ws, int(100 * ev))
                    dir_path = os.path.join(os.getcwd(), "artifacts", dir_path)
                    if not os.path.exists(dir_path):
                        os.mkdir(dir_path)

                    # run single trial and create entry for master file, append
                    result = run_pca_cd(dir_path, df, dm, ws, ev)
                    new_record = pd.DataFrame(
                        {
                            "n": [len(df)],
                            "sample_size_per_window": sample_size,
                            "trial_path": [result["trial_path"]],
                            "dm": [dm],
                            "ws": [ws],
                            "ev": [ev],
                            "num_pcs": [result["num_pcs"]],
                            "runtime": [result["runtime"]],
                        }
                    )
                    new_record.to_csv(log_path, header=None, mode="a")
                    print(new_record)
    print("Done.")


# if __name__ == '__main__':
#     # specify parameter grid
#     data_id = 'circle33' # 'fake_wls'
#     sample_by = 300 # 'tax_yr'
#     drop = ['y']
#     sample_size = 300 # 300, shares meaning with window size below
#     wss = [300]
#     dms = ['kl', 'llh']
#     evs = [0.99]
#     times = 1000
#     df = get_data(data_id, sample_by, sample_size, drop, randomize=True) # True: bootstrapping sample_size points from each year
#     run_trials(df, data_id, times, dms, wss, evs, sample_size)