"""

Data Drift Examples
   1. PCA-Based Change Detection (PCA-CD)
   2. KDQ-Tree Detection Method
-------------------------

These examples show how to set up, run, and produce output from detectors in the 
data_drift module. The parameters aren't necessarily tuned for best 
performance for the input data, just notional. These detectors are generally
applied to the whole feature set for a given data source. 

The example data for kdqTree, Circle,  is a synthetic data source, where drift
occurs in both var1, var2, and the conditional distributions P(y|var1) and
P(y|var2). The drift occurs from index 1000 to 1250, and affects 66% of the
sample.

NOTE: This example takes roughly 30 seconds to run.

"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from menelaus.data_drift import PCACD, KdqTree


# read in Circle dataset
# assumes the script is being run from the root directory.
df = pd.read_csv(
    os.path.join(
        "src", "menelaus", "tools", "artifacts", "dataCircleGSev3Sp3Train.csv"
    ),
    usecols=[0, 1, 2],
    names=["var1", "var2", "y"],
)

################################################################################
################################## PCA_CD ######################################
################################################################################
# PCA-CD is a drift detector that transforms the passed data into its
# principal components, then watches the transformed data for signs of drift
# by monitoring the KL-divergence via the Page-Hinkley algorithm.

pca_cd = PCACD(window_size=50, divergence_metric="intersection")

# set up dataframe to record results
status = pd.DataFrame(columns=["index", "var1", "var2", "drift_detected"])

# Put together a dataframe of several features, each of which abruptly changes
# at index 1000.
np.random.seed(1)
size = 1000
data = pd.concat(
    [
        pd.DataFrame(
            [
                np.random.normal(1, 10, size),
                np.random.uniform(1, 2, size),
                np.random.normal(0, 1, size),
            ]
        ).T,
        pd.DataFrame(
            [
                np.random.normal(9, 10, size),
                np.random.normal(1, 3, size),
                np.random.gamma(20, 30, size),
            ]
        ).T,
    ]
)

# Update the drift detector with each new sample
for i in range(len(df)):
    pca_cd.update(data.iloc[[i]])
    status.loc[i] = [i, data.iloc[i, 0], data.iloc[i, 1], pca_cd.drift_state]


# Plot the features and the drift
plt.figure(figsize=(20, 6))
plt.scatter(status.index, status.var2, label="Var 2")
plt.scatter(status.index, status.var1, label="Var 1", alpha=0.5)
plt.grid(False, axis="x")
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.title("PCA-CD Test Results", fontsize=22)
plt.ylabel("Value", fontsize=18)
plt.xlabel("Index", fontsize=18)
ylims = min(status.var1.min(), status.var2.min()), max(
    status.var1.max(), status.var1.max()
)
plt.ylim(ylims)

# Draw red lines that indicate where drift was detected
plt.vlines(
    x=status.loc[status["drift_detected"] == "drift"]["index"],
    ymin=ylims[0],
    ymax=ylims[1],
    label="Drift Detected",
    color="red",
)

plt.legend()

# PCA-CD detects this very abrupt drift within a few samples of its induction.
# plt.show()
plt.savefig("example_PCA_CD.png")


################################################################################
################################# KDQ-Tree #####################################
################################################################################
# KdqTree monitors incoming features by constructing a tree which partitions the
# feature-space, and then monitoring a divergence statistic that is defined over
# that partition. It watches data within a sliding window of a particular size.
# When that window is full, it builds the reference tree. As the window moves
# forward, point-by-point, the data in that new window is compared against the
# reference tree to detect drift.


# kdqTree does use bootstrapping to define its critical thresholds, so setting
# the seed is important to reproduce exact behavior.
np.random.seed(1)

# Note that the default input_type for KDQTree is "stream".
# The window size, corresponding to the portion of the stream which KDQTree
# monitors, must be specified.
det = KdqTree(window_size=500, alpha=0.05, bootstrap_samples=500, count_ubound=50)

# setup DF to record results
status = pd.DataFrame(columns=["index", "var1", "var2", "drift_detected"])

# iterate through X data and run detector
data = df[["var1", "var2"]]

plot_data = {}
for i in range(len(df)):
    det.update(data.iloc[[i]].values)
    status.loc[i] = [i, data.iloc[i, 0], data.iloc[i, 1], det.drift_state]
    if det.drift_state is not None:
        # capture the visualization data
        plot_data[i] = det.to_plotly_dataframe()

plt.figure(figsize=(20, 6))
plt.scatter("index", "var2", data=status, label="var2")
plt.scatter("index", "var1", data=status, label="var1", alpha=0.5)
plt.grid(False, axis="x")
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.title("KDQ Tree Test Results", fontsize=22)
plt.ylabel("Value", fontsize=18)
plt.xlabel("Index", fontsize=18)
ylims = [-0.05, 1.05]
plt.ylim(ylims)

drift_start, drift_end = 1000, 1250
plt.axvspan(drift_start, drift_end, alpha=0.5, label="Drift Induction Window")

# Draw red lines that indicate where drift was detected
plt.vlines(
    x=status.loc[status["drift_detected"] == "drift"]["index"],
    ymin=ylims[0],
    ymax=ylims[1],
    label="Drift Detected",
    color="red",
)

plt.legend()


# Given a window_size of 500, with only the two input features, KdqTree detects
# a change after about half of the data within its window is in the new regime.
# plt.show()
plt.savefig("example_streaming_kdqtree_feature_stream.png")

# If we save off the to_plotly_dataframe at each drift detection, we can display
# the Kulldorff Spatial Scan Statistic (KSS) for each. Higher values of KSS
# indicate that a given region of the data space has greater divergence between
# the reference and test data.
#
# Note that the structure of the particular tree depends on the reference data
# and the order of the columns within the dataframe!

# Since this data only contains two features, the tree is relatively
# shallow.
for title, df_plot in plot_data.items():
    fig = px.treemap(
        data_frame=df_plot,
        names="name",
        ids="idx",
        parents="parent_idx",
        color="kss",
        color_continuous_scale="blues",
        title=f"Index {title}",
    )
    fig.update_traces(root_color="lightgrey")
    # fig.show()
    fig.write_html(f"example_streaming_kdqtree_treemap_{title}.html")
