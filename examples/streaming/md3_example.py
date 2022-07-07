"""

Margin Density Drift Detection (MD3) Method Example
-------------------------

These examples show how to set up, run, and produce output from the MD3 detector. 
The parameters aren't necessarily tuned for best performance for the input data, 
just notional.

Circle is a synthetic data source of 2000 samples, where drift occurs in both 
var1, var2, and the conditional distributions P(y|var1) and P(y|var2). The drift 
occurs from index 1000 to 1250, and affects 66% of the sample.

These detectors are generally to be applied to the true class and predicted class 
from a particular model. ADWIN is an exception in that it could also be used to 
monitor an arbitrary real-valued feature. So, each of the summary plots displays 
the running accuracy of the classifier alongside the drift detector's output.

They also track the indices of portions of the incoming data stream which are 
more similar to each other -- i.e., data that seems to be part of the same 
concept, which could be used to retrain a model.

"""


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier
from menelaus.concept_drift import MD3


# read in Circle dataset
# assumes the script is being run from the root directory.
# TODO: before merging in PR, change this path to match dev
df = pd.read_csv(
    os.path.join(
        "..", "..", "src", "menelaus", "tools", "artifacts", "dataCircleGSev3Sp3Train.csv"
    ),
    usecols=[0, 1, 2],
    names=["var1", "var2", "y"],
)
drift_start, drift_end = 1000, 1250
training_size = 500


################################################################################
#################################### MD3 #######################################
################################################################################

# Set up classifier: train on first training_size rows
training_data = df.loc[0:training_size, ["var1", "var2", "y"]]
X_train = df.loc[0:training_size, ["var1", "var2"]]
y_train = df.loc[0:training_size, "y"]

np.random.seed(123)
clf = svm.SVC(kernel='linear')
clf.fit(X_train, y_train.values.ravel())

oracle_retrain_labels = 200

# TODO: play around more with this sensitivity
# it also seems like it doesn't make sense to have the default value be 2 --> see what the paper
# says again, i thought they said a value in the range [1, 3] should work but apparently not
md3 = MD3(clf=clf, sensitivity=0.25, oracle_data_length_required=oracle_retrain_labels)
md3.set_reference(training_data, "y")

# Set up DF to record results.
status = pd.DataFrame(
    columns=["index", "y", "margin_density", "drift_detected"]
)
rec_list = []
oracle_list = []

# run MD3
for i in range(training_size, len(df)):

    X_test = df.loc[[i], ["var1", "var2"]]
    y_true = int(df.loc[[i], "y"])

    # call give_oracle_label if detector is currently waiting for oracle data
    if md3.waiting_for_oracle == True:
        oracle_label = df.loc[[i], ["var1", "var2", "y"]]
        md3.give_oracle_label(oracle_label)
        status.loc[i] = [
            i,
            y_true,
            None,
            md3.drift_state,
        ]

    # call update otherwise
    else:
        md3.update(X_test)
        status.loc[i] = [
            i,
            y_true,
            md3.curr_margin_density,
            md3.drift_state,
        ]
    
    # If there was a drift warning, track the window of the labeled
    # oracle data used
    if md3.drift_state == "warning":
        oracle_start = i + 1
        oracle_end = i + md3.oracle_data_length_required
        
        oracle_list.append([oracle_start, oracle_end])
    
plt.figure(figsize=(20, 6))
plt.scatter("index", "margin_density", data=status, label="Margin Density")
plt.grid(False, axis="x")
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.title("MD3 Results: Margin Density", fontsize=22)
plt.ylabel("Value", fontsize=18)
plt.xlabel("Index", fontsize=18)
ylims = [-0.05, 1.1]
plt.ylim(ylims)

plt.axvspan(1000, 1250, alpha=0.5, label="Drift Induction Window")

# Draw red lines that indicate where drift was detected
plt.vlines(
    x=status.loc[status["drift_detected"] == "drift"]["index"],
    ymin=ylims[0],
    ymax=ylims[1],
    label="Drift Detected",
    color="red",
)

# Draw orange lines that indicate where warnings of drift were provided
plt.vlines(
    x=status.loc[status["drift_detected"] == "warning"]["index"],
    ymin=ylims[0],
    ymax=ylims[1],
    label="Warning",
    color="orange",
    alpha=0.3,
)

# Create a list of lines that indicate the retraining windows.
# Space them evenly, vertically.
oracle_list = pd.DataFrame(oracle_list)
oracle_list["y_val"] = np.linspace(
    start=0.05 * (ylims[1] - ylims[0]) + ylims[0],
    stop=0.2 * ylims[1],
    num=len(oracle_list),
)

# Draw green lines that indicate where retraining occurred
plt.hlines(
    y=oracle_list["y_val"],
    xmin=oracle_list[0],
    xmax=oracle_list[1],
    color="green",
    label="Labeled Oracle Data",
)

plt.legend()

# TODO: write a description here for MD3 instead of ADWIN
# After drift is induced, the accuracy decreases enough for ADWIN to shrink its
# window and alarm;  subsequent windows also include data from the old regime,
# so drift continues to be detected until the window shrinks enough to be
# comprised mostly by the new regime.

# plt.show()
plt.savefig("example_MD3.png")
