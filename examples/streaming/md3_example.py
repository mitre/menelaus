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
# TODO: change this path back after done testing
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

md3 = MD3(clf=clf, oracle_data_length_required=oracle_retrain_labels)
md3.set_reference(training_data, "y")

# Set up DF to record results.
status = pd.DataFrame(
    columns=["index", "y", "margin_density", "drift_detected"]
)
rec_list = []

# run MD3
for i in range(training_size, len(df)):

    X_test = df.loc[[i], ["var1", "var2"]]
    y_true = int(df.loc[[i], "y"])

    # call give_oracle_label if detector is currently in warning state
    if md3.drift_state == "warning":
        oracle_label = df.loc[i, ["var1", "var2", "y"]]
        md3.give_oracle_label(oracle_label)
    # call update otherwise
    else:
        md3.update(X_test)

    status.loc[i] = [
        i,
        y_true,
        md3.curr_margin_density,
        md3.drift_state,
    ]
    
    # If drift is detected, examine the window and retrain.
    if md3.drift_state == "drift":
        retrain_start = i + 1
        retrain_end = i + oracle_retrain_labels

        rec_list.append([retrain_start, retrain_end])

# TODO: delete this after drift is actually being detected
rec_list = [[1000, 1250]]
    
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
rec_list = pd.DataFrame(rec_list)
rec_list["y_val"] = np.linspace(
    start=0.05 * (ylims[1] - ylims[0]) + ylims[0],
    stop=0.2 * ylims[1],
    num=len(rec_list),
)

# Draw green lines that indicate where retraining occurred
plt.hlines(
    y=rec_list["y_val"],
    xmin=rec_list[0],
    xmax=rec_list[1],
    color="green",
    label="Retraining Windows",
)

plt.legend()

# TODO: write a description here for MD3 instead of ADWIN
# After drift is induced, the accuracy decreases enough for ADWIN to shrink its
# window and alarm;  subsequent windows also include data from the old regime,
# so drift continues to be detected until the window shrinks enough to be
# comprised mostly by the new regime.

# plt.show()
plt.savefig("example_MD3.png")
