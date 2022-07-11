"""

Margin Density Drift Detection (MD3) Method Example
-------------------------

These examples show how to set up, run, and produce output from the MD3 detector. 
The parameters aren't necessarily tuned for best performance for the input data, 
just notional.

Rainfall is a real data source that concept drift has been injected into. This
set contains approximately 18,000 samples, and the data has been standardized.
Drift starts from index 12,000 and continues through the rest of the dataset.
In this example, we take the first 10,000 samples of the dataset for training
an initial classifier, and then use the remaining samples for testing.

TODO: update this description
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
from menelaus.concept_drift import MD3


# read in rainfall dataset
# assumes the script is being run from the root directory.
# TODO: add this example as part of the concept_drift_example.py script?
# TODO: add this example as one of the example notebooks in dev? --> will have to
#       merge dev back into this branch
# TODO: change method of importing data to match other examples
df = pd.read_csv(
    os.path.join(
        "..", "src", "menelaus", "datasets", "rainfall_data.csv"
    ),
    usecols=[1, 2, 3, 4, 5, 6, 7, 8, 9],
    names=["index", "temperature", "dew_point", "sea_level_pressure", "visibility", "average_wind_speed", "max_sustained_wind_speed", "minimum_temperature", "maximum_temperature", "rain"],
)
df = df.iloc[1: , :].reset_index(drop=True)
drift_start, drift_end = 12000, 18158
training_size = 10000

columns = ["temperature", "dew_point", "sea_level_pressure", "visibility", "average_wind_speed", "max_sustained_wind_speed", "minimum_temperature", "maximum_temperature", "rain"]
features = ["temperature", "dew_point", "sea_level_pressure", "visibility", "average_wind_speed", "max_sustained_wind_speed", "minimum_temperature", "maximum_temperature"]
df[features] = df[features].astype(float)


################################################################################
#################################### MD3 #######################################
################################################################################

# Set up classifier: train on first training_size rows
training_data = df.loc[0:training_size, columns]
X_train = df.loc[0:training_size, features]
y_train = df.loc[0:training_size, "rain"]

np.random.seed(123)
clf = svm.SVC(kernel='linear')
clf.fit(X_train, y_train.values.ravel())
oracle_labels = 1000

# TODO: play around more with this sensitivity
# it also seems like it doesn't make sense to have the default value be 2 --> see what the paper
# says again, i thought they said a value in the range [1, 3] should work but apparently not
md3 = MD3(clf=clf, sensitivity=1.5, oracle_data_length_required=oracle_labels)
md3.set_reference(training_data, "rain")

# TODO: track cumulative accuracy for the classifier over the samples and add
#       to the plot that gets outputted, so that we can track margin vs detector
#       performance and see exactly where accuracy changes and how
# TODO: have two different loops that each output a plot. the first one can
#       retrain the model when drift is confirmed, and the other doesn't retrain.
#       and track accuracy for both and see how the plots differ. or have this on
#       one plot and just two different lines
# Set up DF to record results.
status = pd.DataFrame(
    columns=["index", "y", "margin_density", "drift_detected"]
)
rec_list = []
oracle_list = []

# run MD3
for i in range(training_size, len(df)):

    X_test = df.loc[[i], features]
    y_true = int(df.loc[[i], "rain"])

    # call give_oracle_label if detector is currently waiting for oracle data
    if md3.waiting_for_oracle == True:
        oracle_label = df.loc[[i], columns]
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

plt.axvspan(drift_start, drift_end, alpha=0.5, label="Drift Induction Window")

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
