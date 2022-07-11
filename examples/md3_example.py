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

"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from menelaus.concept_drift import MD3
from menelaus.datasets import fetch_rainfall_data


# TODO: add this example as part of the concept_drift_example.py script?
# TODO: add this example as one of the example notebooks in dev?

# read in Rainfall dataset
df = fetch_rainfall_data()
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

md3 = MD3(clf=clf, sensitivity=1.5, oracle_data_length_required=oracle_labels)
md3.set_reference(training_data, "rain")

# TODO: have two different loops that each output a plot. the first one can
#       retrain the model when drift is confirmed, and the other doesn't retrain.
#       and track accuracy for both and see how the plots differ. or have this on
#       one plot and just two different lines
# Set up DF to record results.
status = pd.DataFrame(
    columns=["index", "y", "margin_density", "accuracy", "drift_detected"]
)
correct = 0
n = 1
rec_list = []
oracle_list = []

# run MD3
for i in range(training_size, len(df)):

    X_test = df.loc[[i], features]
    y_pred = int(clf.predict(X_test))
    y_true = int(df.loc[[i], "rain"])
    
    # increment accuracy
    if y_pred == y_true:
        correct += 1
    accuracy = correct / n

    # call give_oracle_label if detector is currently waiting for oracle data
    if md3.waiting_for_oracle == True:
        oracle_label = df.loc[[i], columns]
        md3.give_oracle_label(oracle_label)
        status.loc[i] = [
            i,
            y_true,
            None,
            accuracy,
            md3.drift_state,
        ]

    # call update otherwise
    else:
        md3.update(X_test)
        status.loc[i] = [
            i,
            y_true,
            md3.curr_margin_density,
            accuracy,
            md3.drift_state,
        ]
    
    # If there was a drift warning, track the window of the labeled
    # oracle data used
    if md3.drift_state == "warning":
        oracle_start = i + 1
        oracle_end = i + md3.oracle_data_length_required
        
        oracle_list.append([oracle_start, oracle_end])

    n += 1
    
plt.figure(figsize=(20, 6))
plt.scatter("index", "margin_density", data=status, label="Margin Density")
plt.scatter("index", "accuracy", data=status, label="Accuracy", color="green")
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

# After drift is induced, the margin density decreases enough for MD3 to
# emit a warning. From there, the predictive accuracy of the classifier is
# tested, and this has already decreased sufficiently for the detector to
# alarm. Then, a new reference batch is set and the detector continues
# tracking the margin density statistic until the next warning.

# plt.show()
plt.savefig("example_MD3.png")
