"""

Margin Density Drift Detection (MD3) Method Example
-------------------------

These examples show how to set up, run, and produce output from the MD3 detector. 
The parameters aren't necessarily tuned for best performance for the input data, 
just notional.

Circle is a synthetic data source, where drift occurs in both var1, var2, and the 
conditional distributions P(y|var1) and P(y|var2). The drift occurs from index 
1000 to 1250, and affects 66% of the sample.

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
X_train = df.loc[0:training_size, ["var1", "var2"]]
y_train = df.loc[0:training_size, "y"]

np.random.seed(123)
clf = svm.SVC(kernel='linear')
clf.fit(X_train, y_train)

md3 = MD3()

i = 650
X_test = df.loc[[i], ["var1", "var2"]]
y_pred = int(clf.predict(X_test))
y_true = int(df.loc[[i], "y"])

