#!/usr/bin/env python
# coding: utf-8

# # Change Detection Examples

# These examples show how to set up, run, and produce output from detectors in the 
# change_detection module. The parameters aren't necessarily tuned for best 
# performance for the input data, just notional.
# 
# Circle is a synthetic data source, where drift occurs in both var1, var2, and the 
# conditional distributions P(y|var1) and P(y|var2). The drift occurs from index 
# 1000 to 1250, and affects 66% of the sample.
# 
# These change detectors can be applied to any given single variable; below, 
# they are applied to var2.

# In[ ]:


## Imports ##

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from menelaus.change_detection import PageHinkley, CUSUM
from menelaus.datasets import fetch_circle_data


# In[ ]:


## Import Data ##

# read in Circle dataset
df = fetch_circle_data()
drift_start, drift_end = 1000, 1250


# ## Page-Hinkley (PH) Test Example

# In[ ]:


## Setup ##

# Set up one-directional PH test: this will only alarm if the mean of the
# monitored variable decreases, and only after seeing 30 or more samples.
ph = PageHinkley(delta=0.01, threshold=15, direction="negative", burn_in=30)

# setup DF to record results
status = pd.DataFrame(columns=["index", "actual value", "drift_detected"])

# iterate through data; feed each sample to the detector, in turn
for i in range(len(df)):
    obs = df["var2"][i]
    ph.update(X=obs)
    status.loc[i] = [i, obs, ph.drift_state]


# In[ ]:


## Plotting ##

# plot the monitored variable and the status of the detector
plt.figure(figsize=(20, 6))
plt.scatter("index", "actual value", data=status, label="Var 2")
plt.grid(False, axis="x")
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.title("PH Test Results", fontsize=22)
plt.ylabel("Value", fontsize=18)
plt.xlabel("Index", fontsize=18)
ylims = [-0.05, 1.1]
plt.ylim(ylims)

plt.axvspan(drift_start, drift_end, alpha=0.5, label="Drift Induction Window")

plt.vlines(
    x=status.loc[status["drift_detected"] == "drift"]["index"],
    ymin=ylims[0],
    ymax=ylims[1],
    label="Drift Detected",
    color="red",
)
plt.legend()


# Page-Hinkley alarms shortly after the drift induction window closes, and then makes
# several apparently erroneous alarms afterwards. The parameters may not be
# well-chosen for the new regime.
# Change detection algorithms come out of process control, so a priori
# characterization of the bounds of the process, not performed here, would not
# be unreasonable.
# 

# In[ ]:


plt.show()
# plt.savefig("example_Page-Hinkley_detections.png")


# ## Cumulative Sum (CUSUM) Test Example

# - This monitors a moving average of var2, starting from an initial estimate of mean
# and standard deviation.
# 
# - It will only alarm if 50 or more samples have been observed since
# initialization/drift.
# 
# - This will alarm if var2 passes a critical value controlled by delta and
# threshold in either direction, positive or negative.
# 
# 

# In[ ]:


## Setup ##
cusum = CUSUM(
    target=np.mean(df.loc[:drift_start, "var2"]),  # mean / std of 'Var 2' pre-drift
    sd_hat=np.std(df.loc[:drift_start, "var2"]),
    burn_in=50,
    delta=0.005,
    threshold=40,
    direction=None,
)

# setup DF to record results
status = pd.DataFrame(columns=["index", "actual value", "drift_detected"])

# iterate through data; feed each sample to the detector, in turn
for i in range(len(df)):
    obs = df["var2"][i]
    cusum.update(obs)
    status.loc[i] = [i, obs, cusum.drift_state]


# In[ ]:


## Plotting ##

# plot the monitored variable and the status of the detector
plt.figure(figsize=(20, 6))
plt.scatter("index", "actual value", data=status, label="Var 2")
plt.grid(False, axis="x")
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.title("CUSUM Test Results", fontsize=22)
plt.ylabel("Value", fontsize=18)
plt.xlabel("Index", fontsize=18)
ylims = [-0.05, 1.1]
plt.ylim(ylims)

plt.axvspan(drift_start, drift_end, alpha=0.5, label="Drift Induction Window")

plt.vlines(
    x=status.loc[status["drift_detected"] == "drift"]["index"],
    ymin=ylims[0],
    ymax=ylims[1],
    label="Drift Detected",
    color="red",
)
plt.legend()


# CUSUM alarms several times within the drift induction window, roughly halfway
# through. After the alarm is reset, change is detected a few more times,
# including an apparently erroneous detection after the drift induction window
# is passed. The current threshold settings may then be too sensitive for the
# new regime.
# 

# In[ ]:



plt.show()
# plt.savefig("example_CUSUM_detections.png")

