""" 

Confidence Distribution Batch Detection (CDBD) Example

This file details how to setup, run, and produce plots for CDBD. This script
monitors the feature "confidence", simulated confidence scores output by a
classifier. Drift occurs in 2018 and persists through 2021. See
src/menelaus/tools/artifacts/README_example_data.txt for more info.
It additionally contains an example of using a custom divergence function. 

CDBD must be setup and run with batches of data containing 1 variable. 

Plots include: 
- A line plot visualizing test statistics for detection of drift

"""

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from menelaus.data_drift.cdbd import CDBD


## Setup ##

# Import data
data_rel_path = os.path.join("..", "..", "src", "menelaus", "tools", "artifacts", "example_data.csv")
data_abs_path = os.path.abspath(data_rel_path)
data = pd.read_csv(data_abs_path, index_col="id")

# Set up reference and test batches, using 2007 as reference year
reference = pd.DataFrame(data[data.year == 2007].loc[:, "confidence"])
all_test = data[data.year != 2007]

# Setup CDBD
np.random.seed(1)
cdbd = CDBD(subsets=8)

# Store drift for test statistic plot
years = all_test.year.unique()
detected_drift = []

# Run CDBD
cdbd.set_reference(reference)
for year, subset_data in all_test.groupby("year"):
    cdbd.update(pd.DataFrame(subset_data.loc[:, "confidence"]))
    detected_drift.append(cdbd.drift_state)


## Plot Line Graph ##


kl_divergence = [
    ep - th for ep, th in zip(cdbd.epsilon_values.values(), cdbd.thresholds.values())
]

plot_data = pd.DataFrame(
    {"Year": years, "KL Divergence": kl_divergence, "Detected Drift": detected_drift}
)
sns.set_style("white")
plt.figure(figsize=(20, 6))
plt.plot("Year", "KL Divergence", data=plot_data, label="KL Divergence", marker=".")
plt.grid(False, axis="x")
plt.xticks(years, fontsize=16)
plt.yticks(fontsize=16)
plt.title("CDBD Test Statistics", fontsize=22)
plt.ylabel("KL Divergence", fontsize=18)
plt.xlabel("Year", fontsize=18)
plt.ylim([min(kl_divergence) - 0.02, max(kl_divergence) + 0.02])
for _, t in enumerate(plot_data.loc[plot_data["Detected Drift"] == "drift"]["Year"]):
    plt.axvspan(
        t - 0.2, t + 0.2, alpha=0.5, color="red", label=("Drift Detected" if _ == 0 else None)
    )
plt.legend()
plt.axhline(y=0, color="orange", linestyle="dashed")

# plt.show()
plt.savefig("example_CDBD_test_statistics.png")


### Custom Divergence Metric ### 

# Define divergence function
def distance_metric(reference_histogram, test_histogram):

    # Convert inputs to appropriate datatype 
    ref = np.array(reference_histogram[0])
    test = np.array(test_histogram[0])

    return np.sqrt(np.sum(np.square(ref-test)))

# Test self-defined divergence metric 
cdbd = CDBD(
    divergence=distance_metric,
    detect_batch=1,
    statistic="stdev",
    significance=0.05,
    subsets=5,
)

cdbd.set_reference(reference)
cdbd.update(pd.DataFrame(data[data.year == 2008].loc[:, "confidence"]))