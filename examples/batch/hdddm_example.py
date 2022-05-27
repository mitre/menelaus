""" 

Hellinger Distance Drift Detection Method (HDDDM) Example

This file details how to setup, run, and produce plots for HDDDM, using both
numeric and categorical data. Drift occurs in 2009, 2012, 2015, 2018, and 2021.
Drift in 2018 persists through 2021. See
src/menelaus/tools/artifacts/README_example_data.txt for more info.

HDDDM must be setup and run with batches of data. 

Plots include:
- A line plot visualizing test statistics for detection of drift 

- A heatmap visualizing "where" drift is occuring, showing features for each
  year's test batch with the greatest Hellinger distance from the reference
  batch.

"""

import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from menelaus.data_drift.hdddm import HDDDM


## Setup ##

# Import data
# assumes the script is being run from the root directory.
data = pd.read_csv(
    os.path.join("src", "menelaus", "tools", "artifacts", "example_data.csv"),
    index_col="id",
)

# Set up reference and test batches, using 2007 as reference year
# -2 indexing removes columns "drift" and "confidence"
reference = data[data.year == 2007].iloc[:, 1:-2]
all_test = data[data.year != 2007]

#  Run HDDDM
hdddm = HDDDM(reference, subsets=8)
# Store epsilons per feature for heatmap
feature_epsilons = []
# Store drift for test statistic plot
detected_drift = []
for year, subset_data in all_test.groupby("year"):
    hdddm.update(subset_data.iloc[:, 1:-2])
    feature_epsilons.append(hdddm.feature_epsilons)
    detected_drift.append(hdddm.drift_state)


## Plot Line Graph ##

# Plot Info:
# HDDDM identifies drifts in 2009, 2010, 2012, 2019, 2021. These drifts involve
# a change in mean or variance. Drift in 2010 is likely identified as the
# distribution returns to state prior to 2009 drift. Drift in 2015, a change in
# correlation, is undetected. Drift in 2018 is detected one year late.

# Calculate Hellinger distances for all years in dataset
years = list(data.year.value_counts().index[1:])
h_distances = [
    ep - th for ep, th in zip(hdddm.epsilon_values.values(), hdddm.thresholds.values())
]

# Plot Hellinger Distance against Year, along with detected drift
plot_data = pd.DataFrame(
    {"Year": years, "Hellinger Distance": h_distances, "Detected Drift": detected_drift}
)
sns.set_style("white")
plt.figure(figsize=(20, 6))
plt.plot(
    "Year", "Hellinger Distance", data=plot_data, label="Hellinger Distance", marker="."
)
plt.grid(False, axis="x")
plt.xticks(years, fontsize=16)
plt.yticks(fontsize=16)
plt.title("HDDDM Test Statistics", fontsize=22)
plt.ylabel("Hellinger Distance", fontsize=18)
plt.xlabel("Year", fontsize=18)
plt.ylim([min(h_distances) - 0.02, max(h_distances) + 0.02])
for _, t in enumerate(plot_data.loc[plot_data["Detected Drift"] == "drift"]["Year"]):
    plt.axvspan(
        t - 0.2, t + 0.2, alpha=0.5, color="red", label=("Drift Detected" if _ == 0 else None)
    )
plt.legend()
plt.axhline(y=0, color="orange", linestyle="dashed")

# plt.show()
plt.savefig("example_HDDDM_test_statistics.png")


## Plot Heatmap ##

# Plot Info:
# Drift in feature B is detected in 2009 and 2010 (as it reverts to normal).
# Drift in feature D is detected in 2012 and 2013 (as it reverts to normal).
# Drift in feature H is detected in 2019. Drift in feature J is detected in 2021.
# The undetected drift occurs in 2015 in the correlations between features E and F.

sns.set_style("whitegrid")
sns.set(rc={"figure.figsize": (15, 8)})

# Flip axis of feature epsilon matrix
inverted_matrix = []
for alpha in range(0, len(list(data.columns[1:-2]))):
    var = []
    for year in range(len(feature_epsilons)):
        var.append(feature_epsilons[year][alpha])
    inverted_matrix.append(var)


# Setup plot
grid_kws = {"height_ratios": (0.9, 0.05), "hspace": 0.3}
f, (ax, cbar_ax) = plt.subplots(2, gridspec_kw=grid_kws)
coloring = sns.cubehelix_palette(start=0.8, rot=-0.5, as_cmap=True)
ax = sns.heatmap(
    inverted_matrix,
    ax=ax,
    cmap=coloring,
    xticklabels=years,
    yticklabels=list(data.columns[1:-2]),
    linewidths=0.5,
    cbar_ax=cbar_ax,
    cbar_kws={"orientation": "horizontal"},
)
ax.set_title('HDDDM Feature Heatmap')
ax.set(xlabel="Years", ylabel="Features")
ax.collections[0].colorbar.set_label("Difference in Hellinger Distance")
ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

# plt.show()
plt.savefig("example_HDDDM_feature_heatmap.png")
