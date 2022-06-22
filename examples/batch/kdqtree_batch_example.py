"""

kdq-Tree Example (batch setting)
----------------------------

This example shows up how to set up, run, and produce output from the kdq-Tree
detector, specifically in the batch data setting. The parameters aren't 
necessarily tuned for best performance, just notional.

Drift in the example dataset occurs in 2009, 2012, 2015, 2018, and 2021. Drift
in 2018 persists through 2021. See
src/menelaus/tools/artifacts/README_example_data.txt for more details.

This example takes roughly a minute to run.

"""

import os
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
from menelaus.data_drift.kdq_tree import KdqTree


# kdq-Tree does use bootstrapping to define its critical thresholds, so setting
# the seed is important to reproduce exact behavior.
np.random.seed(123)

# Import data
data_path = os.path.join("..", "..", "src", "menelaus", "tools", "artifacts", "example_data.csv")
df_orig = pd.read_csv(data_path, index_col="id", dtype={"drift": bool})

# Convert the categorical columns to dummy variables
df = pd.concat([df_orig, pd.get_dummies(df_orig.cat, prefix="cat")], axis=1)

# Capture the column which tells us when drift truly occurred
drift_years = df.groupby("year")["drift"].apply(lambda x: x.unique()[0]).reset_index()

# Because the drift in 2009, 2012, and 2016 is intermittent - it reverts
# back to the prior distribution in the subsequent year - we should also detect
# drift in 2010, 2013, and 2016. So:
drift_years.loc[drift_years["year"].isin([2010, 2013, 2016]), "drift"] = True


df.drop(columns=["cat", "confidence", "drift"], inplace=True)


plot_data = {}
status = pd.DataFrame(columns=["year", "drift"])
det = KdqTree(input_type="batch")

# Set up reference batch, using 2007 as reference year
det.set_reference(df[df.year == 2007].values)

# Batch the data by year and run kdqTree
for group, sub_df in df[df.year != 2007].groupby("year"):
    det.update(sub_df.drop(columns=["year"]).values)
    status = pd.concat(
        [status, pd.DataFrame({"year": [group], "drift": [det.drift_state]})],
        axis=0,
        ignore_index=True,
    )
    if det.drift_state is not None:
        # capture the visualization data
        plot_data[group] = det.to_plotly_dataframe()
        
        # option to specify reference batch to be any year 
        #det.set_reference(df[df.year == XXXX].values)

# Print out the true drift status, and that according to the detector.
# The detector successfully identifies drift in every year but 2018;
# that's the first year where persistent drift, from 2018-2021, was induced.
# The detector picks it up in 2019, the second year of persistent drift.
(
    status.merge(drift_years, how="left", on="year", suffixes=["_kdqTree", "_true"])
    .replace({True: "drift", False: None})
    .to_csv("example_kdqtree_drift_comparison.csv", index=False)
)

# If we save off the to_plotly_dataframe at each drift detection, we can display
# the Kulldorff Spatial Scan Statistic (KSS) for each. Higher values of KSS
# indicate that a given region of the data space has greater divergence between
# the reference and test data.
for year, df_plot in plot_data.items():
    fig = px.treemap(
        data_frame=df_plot,
        names="name",
        ids="idx",
        parents="parent_idx",
        color="kss",
        color_continuous_scale="blues",
        title=year,
    )
    fig.update_traces(root_color="lightgrey")
    # fig.show()
    fig.write_html(f"example_kdqtree_treemap_{year}.html")

# We can see that the regions of greatest drift do line up with at least one of
# the items that were modified in a given year.

# For reference, the detailed descriptions of drift induction:
# Drift 1: change the mean & var of item B in 2009, means will revert for 2010 on
# Drift 2: change the variance of item c and d in 2012 by replacing some with the mean
# keep same mean as other years, revert by 2013
# Drift 3: change the correlation of item e and f in 2015 (go from correlation of 0 to correlation of 0.5)
# Drift 4: change mean and var of H and persist it from 2018 on
# Drift 5: change mean and var just for a year of J in 2021


