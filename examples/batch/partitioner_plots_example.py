"""
Partitioner Plot example
-------------------------

This example shows how to visualize the tree-based partitioning performed by 
KDQTreePartitioner.

KDQTreePartitioner takes input data and partitions the data into a kdq-Tree. New 
data can be passed subsequently and counted according to that same reference 
partition. The two datasets can then be compared.

This partitioner is used within the library within ``data_drift.kdq_tree``, 
where the reference data is used to build the partition, and the test data is 
then divided by that partition, in order to compare the two distributions.
"""

import numpy as np
import plotly.express as px
from menelaus.partitioners.KDQTreePartitioner import KDQTreePartitioner, KDQTreeNode

################################################################################
################################ Basic plot ####################################
################################################################################
# Create some data and build the tree.
data = np.random.randint(0, 10, (20, 3))
kp = KDQTreePartitioner(count_ubound=8)
root = kp.build(data)
KDQTreeNode.as_text(root)

# Grab the output for plotly.
df_plot = kp.to_plotly_dataframe(tree_id1="build")

# This uses only cell_count, which depends only one tree, and doesn't
# compare them.

# Note that it doesn't include any special statistics, just the count.
# df_plot
df_plot.to_csv("example_partitioner_plots_basic.csv", index=False)

# Note that plotly's textinfo value could be used to add alternative values,
# if desired.
fig = px.treemap(
    data_frame=df_plot,
    names="name",
    ids="idx",
    parents="parent_idx",
    values="cell_count",
)
fig.update_traces(root_color="lightgrey")
# fig.show()
fig.write_html(f"example_partitioner_lots_basic_plot.html")

############## Filter by depth
kp = KDQTreePartitioner(count_ubound=25)
df = np.random.sample([50, 3])
df2 = np.random.sample([50, 3])
_ = kp.build(df)
_ = kp.fill(df2, "fill1")
df_plot = kp.to_plotly_dataframe("build", "fill1", max_depth=2)
df_plot

fig = px.treemap(
    data_frame=df_plot,
    names="name",
    ids="idx",
    parents="parent_idx",
    values="cell_count",
)
fig.update_traces(root_color="lightgrey")
# fig.show()
fig.write_html(f"example_partitioner_lots_basic_plot_depth.html")


################################################################################
################################# Modifications ################################
################################################################################
# Count differences between builds can be accessed.
# This can be used to modify the display.
kp = KDQTreePartitioner(count_ubound=25)
df = np.random.sample([50, 3])
df2 = np.random.sample([50, 3])
_ = kp.build(df)
_ = kp.fill(df2, "fill1")
df_plot = kp.to_plotly_dataframe("build", "fill1")
# df_plot
df_plot.to_csv("example_partitioner_plots_modifications.csv", index=False)

fig = px.treemap(
    data_frame=df_plot,
    names="name",
    ids="idx",
    parents="parent_idx",
    color="count_diff",
)
fig.update_traces(root_color="lightgrey")
# fig.show()
fig.write_html(f"example_partitioner_lots_modifications1_count.html")


############## Display additional information
fig = px.treemap(
    data_frame=df_plot,
    names="name",
    ids="idx",
    parents="parent_idx",
    color="count_diff",
)
fig.update_traces(
    root_color="lightgrey", textinfo="label+current path"
)  # see textinfo in https://plotly.com/python/reference/treemap/
# fig.show()
fig.write_html(f"example_partitioner_lots_modifications2_path.html")


##### Access the plot and color using the Kulldorff Spatial Scan Statistic (KSS)
# Higher values of KSS indicate that a given region of the data space has
# greater divergence between two "builds" of data.
fig = px.treemap(
    data_frame=df_plot,
    names="name",
    ids="idx",
    parents="parent_idx",
    color="kss",
    color_continuous_scale="blues",
)
fig.update_traces(root_color="lightgrey")
# fig.show()
fig.write_html(f"example_partitioner_lots_modifications3_kss.html")


############# Outline the cells according to the direction of change in counts
df_plot.loc[df_plot.count_diff < 0, "count_dir"] = "red"
df_plot.loc[df_plot.count_diff == 0, "count_dir"] = "lightgrey"
df_plot.loc[df_plot.count_diff > 0, "count_dir"] = "green"

fig = px.treemap(
    data_frame=df_plot,
    names="name",
    ids="idx",
    parents="parent_idx",
    color="kss",
    color_continuous_scale="blues",
)
# fig.update_traces(marker_line_width=4, marker_line={'color':df_plot.count_dir}) #alternatively, set outline to match the color
fig.update_traces(
    insidetextfont={
        "color": df_plot.count_dir,  # set the text color to same
        # 'size':[14, 18, 18, 18, 18] #can set the text size if desired
    },
    root_color="lightgrey",
)
# fig.show()
fig.write_html(f"example_partitioner_lots_modifications4_outline.html")


################################################################################
################################# Alternatives #################################
################################################################################
# For the most part, sunburst and icicle plots take the same arguments and
# behave, though they're not as space-efficient visually.

kp = KDQTreePartitioner(count_ubound=25)
df = np.random.sample([50, 3])
df2 = np.random.sample([50, 3])
_ = kp.build(df)
_ = kp.fill(df2, "fill1")
df_plot = kp.to_plotly_dataframe("build", "fill1")
df_plot

df_plot.loc[df_plot.count_diff < 0, "count_dir"] = "red"
df_plot.loc[df_plot.count_diff == 0, "count_dir"] = "lightgrey"
df_plot.loc[df_plot.count_diff > 0, "count_dir"] = "green"

############# Sunburst plot
fig = px.sunburst(
    data_frame=df_plot,
    names="name",
    ids="idx",
    parents="parent_idx",
    values="cell_count",
)
# fig.update_traces(marker_line_width=4, marker_line={'color':df_plot.count_dir})
fig.update_traces(
    insidetextfont={"color": df_plot.count_dir},
    root_color="lightgrey",
)
# fig.show()
fig.write_html(f"example_partitioner_lots_alternatives_sunburst.html")


############# Icicle plot
fig = px.icicle(
    data_frame=df_plot,
    names="name",
    ids="idx",
    parents="parent_idx",
    values="cell_count",
)
# fig.update_traces(marker_line_width=4, marker_line={'color':df_plot.count_dir})
fig.update_traces(
    insidetextfont={"color": df_plot.count_dir},
    root_color="lightgrey",
)
# fig.show()
fig.write_html(f"example_partitioner_lots_alternatives_icicle.html")
