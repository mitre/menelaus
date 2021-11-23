import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
from molten.drift_detector import DriftDetector
from molten.distribution.kl_divergence import kl_divergence


class KdqTree(DriftDetector):
    """kdqTree is a drift detection algorithm which detects drift via the
    Kullback-Leibler divergence, calculated after partitioning the data space
    via constructing a k-d-quad-tree (kdq-tree). A reference window of initial
    data is compared to a test window of later data. The Kullback-Leibler
    divergence between the empirical distributions of the reference and test
    windows is calculated, and drift is alarmed when a threshold is reached.

    A kdqtree is a combination of k-d trees and quad-trees; it is a binary tree
    (k-d) whose nodes contain square cells (quad) which are created via
    sequential splits along each dimension. This structure allows the
    calculation of the K-L divergence for continuous distributions, as the K-L
    divergence is defined on probability mass functions. The reference window is
    used to construct a kdq-tree via theory of types, and the data in both the
    reference and test window are binned into this kdq-tree. The K-L divergence
    can then be calculated between the two windows.

    The threshold for drift is determined using the desired alpha level, by a
    bootstrap estimate for the critical value of the K-L divergence, drawing
    num_bootstrap_samples samples, 2*window_size times, from the reference
    window.

    Additionally, the Kulldorff spatial scan statistic, which is a special case
    of the KL-divergence, can be calculated at each node of the kdq-tree, which
    gives a measure of the regions of the data space which have the greatest
    divergence between the reference and test windows. This can be used to
    visualize which regions of data space have the greatest drift, implemented
    as kdqTreeDetector.drift_visualization. Note that these statistics are
    specific to the partitions of the data space by the kdq-tree, rather than
    (necessarily) the maximally different region in general.

    Note also that this algorithm could be used with other types of trees; the
    reference paper and this implementation use kdq-trees.

    Ref. T. Dasu, S. Krishnan, S. Venkatasubramanian, and K. Yi, “An
    information-theoretic approach to detecting changes in multidimensional
    data streams,” in Proc. Symp. the Interface of Statistics,
    Computing Science, and Applications. Citeseer, 2006, Conference
    Proceedings, pp. 1–24.
    """

    def __init__(
        self,
        window_size,
        min_points_in_bin=100,
        num_bootstrap_samples=500,
        gamma=0.05,
        alpha=0.01,
        verbose=True,
    ):
        """
        Args:
            window_size ([type]): the minimum number of samples required to test
                whether drift has occurred, equivalent to "maximum number of points
                in cell" in Dasu (2006)
            min_points_in_bin (int, optional): size of window over which kdq
                detects drift. Defaults to 100.
            num_bootstrap_samples (int, optional): the number of bootstrap
                samples to use to approximate the empirical distributions.
                Equivalent to kappa in Dasu (2006). Dasu recommends 500-1000
                samples. Defaults to 500.
            gamma (float, optional): Persistence factor. Defaults to 0.05.
            alpha (float, optional): Achievable significance level. Defaults to 0.01.
            verbose (bool, optional): prints progress to console. Defaults to True.
        """
        super().__init__()
        self._min_points_in_bin = min_points_in_bin
        self._window_size = window_size
        self._num_bootstrap_samples = num_bootstrap_samples
        self._gamma = gamma
        self._verbose = verbose
        self._alpha = alpha

        self._build_reference_and_test = True

        self._kdq_tree_nodes = None
        self._alphabet = None

        self.window_data = {"reference": pd.DataFrame(), "test": pd.DataFrame()}
        self._critical_distance = None

        self.drift_tracker = {"dist": [], "critical_distance": [], "id": []}
        self.drift_location = {
            "spatial_scan_statistic": [],
            "kdq_tree_nodes": [],
            "id": [],
        }

        self._c = 0
        self._iter = 0

    def update(self, next_obs, *args, **kwargs):  # pylint: disable=arguments-differ
        """Update the detector with a new sample.

        Args:
            next_obs: the next observation in the stream, as a dataframe.
        """
        if self.drift_state == "drift":
            super().reset()

        super().update()

        if self._verbose:
            print(f"Row Index: {next_obs.index.values[0]}")

        if self._build_reference_and_test:
            if len(self.window_data["reference"]) < self._window_size:
                self.window_data["reference"] = self.window_data["reference"].append(
                    next_obs
                )

            elif len(self.window_data["test"]) < self._window_size:
                self.window_data["test"] = self.window_data["test"].append(next_obs)

                if len(self.window_data["test"]) == self._window_size:
                    self._build_reference_and_test = False

                    self._kdq_tree_nodes = self._kdq_tree_build_nodes(
                        data_input=self.window_data["reference"],
                        min_points=self._min_points_in_bin,
                        verbose=self._verbose,
                    )

                    # Merge in the bins for both the reference and test windows
                    kdqbins = self._kdq_tree_build_bins(self._kdq_tree_nodes)
                    self.window_data["reference"] = self.window_data["reference"].merge(
                        kdqbins, how="left", left_index=True, right_on="id"
                    )
                    self.window_data["reference"].set_index("id", inplace=True)
                    self.window_data["reference"].index.name = self.window_data[
                        "test"
                    ].index.name

                    self._alphabet = list(kdqbins.bin.unique())

                    test_bins = []
                    for i in range(len(self.window_data["test"])):
                        test_bins.append(
                            self._kdq_tree_binner(
                                kdq_tree_nodes=self._kdq_tree_nodes,
                                data_point=self.window_data["test"].iloc[[i]],
                            )
                        )
                    self.window_data["test"] = self.window_data["test"].assign(
                        bin=test_bins
                    )

                    # Build bootstrap samples from the reference window These
                    # are used to define the critical region for the divergence
                    # metric, given the desired alpha, by using the appropriate
                    # percentile among calculated divergences comparing the
                    # first half of the bootstrap samples to the second.
                    bootstrap_samples = self._bootstrapping(
                        self.window_data["reference"]["bin"].tolist(),
                        2 * self._window_size,
                        self._num_bootstrap_samples,
                    )
                    bootstrap_types = [
                        [
                            self._compute_type(x[: len(x) // 2], self._alphabet),
                            self._compute_type(
                                list(x[(len(x) // 2) + 1 :]), self._alphabet
                            ),
                        ]
                        for x in bootstrap_samples
                    ]
                    bootstrap_kl_distances = [
                        kl_divergence(x[0], x[1]) for x in bootstrap_types
                    ]
                    bootstrap_kl_distances.sort()

                    self._critical_distance = bootstrap_kl_distances[
                        int(np.ceil(self._num_bootstrap_samples * self._alpha))
                    ]

        else:
            next_obs = next_obs.assign(
                bin=self._kdq_tree_binner(
                    kdq_tree_nodes=self._kdq_tree_nodes, data_point=next_obs
                )
            )

            self.window_data["test"] = (
                self.window_data["test"].iloc[1:, :].append(next_obs)
            )

            type1 = self._compute_type(
                self.window_data["reference"]["bin"].tolist(), self._alphabet
            )
            type2 = self._compute_type(
                self.window_data["test"]["bin"].tolist(), self._alphabet
            )

            dist = kl_divergence(type1, type2)

            self.drift_tracker["dist"].append(dist)
            self.drift_tracker["critical_distance"].append(self._critical_distance)
            self.drift_tracker["id"].append(next_obs.index.values[0])

            if self._verbose:
                print(f"Distance: {str(round(dist, 4))}")
                print("------------------------------")

            if dist > self._critical_distance:
                self._c = self._c + 1

                if self._c > self._gamma * self._window_size:

                    if self._verbose:
                        print("Change detected!")

                    self.drift_state = "drift"

                    self.drift_location["id"].append(next_obs.index.values[0])

                    kdqbins_filtered = self._kdq_tree_build_bins(
                        self._kdq_tree_nodes[self._kdq_tree_nodes["depth"] < 9]
                    )

                    self.drift_location["spatial_scan_statistic"].append(
                        self._kulldorff_spatial_scan_statistic(
                            alphabet=list(kdqbins_filtered.bin.unique()),
                            bins_w1=[
                                x
                                for x, y in zip(
                                    list(kdqbins_filtered.bin),
                                    list(kdqbins_filtered.id),
                                )
                                if y in self.window_data["reference"].index.tolist()
                            ],
                            bins_w2=[
                                x
                                for x, y in zip(
                                    list(kdqbins_filtered.bin),
                                    list(kdqbins_filtered.id),
                                )
                                if y in self.window_data["test"].index.tolist()
                            ],
                        ).sort_values(
                            "kulldorff_spatial_scan_statistic", ascending=False
                        )
                    )

                    self.drift_location["kdq_tree_nodes"].append(self._kdq_tree_nodes)

                    self._build_reference_and_test = True
                    self.window_data["reference"] = self.window_data["test"].drop(
                        columns=["bin"]
                    )
                    self.window_data["test"] = pd.DataFrame()
                    self._c = 0
            else:
                self._c = 0

    def drift_visualization(self, id_date_df=None, save_fig=None):
        """
        Args:
            id_date_df: (Default value = None)
            save_fig: (Default value = None)
        """
        if id_date_df is None:
            kl_distance_ts = pd.DataFrame(
                self.drift_tracker, index=range(len(self.drift_tracker["dist"]))
            )

            with plt.style.context("fivethirtyeight"):
                plt.figure(figsize=(20, 8))
                plt.plot(
                    kl_distance_ts["id"],
                    kl_distance_ts["dist"],
                    linestyle="-",
                    marker=",",
                    zorder=0,
                    linewidth=0.85,
                )
                plt.plot(
                    kl_distance_ts["id"],
                    kl_distance_ts["critical_distance"],
                    linestyle="--",
                    marker=",",
                    linewidth=1,
                    color="goldenrod",
                    zorder=1,
                )
                plt.vlines(
                    x=kl_distance_ts.loc[
                        kl_distance_ts.dist >= kl_distance_ts.critical_distance, :
                    ].id,
                    ymin=min(kl_distance_ts["dist"]),
                    ymax=max(kl_distance_ts["critical_distance"]),
                    linestyle="-",
                    color="darkred",
                    alpha=0.05,
                    linewidth=0.5,
                    zorder=2,
                )
                plt.title("Kullback-Leibler Divergence", fontsize=16, y=1.01)
                plt.suptitle("KDQ Tree Drift Detection", fontsize=24, y=0.97)
                plt.xlabel("Time")
                plt.ylabel("Divergence Value")

                ax_out = plt.gca()
                ax_out.axes.yaxis.set_ticks([])

                if save_fig is not None:
                    plt.savefig(f"{save_fig}")

                plt.show()
        else:
            kl_distance_ts = pd.DataFrame(
                self.drift_tracker, index=range(len(self.drift_tracker["dist"]))
            ).merge(id_date_df, how="left", on="id")
            with plt.style.context("fivethirtyeight"):
                plt.figure(figsize=(20, 8))
                plt.plot_date(
                    kl_distance_ts.date,
                    kl_distance_ts["dist"],
                    linestyle="-",
                    marker=",",
                    zorder=0,
                    linewidth=0.85,
                )
                plt.plot_date(
                    kl_distance_ts.date,
                    kl_distance_ts["critical_distance"],
                    linestyle="--",
                    marker=",",
                    linewidth=1,
                    color="goldenrod",
                    zorder=1,
                )
                plt.vlines(
                    x=kl_distance_ts.loc[
                        kl_distance_ts.dist >= kl_distance_ts.critical_distance, :
                    ].date,
                    ymin=min(kl_distance_ts["dist"]),
                    ymax=max(kl_distance_ts["critical_distance"]),
                    linestyle="-",
                    color="darkred",
                    alpha=0.05,
                    linewidth=0.5,
                    zorder=2,
                )
                plt.title("Kullback-Leibler Divergence", fontsize=16, y=1.01)
                plt.suptitle("KDQ Tree Drift Detection", fontsize=24, y=0.97)
                plt.xlabel("Time")
                plt.ylabel("Divergence Value")

                ax_out = plt.gca()
                ax_out.axes.yaxis.set_ticks([])

                if save_fig is not None:
                    plt.savefig(f"{save_fig}")

                plt.show()

    def drift_location_visualization(self):
        """ """
        if len(self.drift_location["spatial_scan_statistic"]) == 0:
            print("No drift detected")
            return None

        figs = []
        for i in range(len(self.drift_location["spatial_scan_statistic"])):
            spatial_scan_statistic = self.drift_location["spatial_scan_statistic"][i]
            kdq_tree_nodes = self.drift_location["kdq_tree_nodes"][i]
            current_id = self.drift_location["id"][i]

            bin_df = pd.DataFrame(
                columns=["kdqTree"]
                + ["split" + str(x) for x in range(11)[1:]]
                + ["spatial_scan_statistic"]
            )
            bins = spatial_scan_statistic["bin"].tolist()

            for j in range(len(bins)):

                tmp_bin = bins[j]
                tmp_statistics = spatial_scan_statistic[
                    "kulldorff_spatial_scan_statistic"
                ].tolist()[j]

                bin_path = self._tree_parser(kdq_tree_nodes, tmp_bin)

                nodes = ["kdqTree"] + bin_path.split(", ") + [f"Bin {tmp_bin}"]
                if len(nodes) < 11:
                    nodes = nodes + [None] * (11 - len(nodes))

                bin_df.loc[bin_df.shape[0]] = nodes + [tmp_statistics]

            bin_df_columns = bin_df.columns
            for col in bin_df_columns:
                if bin_df[[col]].isnull().all().values[0]:
                    bin_df.drop(columns=[col], inplace=True)

            figs.append(
                px.treemap(
                    bin_df,
                    path=["kdqTree"]
                    + ["split" + str(x) for x in range(bin_df.shape[1] - 2)[1:]],
                    color="spatial_scan_statistic",
                    color_continuous_scale="blues",
                    title=f"Drift Location ID {current_id}",
                )
            )
        return figs

    def _bootstrapping(self, list_input: list, sample_size: int, num_samples: int):
        """Computes bootstrap samples from a given list of objects

        Args:
            list_input (list): List of objects to compute bootstrap samples from
            sample_size (int): Size of sample to take for each bootsrap. Sample q/ replacement
            num_samples (int): Number of bootstrap samples to conduct
            list_input (list):
            sample_size (int):
            num_samples (int):

        Returns:
            List of bootstrap samples.

        """

        bootstrap_samples = []

        for _ in range(num_samples):
            bootstrap_samples.append(
                np.random.choice(list_input, size=sample_size).tolist()
            )

        return bootstrap_samples

    def _compute_type(self, multiset: list, alphabet: list):
        """Computes the type for each letter in an alphabet, according to the theory of types

        Args:
            multiset (list): Multiset of letters in alphabet
            alphabet (list): List of all letters in alphabet

        Returns:
            The Type of multiset

        """

        computed_type = [
            (multiset.count(a) + 0.5) / (len(multiset) + len(alphabet) / 2)
            for a in alphabet
        ]

        return computed_type

    def _kdq_tree_binner(self, kdq_tree_nodes, data_point):
        """

        Args:
            kdq_tree_nodes:
            data_point:

        Returns:

        """
        next_id = 1
        leaf_node = False
        while not leaf_node:
            current_id = next_id
            row_select = kdq_tree_nodes[(kdq_tree_nodes.node_id == current_id)]

            # Move to the left cut
            next_id = 2 * row_select.node_id.values[0]

            # If it is a right cut, we go to the 2*n + 1 node
            if (
                data_point[[row_select.axis.values[0]]].values[0]
                > row_select.cutpoint.values[0]
            ):
                next_id += 1

            # If this proposed next node does not exist, then we have reached a leaf node
            # Similarly, we could check the type of node and see if it is a leaf of any type
            if next_id not in kdq_tree_nodes.node_id.values:
                bin_id = row_select.bin_id.values[0]
                leaf_node = True

        return bin_id

    def _kdq_tree_build_bins(self, clean_results):
        """Builds kdqTree bins from nodes and splits defined in the output of
        _kdq_tree_build_nodes()

        Args: clean_results (DataFrame): Dataframe of nodes and splits produced
            from _kdq_tree_build_nodes()

        Returns: DataFrame of 2 columns - observation IDs and bin IDs

        """
        ids = []
        bins = []
        for col in clean_results.columns[7:]:
            tmp_df = clean_results.loc[clean_results[col] == True, :][
                ["depth", "bin_id", col]
            ].sort_values(by="depth", ascending=False)
            ids.append(col[1:])
            bins.append(tmp_df.bin_id.tolist()[0])

        kdqbins = pd.DataFrame({"id": [int(x) for x in ids], "bin": bins}).sort_values(
            by="bin"
        )

        return kdqbins

    def _nested_dict_to_dataframe(self, nested_dict):
        """Unnests a nested node dictionary into a cleaned dataframe

        Args:
          nested_dict (dict): Nested dictionary of nodes output from kdqTreeSplits()

        Returns:
          Unnested pandas dataframe of nodes and splits

        """

        out = pd.DataFrame()

        if isinstance(nested_dict, dict):
            for value in nested_dict.values():
                out = pd.concat([out, self._nested_dict_to_dataframe(value)])

        elif isinstance(nested_dict, pd.DataFrame):
            out = nested_dict

        return out

    def _kdq_tree_splits(
        self,
        data_input: pd.DataFrame,
        original_ids: list,
        node: dict,
        node_id: int,
        depth: int,
        axis: int,
        min_points: int,
        max_value: int,
        min_value: int,
        verbose=True,
    ):
        """Recursive function to create splits in a tree at a given node

        Args:
            data_input (pd.DataFrame): Set of points in data
            original_ids (list): Original list of IDs corresponding to each row of
                data_input at the root
            node (dict): A dictionary containing keys left_node, right_node, and
                results
            node_id (int): ID of current node to be included in the dataframe of
                `results` value
            depth (int): Current depth of the node we are creating a split at
            axis (int): Defines the index of the column we should be making a
                split on
            min_points (int): Minimum number of points allowed in each bin
            max_value (int): Maximum value in the given axis
            min_value (int): Minimum value in the given axis
            verbose: Determines if intermediate output should be printed to
                console. Defaults to True.

        Returns:
            Node dictionary

        """

        if verbose:
            print(f"Node ID: {str(node_id)}")

        if data_input.shape[0] > min_points:
            # depth = depth+1

            # axis = ((depth - 1) % (data_input.shape[1] - 1)) + 1

            cutpoint = (max_value + min_value) / 2

            left_of_cutpoint = data_input.iloc[:, axis] < cutpoint

            bools_df = pd.DataFrame(
                [(x in list(data_input.index)) for x in original_ids]
            ).transpose()
            bools_df.columns = ["d" + str(x) for x in original_ids]

            points_in_left_cell = left_of_cutpoint.sum()
            points_in_right_cell = data_input.shape[0] - points_in_left_cell

            if points_in_left_cell < min_points and points_in_right_cell < min_points:
                node_type = "Leaf Node"

            elif points_in_left_cell < min_points:
                node_type = "Left Leaf Node"

            elif points_in_right_cell < min_points:
                node_type = "Right Leaf Node"

            else:
                node_type = "Node"

            node["results"] = pd.concat(
                [
                    pd.DataFrame(
                        {
                            "node_id": node_id,
                            "depth": depth,
                            "axis": list(data_input.columns)[axis],
                            "num_points": data_input.shape[0],
                            "cutpoint": cutpoint,
                            "type": node_type,
                        },
                        index=[0],
                    ),
                    bools_df,
                ],
                axis=1,
            )

            depth = depth + 1
            axis_next = (depth % (data_input.shape[1] - 1)) + 1
            max_next = np.max(data_input.iloc[:, axis_next])
            min_next = np.min(data_input.iloc[:, axis_next])

            node["left_child"] = self._kdq_tree_splits(
                data_input.loc[left_of_cutpoint, :],
                original_ids=original_ids,
                node=node.copy(),
                node_id=2 * node_id,
                depth=depth,
                axis=axis_next,
                max_value=max_next,
                min_value=min_next,
                min_points=min_points,
                verbose=verbose,
            )

            node["right_child"] = self._kdq_tree_splits(
                data_input.loc[[not x for x in left_of_cutpoint], :],
                original_ids=original_ids,
                node=node.copy(),
                node_id=(2 * node_id) + 1,
                depth=depth,
                axis=axis_next,
                max_value=max_next,
                min_value=min_next,
                min_points=min_points,
                verbose=verbose,
            )

            return node

    def _kdq_tree_build_nodes(
        self, data_input: pd.DataFrame, min_points=200, verbose=True
    ):
        """Build out the nodes of the kdqTree

        Args:
            data_input (pd.DataFrame): Set of points in data
            min_points (int): Minimum number of points allowed in each bin.
                Default value = 200.
            verbose: Determines if intermediate output should be printed to
                console. Default value = True.

        Returns:
            Dataframe of splits in kdqTree

        """

        original_ids = list(data_input.index)

        depth = 0
        node = {"left_child": None, "right_child": None}

        bools_df = pd.DataFrame([True] * data_input.shape[0]).transpose()
        bools_df.columns = ["d" + str(x) for x in original_ids]

        results = pd.concat(
            [
                pd.DataFrame(
                    {
                        "node_id": 1,
                        "depth": 0,
                        "axis": None,
                        "num_points": data_input.shape[0],
                        "cutpoint": None,
                        "type": "Node",
                    },
                    index=[0],
                ),
                bools_df,
            ],
            axis=1,
        )

        node["results"] = results

        if verbose:
            print("Building splits in kdqtree")
        nested_dict_results = self._kdq_tree_splits(
            data_input=data_input,
            original_ids=original_ids,
            node=node,
            node_id=1,
            depth=depth,
            axis=1,
            max_value=np.max(data_input.iloc[:, 1]),
            min_value=np.min(data_input.iloc[:, 1]),
            min_points=min_points,
            verbose=verbose,
        )

        clean_results = self._nested_dict_to_dataframe(nested_dict_results)
        clean_results = clean_results.sort_values("node_id").assign(
            bin_id=range(clean_results.shape[0])
        )
        clean_results = clean_results[
            clean_results.columns.tolist()[-1:] + clean_results.columns.tolist()[:-1]
        ]

        clean_results.index = range(clean_results.shape[0])

        return clean_results

    def _kulldorff_spatial_scan_statistic(self, alphabet, bins_w1, bins_w2):
        """Computes Kulldorf Spatial Scan Statistic between two bins in an
        alphabet

        Args:
            alphabet (list): List of all elements in set (alphabet)
            bins_w1 (list): List of bins comprising window 1
            bins_w2 (list): List of bins comprising window 2

        Returns:
            DataFrame containing the input alphabet and the Kulldorf Spatial
                Scan Statistic for each element

        """

        len_w1 = len(bins_w1)
        len_w2 = len(bins_w2)

        w1_in_alpha = [bins_w1.count(alpha) for alpha in alphabet]
        w2_in_alpha = [bins_w2.count(alpha) for alpha in alphabet]

        statistic = [
            ((w1_counts + 0.5) * np.log((w1_counts + 0.5) / (w2_counts + 0.5)))
            + (
                (len_w1 - w1_counts + 0.5)
                * np.log((len_w1 - w1_counts + 0.5) / (len_w2 - w2_counts + 0.5))
            )
            - ((len_w1 + 1) * np.log((len_w1 + 1) / (len_w2 + 1)))
            for w1_counts, w2_counts, in zip(w1_in_alpha, w2_in_alpha)
        ]

        return pd.DataFrame(
            {"bin": alphabet, "kulldorff_spatial_scan_statistic": statistic}
        )

    def _tree_parser(self, kdq_tree_nodes, bin_id):
        """Parse tree to bin into text

        Args:
            kdq_tree_nodes (pd.DataFrame): Dataframe of kdqTree nodes
            bin_id (int): ID of the bin to which we should parse the tree

        Returns:
            String of text that follows path from root to bin

        """

        node_ids = [
            kdq_tree_nodes[kdq_tree_nodes.bin_id == bin_id]["node_id"].tolist()[0]
        ]

        not_root = True
        tmp_id = node_ids[0]

        while not_root:
            if tmp_id % 2 == 0:
                tmp_id = tmp_id // 2
                operator = "<"
            else:
                tmp_id = (tmp_id - 1) // 2
                operator = ">"

            node_ids.append(tmp_id)

            if tmp_id == 1:
                not_root = False

        split_string = ""
        for node_id in node_ids[::-1]:
            tmp_row = kdq_tree_nodes[kdq_tree_nodes.node_id == node_id]

            if node_id % 2 == 0:
                operator = "<"
            else:
                operator = ">"

            split_string = (
                split_string
                + f"{tmp_row['axis'].tolist()[0]} {operator} \
                    {round(tmp_row['cutpoint'].tolist()[0],2)}, "
            )

        split_string = split_string[:-2]

        return split_string
