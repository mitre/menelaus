import numpy as np
import pandas as pd
import scipy.stats


class KDQTreePartitioner:
    """This class encapsulates a KDQ Tree for partitioning data which runs
    through drift detection algorithms, as described by Dasu et. al. (2006).

    The general algorithm to build a tree performs until a stop condition:

        * Identify axis to split upon.
        * Identify midpoint at axis.
        * Split into lower/upper halves using midpoint.
        * Recursively split sub-trees.

    This class can be used to build a tree, 'fill' a tree (send new
    data to be forcibly partitioned according to an existing build),
    access leaf nodes, and perform visualization / printing. There
    are also convenience functions to calculate distances between
    a built and filled tree.

    Attributes:
        count_ubound (int): upper bound of count of data points to be
            partitioned into any one cell
        cutpoint_proportion_lbound (float): min. proportion of all features'
            range values that any constructed cell size must satisfy
        node (KDQTreeNode): reference to root node of build tree
        leaves (list(KDQTreeNode)): list of leaf nodes of build tree
    """

    def __init__(self, count_ubound=200, cutpoint_proportion_lbound=0.25):
        """
        Args:
            count_ubound (int, optional): upper bound of count of data points to
                be partitioned into any one cell. Default ``200``.
            cutpoint_proportion_lbound (float, optional): min. proportion of all
                features' range values that any constructed cell size must
                satisfy. Default ``0.25``.
        """
        self.count_ubound = count_ubound
        self.cutpoint_proportion_lbound = cutpoint_proportion_lbound
        self.node = None
        self.leaves = []

    def build(self, data):
        """
        Creates a new kdqTree by partitioning the data into square nodes in
        the feature-space.

        Args:
            data (numpy.array): data to be partitioned

        Returns:
            KDQTreeNode: root node of constructed KDQ-Tree
        """
        if len(data.shape) <= 1:
            return None
        _, num_cols = data.shape
        min_cutpoint_sizes = [
            int(self.cutpoint_proportion_lbound * np.ptp(data[:, axis]))
            for axis in range(num_cols)
        ]
        self.node = KDQTreeNode.build(
            data, self.count_ubound, min_cutpoint_sizes, self.leaves
        )
        return self.node

    def reset(self, value=0, tree_id="build"):
        """Sets the counts for the given tree_id to the given value.

        Args:
            value (int, optional): value to be written for each node. Default
                ``0``.
            tree_id (str, optional): identifier for set of counts to be
                overwritten. Default ``'build'``.
        """
        KDQTreeNode.reset(self.node, value=value, tree_id=tree_id)

    def fill(self, data, tree_id, reset=False):
        """For a new sample data, partition it according to the pre-existing tree
        structure. Its counts will be added to those for the tree_id index.

        Args:
            data (numpy.array): new data to be partitioned
            tree_id (str): identifier for new data counts to be stored
            reset (bool, optional): if ``True``, counts for this ``tree_id``
                will be overwritten with ones calculated from current sample,
                else added. Default ``False``

        Returns:
            KDQTreeNode: root node of KDQ-Tree
        """
        if self.node is None or len(data.shape) <= 1:
            return None
        KDQTreeNode.fill(data, self.node, self.count_ubound, tree_id, reset)
        return self.node

    def leaf_counts(self, tree_id):
        """Return the counts for each leaf of the tree at ``tree_id``.

        Args:
            tree_id (str): identifier of tree for which to return counts

        Returns:
            list: list of counts at leaves of KDQ-Tree
        """
        ret = None
        if self.leaves:
            ret = [
                leaf.num_samples_in_compared_subtrees[tree_id] for leaf in self.leaves
            ]
        return ret

    def kl_distance(self, tree_id1, tree_id2):
        """For the empirical distributions defined by the counts at tree_id1 and
        tree_id2, calculate the Kullback-Leibler divergence from tree_id2 to
        tree_id1.

        Args:
            tree_id1 (str): identifier for reference tree
            tree_id2 (str): identifier for comparison tree

        Returns:
            float: Kullback-Leibler divergence among trees
        """
        if self.leaves == []:
            return None
        counts1 = self.leaf_counts(tree_id1)
        counts2 = self.leaf_counts(tree_id2)
        hist1 = KDQTreePartitioner._distn_from_counts(counts1)
        hist2 = KDQTreePartitioner._distn_from_counts(counts2)
        distance = scipy.stats.entropy(hist1, hist2)
        return distance

    @staticmethod
    def _distn_from_counts(counts):
        """Calculate an empirical distribution across the alphabet defined by
        the bins (here, indices in the counts array), using Dasu's correction.

        Args:
            counts (numpy.array): array of counts

        Returns:
            numpy.array: array of frequencies
        """
        total = np.sum(counts)
        hist = np.array(counts) + 0.5
        hist = hist / (total + len(hist) / 2)
        return hist

    def to_plotly_dataframe(
        self, tree_id1="build", tree_id2=None, max_depth=None, input_cols=None
    ):
        """Generates a dataframe containing information about the kdqTree's structure
        and some node characteristics, intended for use with plotly. DataFrame columns
        capture:

            * ``name``, a label corresponding to which feature this split is on.
            * ``idx``, a unique ID for the node, to pass to
                ``plotly.express.treemap``'s ``id`` argument.
            * ``parent_idx``, the ID of the node's parent.
            * ``cell_count``, how many samples are in this node in the reference
                tree.
            * ``depth``, how deep the node is in the tree.
            * ``count_diff``, if ``tree_id2`` is specified, the change in counts
                from the reference tree.
            * ``kss``, the Kulldorff Spatial Scan Statistic for this node,
                defined as the KL-divergence for this node between the reference
                and test trees, using the individual node and all other nodes
                combined as the bins for the distributions.

        Args:
            tree_id1 (str): identifier for reference tree. Default ``'build'``
            tree_id2 (str): identifier for test tree. Default ``None``.
            max_depth (int, optional): tree depth up to which the method
                recurses. Default ``None``.
            input_cols (list, optional): list of column names for the input
                data. Default ``None``.

        Returns:
            pandas.DataFrame: where each row represents a node
        """
        arr = []
        KDQTreeNode.as_flattened_array(
            self.node,
            tree_id1=tree_id1,
            tree_id2=tree_id2,
            output=arr,
            input_cols=input_cols,
        )
        df = pd.DataFrame.from_dict(arr)
        if max_depth:
            df = df[df.depth <= max_depth]
        if tree_id2 is not None:
            # could be more efficient by avoiding the use of pd.apply?
            kss_counts = pd.DataFrame(
                {
                    "node_count_ref": df["cell_count"],
                    "node_count_test": df["count_diff"] + df["cell_count"],
                }
            )
            test_max = kss_counts["node_count_test"].max()
            ref_max = kss_counts["node_count_ref"].max()
            df["kss"] = kss_counts.apply(
                KDQTreePartitioner._calculate_kss, args=(ref_max, test_max), axis=1
            )
        return df

    @staticmethod
    def _calculate_kss(df, ref_max, test_max):
        """For the given node's test counts and reference counts, the Kulldorff
        Spatial Scan Statistic can be calculated by treating that node and its
        complement (all the other nodes in the tree, described by ref_max and
        test_max) as two bins in an empirical distribution, and then calculating
        the Kullback-Leibler divergence from the test to the reference.

        Args:
            df (pandas.DataFrame): single-row dataframe containing test and
                reference counts
            ref_max (int): total number of observations in reference sample
            test_max (int): total number of observations in test sample

        Returns:
            float: KSS from test data to reference data
        """
        ref_dist = KDQTreePartitioner._distn_from_counts(
            np.array([df["node_count_ref"], ref_max - df["node_count_ref"]])
        )
        test_dist = KDQTreePartitioner._distn_from_counts(
            np.array([df["node_count_test"], test_max - df["node_count_test"]])
        )
        return scipy.stats.entropy(ref_dist, test_dist)


class KDQTreeNode:
    """
    This class represents a node in the KDQ Tree data structure described above.
    Its static methods provide the engine for recursively building, filling, and
    displaying trees.

    Build / fill trees are identified by a tree ID. During initialization, a
    node typically takes ``num_samples_in_compared_subtrees`` (among other
    values needed for building), which is a ``dict`` containing counts for each
    tree ID.

    Each node stores the number of data points contained within itself and its
    subtrees as a value in this ``dict``. A ``tree_id`` key identifies which
    build those counts are associated with (e.g., the first call to ``build()``
    stores counts in the ``node.num_samples_in_compared_subtrees['build']`` by
    default). Subsequent calls to ``fill()`` with new data store additional
    counts for the node, according to the ID the user passes in.

    Attributes:
        num_samples_in_compared_subtrees (dict): number of data points, by tree
            ID, contained within node and its children
        axis (int): axis at which to build/fill (for recursive construction)
        midpoint_at_axis (float): midpoint at provided axis
        left (KDQTreeNode): left child
        right (KDQTreeNode): right child
    """

    def __init__(
        self, num_samples_in_compared_subtrees, axis, midpoint_at_axis, left, right
    ):
        """
        Args:
            num_samples_in_compared_subtrees (dict): number of data points, by
                tree ID, contained within node and its children
            axis (int): axis at which to build/fill (for recursive construction)
            midpoint_at_axis (float): midpoint at provided axis
            left (KDQTreeNode): left child
            right (KDQTreeNode): right child
        """
        self.num_samples_in_compared_subtrees = num_samples_in_compared_subtrees
        self.axis = axis
        self.midpoint_at_axis = midpoint_at_axis
        self.left = left
        self.right = right

    @staticmethod
    def build(data, count_ubound, min_cutpoint_sizes, leaves, depth=0):
        """Creates a new kdqTree by partitioning the data into square nodes in
        the feature-space.

        Args:
            data (numpy.array): data to be partitioned
            count_ubound (int): upper bound of count of points to be put into
                any 1 cell
            min_cutpoint_sizes (list): minimum sizes of features that a leaf
                can have
            leaves (list): list of leaf nodes
            depth (int, optional): current depth of tree. Default ``0``.

        Returns:
            KDQTreeNode: root node of tree
        """
        n, m = data.shape
        if n == 0 or m == 0:
            return None
        axis = depth % m
        min_value_at_axis = np.min(data[:, axis])
        midpoint_at_axis = min_value_at_axis + (np.ptp(data[:, axis]) / 2)
        new_cell_size = midpoint_at_axis - min_value_at_axis
        if (
            n <= count_ubound
            or np.unique(data).size <= count_ubound
            or new_cell_size <= min_cutpoint_sizes[axis]
        ):
            leaf = KDQTreeNode({"build": n}, None, None, None, None)
            leaves.append(leaf)
            return leaf
        upper_data = data[data[:, axis] > midpoint_at_axis]
        lower_data = data[data[:, axis] <= midpoint_at_axis]
        total_points = upper_data.shape[0] + lower_data.shape[0]
        node = KDQTreeNode(
            {"build": total_points},
            axis=axis,
            midpoint_at_axis=midpoint_at_axis,
            left=KDQTreeNode.build(
                lower_data, count_ubound, min_cutpoint_sizes, leaves, depth + 1
            ),
            right=KDQTreeNode.build(
                upper_data, count_ubound, min_cutpoint_sizes, leaves, depth + 1
            ),
        )
        return node

    @staticmethod
    def fill(data, node, count_ubound, tree_id, reset=False):
        """
        For a new sample, partition it according to the pre-existing subtree
        rooted at node. Its counts will be added to those for the ``tree_id``
        index.

        Args:
            data (numpy.array): new data to be partitioned into existing tree
            node (KDQTreeNode): current node
            count_ubound (int): upper bound of points in any one cell
            tree_id (str): identifier for new data counts to be stored
            reset (bool, optional): if ``True``, counts for this ``tree_id``
                will be overwritten with ones calculated from current sample,
                else added. Default ``False``.
        """
        # case: no more nodes
        if node is None:
            return  # this line shows as dead to coverage statistics, but that should be a bug, since otherwise this function wouldn't terminate
        n = data.shape[0]
        axis = node.axis
        # basically, matches the return Node(n, None) case above
        if (node is not None) and node.axis is None:
            # update by ID
            if tree_id not in node.num_samples_in_compared_subtrees.keys() or reset:
                node.num_samples_in_compared_subtrees[tree_id] = n
            else:
                node.num_samples_in_compared_subtrees[tree_id] += n
            return
        # case: continue parsing
        midpoint_at_axis = node.midpoint_at_axis
        upper_data = data[data[:, axis] > midpoint_at_axis]
        lower_data = data[data[:, axis] <= midpoint_at_axis]
        total_points = upper_data.shape[0] + lower_data.shape[0]
        # update by ID
        if tree_id not in node.num_samples_in_compared_subtrees.keys() or reset:
            node.num_samples_in_compared_subtrees[tree_id] = total_points
        else:
            node.num_samples_in_compared_subtrees[tree_id] += total_points
        # recurse
        KDQTreeNode.fill(upper_data, node.right, count_ubound, tree_id, reset)
        KDQTreeNode.fill(lower_data, node.left, count_ubound, tree_id, reset)

    @staticmethod
    def reset(node, value, tree_id):
        """For the subtree rooted at ``node``, set the counts for ``tree_id`` equal to
        ``value``.

        Args:
            node (KDQTreeNode): root of subtree
            value (int): value to set counts to
            tree_id (str): identifier for tree
        """
        if node:
            node.num_samples_in_compared_subtrees[tree_id] = value
            KDQTreeNode.reset(node.left, value, tree_id)
            KDQTreeNode.reset(node.right, value, tree_id)

    @staticmethod
    def as_text(node, tree_id="build"):
        """Produce a text representation of the tree structure and its counts.

        Args:
            node (KDQTreeNode): root node of desired subtree
            tree_id (str, optional): identifier for desired subtree. Default ``build``.
        """
        # TODO - to avoid a recursive printing problem, this prints rather than storing an ongoing output string
        if node:
            print(f"\nsubtree count: {node.num_samples_in_compared_subtrees[tree_id]}")
            if node.left:
                print(f"\tleft: {node.left.num_samples_in_compared_subtrees[tree_id]}")
            if node.right:
                print(
                    f"\tright: {node.right.num_samples_in_compared_subtrees[tree_id]}"
                )
            KDQTreeNode.as_text(node.left, tree_id)
            KDQTreeNode.as_text(node.right, tree_id)

    @staticmethod
    def as_flattened_array(
        node,
        tree_id1,
        tree_id2,
        output=[],
        name="kdqTree",
        parent_idx=None,
        depth=0,
        input_cols=None,
    ):
        """Generates a list containing dicts with information about each node's
        structure for the tree rooted at node.

        Args:
            node (KDQTreeNode): root node of desired subtree
            tree_id1 (str): identifier for reference tree
            tree_id2 (str): identifier for test tree
            output (list, optional): list of dictionaries containing information
                about each node. Default ``[]``.
            name (str): name of root node. Default ``'kdqTree'``.
            parent_idx (int, optional): unique ID for parent of current node.
                Default ``None``.
            depth (int, optional): depth of current subtree. Default ``0``.
            input_cols (list, optional): list of column names for the input
                data. Default ``None``.
        """
        if node and (tree_id1 in node.num_samples_in_compared_subtrees.keys()):
            # basic plotting features
            current_data = {
                "name": name,
                "idx": id(node),
                "parent_idx": parent_idx,
                "cell_count": node.num_samples_in_compared_subtrees[tree_id1],
                "depth": depth,
            }
            # advanced plotting features
            if tree_id2:
                if tree_id2 in node.num_samples_in_compared_subtrees.keys():
                    count_diff = (
                        node.num_samples_in_compared_subtrees[tree_id2]
                        - node.num_samples_in_compared_subtrees[tree_id1]
                    )
                else:
                    count_diff = 0 - node.num_samples_in_compared_subtrees[tree_id1]
                current_data["count_diff"] = count_diff
            # append and recurse
            output.append(current_data)
            if node.left is not None:
                if input_cols is not None:
                    axis_name = (
                        f"{input_cols[node.axis]} <= {round(node.midpoint_at_axis, 3)}"
                    )
                else:
                    axis_name = f"ax {node.axis} <= {round(node.midpoint_at_axis, 3)}"

                KDQTreeNode.as_flattened_array(
                    node.left,
                    tree_id1,
                    tree_id2,
                    output,
                    axis_name,
                    id(node),
                    depth + 1,
                    input_cols,
                )
            if node.right is not None:
                if input_cols is not None:
                    axis_name = (
                        f"{input_cols[node.axis]} > {round(node.midpoint_at_axis, 3)}"
                    )
                else:
                    axis_name = f"ax {node.axis} > {round(node.midpoint_at_axis, 3)}"

                KDQTreeNode.as_flattened_array(
                    node.right,
                    tree_id1,
                    tree_id2,
                    output,
                    axis_name,
                    id(node),
                    depth + 1,
                    input_cols,
                )
