import numpy as np
import scipy.stats

# from ete3 import Tree
# TODO: method docstrings


class KDQTreePartitioner:
    """This class encapsulates a KDQ Tree for partitioning data which runs
    through drift detection algorithms, as described by Dasu et. al.
    (2006).

    The general algorithm to build a tree does until a stop condition:
    - identify axis to split upon
    - identify midpoint at axis
    - split into lower/upper halves using midpoint
    - recursively split sub-trees

    This class can be used to build a tree, 'fill' a tree (send new
    data to be forcibly partitioned according to an existing build),
    access leaf nodes, and perform visualization / printing. There
    are also convenience functions to calculate distances between
    a build and fill tree.

    Attributes:
    count_ubound (int): upper bound of data points to be
        partitioned into any one cell
    cutpoint_proportion_lbound (float): minimum proportion of all
        features' range values that any constructed cell size must satisfy
    """

    def __init__(self, count_ubound=200, cutpoint_proportion_lbound=0.25):
        self.count_ubound = count_ubound
        self.cutpoint_proportion_lbound = cutpoint_proportion_lbound
        self.node = None
        self.leaves = []

    def build(self, data):
        if len(data.shape) <= 1:
            return None
        _, m = data.shape
        min_cutpoint_sizes = [
            int(self.cutpoint_proportion_lbound * np.ptp(data[:, axis]))
            for axis in range(m)
        ]
        self.node = KDQTreeNode.build(
            data, self.count_ubound, min_cutpoint_sizes, self.leaves
        )
        return self.node

    def reset(self, value=0, tree_id="build"):
        KDQTreeNode.reset(self.node, value=value, tree_id=tree_id)

    def fill(self, data, tree_id, reset=False):
        if self.node is None or len(data.shape) <= 1:
            return None
        KDQTreeNode.fill(data, self.node, self.count_ubound, tree_id, reset)
        return self.node

    def leaf_counts(self, tree_id):
        ret = None
        if self.leaves:
            ret = [
                leaf.num_samples_in_compared_subtrees[tree_id] for leaf in self.leaves
            ]
        return ret

    def kl_distance(self, tree_id1, tree_id2):
        assert self.leaves is not None
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
            counts (array): an array of counts

        Returns:
            hist: an array of frequencies
        """
        total = np.sum(counts)
        hist = np.array(counts) + 0.5
        hist = hist / (total + len(hist) / 2)
        return hist

    # def to_newick_tree(self, tree_id='build', depth=0):
    #     newick_tree = KDQTreeNode.as_newick_tree(self.node)
    #     newick_tree = f"{newick_tree};" # extra () around tree may or may not be correct
    #     # newick_tree = Tree(newick_tree, format=8)
    #     return newick_tree


class KDQTreeNode:
    """This class represents a node in the KDQ Tree data structure described above.
    Its static methods provide the engine for recursively building, filling,
    and output-ing trees.

    Build / fill trees are identified by a fill tree. During initialization,
    a node typically takes num_samples_in_compared_subtrees (among other values
    needed for building), which is a dict.

    Each node stores as a value this dict the number of data points contained
    within itself and its subtrees. A 'tree_id' key identifies which build
    those counts are associated with (e.g., the first call to build() stores
    counts in the node.num_samples_in_compared_subtrees['build'] by default).
    Subsequent calls to fill() with new data store additional counts for the node,
    according to the ID the user passes in.

    Attributes:
    num_samples_in_compared subtrees (dict): number of counts for some ID
        tree_id, contained within node and its children
    axis (int): axis at which to build / fill (for recursive construction)
    midpoint_at_axis (float): midpoint at provided axis
    left (KDQTreeNode): left child
    right (KDQTreeNode): right child
    """

    def __init__(
        self, num_samples_in_compared_subtrees, axis, midpoint_at_axis, left, right
    ):
        self.num_samples_in_compared_subtrees = num_samples_in_compared_subtrees
        self.axis = axis
        self.midpoint_at_axis = midpoint_at_axis
        self.left = left
        self.right = right

    @staticmethod
    def build(data, count_ubound, min_cutpoint_sizes, leaves, depth=0):
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
        # case: no more nodes
        if node is None:
            return
        n = data.shape[0]
        axis = node.axis
        # basically, matches the return Node(n, Nonex4) case above (see next TODO)
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
        if node:
            node.num_samples_in_compared_subtrees[tree_id] = value
            KDQTreeNode.reset(node.left, value, tree_id)
            KDQTreeNode.reset(node.right, value, tree_id)

    @staticmethod
    def as_text(node, tree_id="build"):
        # TODO - to avoid a recursive printing problem, this prints rather than storing an ongoing output string
        if not node:
            return
        print(f"\nsubtree count: {node.num_samples_in_compared_subtrees[tree_id]}")
        if node.left:
            print(f"\tleft: {node.left.num_samples_in_compared_subtrees[tree_id]}")
        if node.right:
            print(f"\tright: {node.right.num_samples_in_compared_subtrees[tree_id]}")
        KDQTreeNode.as_text(node.left, tree_id)
        KDQTreeNode.as_text(node.right, tree_id)

    @staticmethod
    def as_newick_tree(node, tree_id="build"):
        # TODO - specify depth
        # TODO - newick (C,D) may not mean C is left and D is right - address?
        output = node.num_samples_in_compared_subtrees[tree_id]
        if node.left and not node.right:
            output = f"(,{KDQTreeNode.as_newick_tree(node.left)}){output}"
        elif not node.left and node.right:
            output = f"({KDQTreeNode.as_newick_tree(node.right)},){output}"
        elif node.left and node.right:
            output = f"({KDQTreeNode.as_newick_tree(node.right)},{KDQTreeNode.as_newick_tree(node.left)}){output}"
        elif not node.left and not node.right:
            output = f"{output}"  # format 8? TODO - root may still be an issue
        else:
            pass
        return output
