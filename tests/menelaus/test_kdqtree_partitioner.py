import io
import sys
import numpy as np
import scipy.stats
from menelaus.partitioners.KDQTreePartitioner import KDQTreePartitioner, KDQTreeNode


def test_build_empty():
    kp = KDQTreePartitioner(count_ubound=200, cutpoint_proportion_lbound=0.25)
    empty_array = np.array([])
    empty_array_nested = np.array([[]])
    root = kp.build(empty_array)
    assert root is None
    assert kp.node is None
    assert kp.leaves == []
    assert kp.kl_distance("foo", "bar") is None
    assert kp.fill(tree_id="foo", data=np.array([1, 1])) is None
    root = kp.build(empty_array_nested)
    assert root is None
    assert kp.node is None
    assert kp.leaves == []
    assert kp.kl_distance("foo", "bar") is None
    assert kp.fill(tree_id="foo", data=np.array([1, 1])) is None


def test_build_single():
    kp = KDQTreePartitioner(count_ubound=200, cutpoint_proportion_lbound=0.25)
    single_obs_array = np.array([[1, 2, 3]])
    root = kp.build(single_obs_array)
    assert root.num_samples_in_compared_subtrees["build"] == 1
    assert len(root.num_samples_in_compared_subtrees.keys()) == 1
    assert len(kp.leaves) == 1
    # leaf node remaining attributes are None
    assert root.axis is None
    assert root.midpoint_at_axis is None
    assert root.left is None
    assert root.right is None


def test_build_simple():
    kp = KDQTreePartitioner(count_ubound=2, cutpoint_proportion_lbound=0.1)
    simple_array = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]])
    root = kp.build(simple_array)
    assert root.num_samples_in_compared_subtrees["build"] == simple_array.shape[0]
    for leaf in kp.leaves:
        assert leaf.axis is None
        assert leaf.midpoint_at_axis is None
        assert leaf.left is None
        assert leaf.right is None
    leaf_counts = [leaf.num_samples_in_compared_subtrees["build"] for leaf in kp.leaves]
    assert leaf_counts == [2, 1, 2, 1]


def test_build_complex():
    kp = KDQTreePartitioner(count_ubound=2, cutpoint_proportion_lbound=0.2)
    imbalanced_array = np.array(
        [[1, 2, 3], [1, 2, 3], [1, 2, 3], [4, 5, 6], [4, 5, 6], [4, 5, 6], [7, 8, 9]]
    )
    _ = kp.build(imbalanced_array)
    assert kp.leaf_counts("build") == [3, 3, 1]


def test_reset():
    kp = KDQTreePartitioner(count_ubound=2, cutpoint_proportion_lbound=0.1)
    simple_array = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]])
    _ = kp.build(simple_array)
    root = kp.node
    kp.reset()
    assert root.num_samples_in_compared_subtrees["build"] == 0
    assert len(kp.leaves) == 4
    for leaf in kp.leaves:
        assert leaf.num_samples_in_compared_subtrees["build"] == 0
    _ = kp.fill(simple_array, "fill1")
    root = kp.node
    kp.reset("x", "fill1")
    assert root.num_samples_in_compared_subtrees["fill1"] == "x"
    assert len(kp.leaves) == 4
    for leaf in kp.leaves:
        assert leaf.num_samples_in_compared_subtrees["fill1"] == "x"


def test_fill_single():
    kp = KDQTreePartitioner(count_ubound=200, cutpoint_proportion_lbound=0.25)
    single_obs_array = np.array([[1, 2, 3]])
    single_obs_array_flipped = np.array([[3, 2, 1]])
    _ = kp.build(single_obs_array)
    root_filled = kp.fill(single_obs_array_flipped, "fill1")
    assert root_filled.num_samples_in_compared_subtrees["fill1"] == 1
    assert root_filled.num_samples_in_compared_subtrees["build"] == 1
    assert len(kp.leaves) == 1
    # leaf node remaining attributes are None
    assert root_filled.axis is None
    assert root_filled.midpoint_at_axis is None
    assert root_filled.left is None
    assert root_filled.right is None


def test_fill_simple():
    kp = KDQTreePartitioner(count_ubound=2, cutpoint_proportion_lbound=0.1)
    simple_array = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]])
    root = kp.build(simple_array)
    root = kp.fill(simple_array, "fill1")
    assert root.num_samples_in_compared_subtrees["build"] == simple_array.shape[0]
    assert root.num_samples_in_compared_subtrees["fill1"] == simple_array.shape[0]
    assert kp.leaf_counts("build") == [2, 1, 2, 1]
    assert kp.leaf_counts("fill1") == [2, 1, 2, 1]


def test_fill_complex():
    kp = KDQTreePartitioner(count_ubound=2, cutpoint_proportion_lbound=0.2)
    imbalanced_array = np.array(
        [[1, 2, 3], [1, 2, 3], [1, 2, 3], [4, 5, 6], [4, 5, 6], [4, 5, 6], [7, 8, 9]]
    )
    _ = kp.build(imbalanced_array)
    rng = np.random.default_rng(0)
    rng.shuffle(imbalanced_array)
    _ = kp.fill(imbalanced_array, "fill1")
    assert kp.leaf_counts("fill1") == [3, 3, 1]


def test_leaf_counts_edge_cases():
    # empty case
    kp = KDQTreePartitioner(count_ubound=2, cutpoint_proportion_lbound=0.1)
    assert kp.leaf_counts("build") is None
    # single case
    single_obs_array = np.array([[1, 2, 3]])
    _ = kp.build(single_obs_array)
    assert kp.leaf_counts("build") == [1]
    _ = kp.fill(single_obs_array, "fill1")
    assert kp.leaf_counts("fill1") == [1]


def test_kl_distance():
    # TODO - later we may want _distn function to have its own unit tests
    # identical case
    kp = KDQTreePartitioner(count_ubound=2, cutpoint_proportion_lbound=0.1)
    simple_array = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]])
    kp.build(simple_array)
    kp.fill(simple_array, "fill1")
    assert kp.kl_distance("build", "fill1") == 0
    # empty case
    # TODO - when 1+ of build/fill tree leaves = None, that needs handling in the code
    # multi case
    fill2_array = np.array([[1, 9], [2, 8], [3, 7], [4, 6], [5, 5]])
    kp.fill(fill2_array, "fill2")
    # computation logic - how do we make it so we're not repeating core logic?
    expected_leaf_counts_build = [2, 1, 2, 1]
    expected_hist_build = np.array(expected_leaf_counts_build) + 0.5
    expected_hist_build = expected_hist_build / (
        np.sum(expected_leaf_counts_build) + len(expected_hist_build) / 2
    )
    expected_leaf_counts_fill2 = [0, 3, 2, 0]
    expected_hist_fill2 = np.array(expected_leaf_counts_fill2) + 0.5
    expected_hist_fill2 = expected_hist_fill2 / (
        np.sum(expected_leaf_counts_fill2) + len(expected_hist_fill2) / 2
    )
    expected_kl_distance = scipy.stats.entropy(expected_hist_build, expected_hist_fill2)
    assert kp.kl_distance("build", "fill2") == expected_kl_distance
    assert kp.kl_distance("fill1", "fill2") == expected_kl_distance


def test_to_plotly_dataframe():
    kp = KDQTreePartitioner(count_ubound=2, cutpoint_proportion_lbound=0.2)
    imbalanced_array = np.array(
        [[1, 2, 3], [1, 2, 3], [1, 2, 3], [4, 5, 6], [4, 5, 6], [4, 5, 6], [7, 8, 9]]
    )
    _ = kp.build(imbalanced_array)
    rng = np.random.default_rng(0)
    rng.shuffle(imbalanced_array)
    _ = kp.fill(imbalanced_array, "fill1")
    df_plot = kp.to_plotly_dataframe()
    assert df_plot.shape[0] > 0
    assert set(df_plot.columns) == set(
        ["name", "idx", "parent_idx", "cell_count", "depth"]
    )
    df_plot = kp.to_plotly_dataframe(tree_id2="fill1", max_depth=1)
    assert df_plot.shape[0] == 3
    assert set(df_plot.columns) == set(
        ["name", "idx", "parent_idx", "cell_count", "depth", "count_diff", "kss"]
    )
    assert df_plot.loc[1:, "name"].apply(lambda x: True if ("ax" in x) else False).all()
    df_plot = kp.to_plotly_dataframe(tree_id2="fill1", max_depth=1, input_cols=["col1", "col2", "col3"])
    assert df_plot.shape[0] == 3
    assert set(df_plot.columns) == set(
        ["name", "idx", "parent_idx", "cell_count", "depth", "count_diff", "kss"]
    )
    assert df_plot.loc[1:, "name"].apply(lambda x: True if ("col" in x) else False).all()
    assert df_plot.loc[1:, "name"].apply(lambda x: True if ("ax" not in x) else False).all()


def test_as_text():
    # very basic check for consistent behavior
    kp = KDQTreePartitioner(count_ubound=2, cutpoint_proportion_lbound=0.1)

    # capture the print output
    output = io.StringIO()
    sys.stdout = output

    data = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]])
    root = kp.build(data)
    KDQTreeNode.as_text(root)
    output.getvalue()
    assert output.getvalue().count("subtree") == 7

    output = io.StringIO()
    sys.stdout = output
    data = np.array([])
    root = kp.build(data)
    KDQTreeNode.as_text(root)
    assert output.getvalue() == ""


def test_missing_tree_flattened_array():
    # given a tree_id which is not present in the tree, make sure its counts are
    # being treated as 0 everywhere
    kp = KDQTreePartitioner(count_ubound=2, cutpoint_proportion_lbound=0.1)
    data = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]])
    root = kp.build(data)
    out = []
    KDQTreeNode.as_flattened_array(
        root, tree_id1="build", tree_id2="garbage", output=out
    )
    assert all([guy["count_diff"] <= 0 for guy in out])
