"""
Partitioner algorithms are typically an early data-processing component for drift
detection methods. Data may be partitioned into some new structure (e.g., trees,
histograms, or other distributional representations). Separate structures (representing
separate batches of data) may then be more efficiently compared by drift detection
algorithms in order to raise alarms.
"""

from menelaus.partitioners.KDQTreePartitioner import KDQTreePartitioner, KDQTreeNode
from menelaus.partitioners.NNSpacePartitioner import NNSpacePartitioner
