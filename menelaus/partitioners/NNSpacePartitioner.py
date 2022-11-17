import numpy as np
from sklearn.neighbors import NearestNeighbors


class NNSpacePartitioner:
    """
    This class encodes the Nearest Neighbors Space Partitioning
    scheme (NNPS) for use in the Nearest Neigbors Density Variation
    Identification (NN-DVI) drift detection algorithm, both of which
    are introduced in Liu et al. (2018).

    Broadly, NNSP combines two input data samples, finds the adjacency
    matrix using Nearest Neighbor search, and transforms the data points
    into a set of shared subspaces for estimating density and changes in
    density between the two samples, via a distance function.

    Attributes:
        k (int): the 'k' in k-Nearest-Neighbor (k-NN) search
        D (numpy.array): combined data from two samples
        v1 (numpy.array): indices of first sample data within ``D``
        v2 (numpy.array): indices of second sample data within ``D``
        nnps_matrix (numpy.array): NNSP shared-subspace representation of ``D``
            and its adjacency matrix
        adjacency_matrix (numpy.array): result of k-NN search on ``D``
    """

    def __init__(self, k: int):
        """
        Args:
            k (int): the 'k' in k-NN search, describing size of searched neighborhood
        """
        self.k = k
        self.D = None
        self.v1 = None
        self.v2 = None
        self.nnps_matrix = None
        self.adjacency_matrix = None

    def build(self, sample1: np.array, sample2: np.array):
        """
        Builds an NNSP representation matrix given two samples of data.
        Internally stores computed union set, adjacency matrix, index
        arrays for the two samples, and NNSP representation matrix.

        Args:
            sample1 (numpy.array): first (possibly the 'reference') sample set
            sample2 (numpy.array): second (possibly the 'test') sample set
        """
        data = np.vstack((sample1, sample2))
        D, inverted_indices = np.unique(data, axis=0, return_inverse=True)
        self.D = D
        v1, v2 = np.array_split(inverted_indices, 2)
        v1_onehot = np.zeros(D.shape[0])
        v2_onehot = np.zeros(D.shape[0])
        # XXX - Alternatively, v1_onehot = np.identity(adjacency_matrix.shape[0])[v1] - Anmol
        v1_onehot[v1] = 1.0
        v2_onehot[v2] = 1.0
        self.v1 = v1_onehot
        self.v2 = v2_onehot
        nn = NearestNeighbors(n_neighbors=self.k).fit(D)
        # TODO: maybe we can gain performance by performing operations using the returned
        # scipy.sparse array, as opposed to converting this way.
        M_adj = nn.kneighbors_graph(D).toarray()
        self.adjacency_matrix = M_adj
        # XXX - NearestNeighbors already adds the self-neighbors
        # TODO - check about order preservation
        P_nnps = M_adj
        weight_array = np.sum(P_nnps, axis=1).astype(int)
        Q = np.lcm.reduce(weight_array)
        m = Q / weight_array
        m = m * np.identity(len(m))
        self.nnps_matrix = np.matmul(m, P_nnps)

    @staticmethod
    def compute_nnps_distance(nnps_matrix, v1, v2):
        """
        Breaks NNSP reprsentation matrix into NNPS matrices
        for two samples using indices, computes difference in
        densities of shared subspaces, between samples.

        Args:
            nnps_matrix (numpy.array): NNSP representation matrix
            v1 (numpy.array): indices of first sample in ``D``, ``nnps_matrix``,
                in one-hot encoding format
            v2 (numpy.array): indices of second sample in ``D``, ``nnps_matrix``,
                in one-hot encoding format

        Returns:
            float: distance value between samples, representing difference
                in shared subspace densities
        """
        M_s1 = np.dot(v1, nnps_matrix)
        M_s2 = np.dot(v2, nnps_matrix)

        # These commented lines would only be relevant if there were overlap
        # between the two vectors, which there never should be for our use case.
        # Otherwise, this is always going to be the number of elements.
        # membership = np.sum(np.array([v1, v2]), axis=0)
        # membership = membership >= 1  # in case of overlap
        # denom = sum(membership)
        denom = len(v1)

        d_nnps = np.sum(np.abs(M_s1 - M_s2) / (M_s1 + M_s2))
        d_nnps /= denom
        return d_nnps
