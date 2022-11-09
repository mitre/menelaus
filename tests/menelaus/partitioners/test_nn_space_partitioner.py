import numpy as np

from menelaus.partitioners import NNSpacePartitioner

def test_nnsp_init():
    ''' Check correct initialization state for NNSpacePartitioner'''
    part = NNSpacePartitioner(30)
    assert part.k == 30
    attrs = [part.adjacency_matrix, part.nnps_matrix, part.D, part.v1, part.v2]
    for attr in attrs:
        assert attr is None

def test_nnsp_build():
    ''' Check basic correct execution for NNSpacePartitioner.build '''
    part = NNSpacePartitioner(k=5)
    part.build(np.random.randint(0,5,(6,6)), np.random.randint(0,5,(6,6)))
    assert part.D.shape[0] <= 6*2
    assert part.D.shape[1] == 6
    assert len(part.v1) <= 6*2
    assert len(part.v2) <= 6*2
    assert part.adjacency_matrix.shape[0] <= 6*2
    assert part.nnps_matrix.shape[0] <= 6*2

def test_nnsp_compute_nnps_distance_1():
    ''' Check correct computation for NNPS distance function in 0-case '''
    nnps_mat = np.random.randint(0,5,(6,6))
    v1 = np.random.randint(0,2,6)
    distance = NNSpacePartitioner.compute_nnps_distance(nnps_mat, v1, v1)
    assert distance == 0

def test_nnsp_compute_nnps_distance_2():
    ''' Check correct computation for NNPS distance in normal case '''
    # XXX - Hard-coded example added by AS as it becomes complicated to
    #       test dynamically
    np.random.seed(123)
    nnps_mat = np.random.randint(0,5,(6,6))
    v1 = np.random.randint(0,2,6)
    v2 = 1-v1
    distance = NNSpacePartitioner.compute_nnps_distance(nnps_mat, v1, v2)
    assert distance >= 0.29 and distance <= 0.3