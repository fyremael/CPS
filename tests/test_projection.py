import numpy as np

from cps.projection import arnoldi_projection, project_dense_operator


def test_dense_projection():
    a = np.diag([1.0, 2.0, 3.0])
    q = np.eye(3)[:, :2]
    assert np.allclose(project_dense_operator(a, q), np.diag([1.0, 2.0]))


def test_arnoldi_projection_shapes():
    a = np.array([[0.8, 1.0, 0.0], [0.0, 0.7, 1.0], [0.0, 0.0, 0.6]])
    q, h = arnoldi_projection(lambda x: a @ x, dimension=3, rank=3, initial=np.ones(3))
    assert q.shape[0] == 3
    assert h.shape[0] == h.shape[1] == q.shape[1]
    assert np.allclose(q.conj().T @ q, np.eye(q.shape[1]), atol=1e-8)
