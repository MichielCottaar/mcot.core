from mcot.core._scripts.surface import gradient
import numpy as np
from scipy import sparse, spatial
from numpy import testing
from mcot.core.surface.test_data import triangle_mesh, mesh_to_cortex


def test_histogram_intersection():
    first = np.array([1., 0])
    second = np.array([0., 1])
    balanced = np.array([1., 1])

    for arr in (first, second, balanced):
        assert gradient._histogram_intersection(arr, arr) == 1.

    assert gradient._histogram_intersection(first, second) == 0.
    assert gradient._histogram_intersection(first, balanced) == 0.5
    assert gradient._histogram_intersection(second, balanced) == 0.5


def test_distance():
    dense_graph = np.zeros((5, 5), dtype='bool')
    for idx1, idx2 in [
        (0, 1),
        (0, 3),
        (1, 2),
    ]:
        dense_graph[idx1, idx2] = True
        dense_graph[idx2, idx1] = True

    sparse_graph = sparse.coo_matrix(dense_graph)

    features = np.random.randn(5, 6)

    for metric in gradient.scipy_options:
        res = np.array(gradient.distance(sparse_graph, features, metric=metric).todense())
        full_res = spatial.distance.squareform(
                spatial.distance.pdist(features, metric=metric)
        )

        testing.assert_allclose(res[dense_graph], full_res[dense_graph])
        testing.assert_equal(res[~dense_graph], 0.)


def test_reduce():
    dense_graph = np.zeros((5, 5), dtype='bool')
    for idx1, idx2 in [
        (0, 1),
        (0, 3),
        (1, 2),
        (0, 2)
    ]:
        dense_graph[idx1, idx2] = True
        dense_graph[idx2, idx1] = True

    distance = dense_graph * np.random.rand(5, 5)
    distance += distance.T

    sparse_distance = sparse.coo_matrix(distance)

    nan_distance = distance.copy()
    nan_distance[distance == 0.] = np.nan

    assert np.allclose(
            gradient.reduce(sparse_distance, 'max'),
            np.nanmax(nan_distance, 0), equal_nan=True
    )

    assert np.allclose(
            gradient.reduce(sparse_distance, 'mean'),
            np.nanmean(nan_distance, 0), equal_nan=True
    )

    assert np.allclose(
            gradient.reduce(sparse_distance, 'median'),
            np.nanmedian(nan_distance, 0), equal_nan=True
    )

    assert np.allclose(
            gradient.reduce(sparse_distance, 'min'),
            np.nanmin(nan_distance, 0), equal_nan=True
    )


def test_main():
    surface = mesh_to_cortex(triangle_mesh())
    for arr in (
            np.ones((3, 5)) * 5,
            np.ones((3, 5)) * np.random.randn(5)[None, :],
    ):
        for reduction in ('min', 'max', 'mean', 'median'):
            res = gradient.run(surface, arr, distance_metric='euclidean', reduction=reduction)
            assert res.shape == (3, )
            assert (res == 0.).all()
            res = gradient.run(surface, arr, distance_metric='histogram_intersection', reduction=reduction)
            assert res.shape == (3, )
            testing.assert_allclose(res, 1.)
    arr = np.arange(3)[:, None]
    res = gradient.run(surface, arr, distance_metric='euclidean', reduction='min')
    assert (res == 1).all()
    res = gradient.run(surface, arr, distance_metric='euclidean', reduction='max')
    assert (res == [2, 1, 2]).all()
    res = gradient.run(surface, arr, distance_metric='euclidean', reduction='mean')
    assert (res == [1.5, 1, 1.5]).all()
    res = gradient.run(surface, arr, distance_metric='euclidean', reduction='median')
    assert (res == [1.5, 1, 1.5]).all()

    arr = np.eye(3)
    for reduction in ('min', 'max', 'mean', 'median'):
        res = gradient.run(surface, arr, distance_metric='histogram_intersection', reduction=reduction)
        assert res.shape == (3, )
        assert (res == 0).all()


