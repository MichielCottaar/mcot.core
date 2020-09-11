import numpy as np
from mcot.core.spherical import euler2mat
from mcot.core.surface import Mesh2D, track
from numpy.testing import assert_allclose


def get_rectangle(phi, theta, psi, split):
    vertices = np.array([
        (0, 0),
        (0, 1),
        (0.5, 0),
        (1, 1),
        (1, 0),
        (split, 1),
    ])
    faces = np.array([
        (0, 1, 2),
        (1, 2, 5),
        (3, 2, 5),
        (2, 3, 4),
    ])
    orientations = np.array([
        (0, 1),
        (1, 0),
        (1, 0),
        (0, -1),
    ])
    mat = euler2mat(phi, theta, psi)

    return Mesh2D(
            vertices=mat[:, :2].dot(vertices.T),
            faces=faces.T,
    ), mat[:, :2].dot(orientations.T).T


def test_gradient():
    np.random.seed(1234)
    for pos in np.random.randn(5, 3, 5):
        surf = Mesh2D(
                vertices=pos,
                faces=np.array([[0, 0], [1, 3], [2, 4]])
        )
        assert surf.ndim == 3
        assert surf.nvertices == 5
        assert surf.nfaces == 2

        for grad in np.random.randn(4, 3):
            on_vertices = (pos * grad[:, None]).sum(0)
            grad_surf = surf.gradient(on_vertices)
            grad_proj = grad[:, None] - (grad[:, None] * surf.normal()).sum(0)[None, :] * surf.normal()
            assert_allclose(grad_surf, grad_proj)


def test_simple():
    np.random.seed(12345)
    for phi, theta, psi, split in np.random.rand(3, 4):
        print('split', split)
        surf, orientation = get_rectangle(phi, theta, psi, split)

        assert surf.nvertices == 6
        assert surf.nfaces == 4

        predicted_orientations = np.array([
            (1, 0),
            (0, 1),
            (0, -1),
            (-1 / np.sqrt(2), 1 / np.sqrt(2))
        ])
        flat_orient = track.flatten_gradient(surf, orientation)
        assert_allclose(flat_orient, predicted_orientations, atol=1e-12)

        for pos in np.random.rand(4):
            new_triangle, new_pos1, new_pos2 = track.surface_step(0, 0, pos, surf.faces.T, flat_orient)
            assert new_triangle == 1
            assert_allclose(new_pos1, pos)
            assert_allclose(new_pos2, 0, atol=1e-8)

            new_triangle, new_pos1, new_pos2 = track.surface_step(new_triangle, new_pos1, new_pos2,
                                                                  surf.faces.T, flat_orient)
            assert new_triangle == 2

            new_triangle, new_pos1, new_pos2 = track.surface_step(new_triangle, new_pos1, new_pos2,
                                                                  surf.faces.T, flat_orient)
            assert new_triangle == 3
            assert_allclose(new_pos1, 1 - pos)
            assert_allclose(new_pos2, 0, atol=1e-8)

            new_triangle, new_pos1, new_pos2 = track.surface_step(new_triangle, new_pos1, new_pos2,
                                                                  surf.faces.T, flat_orient)
            assert new_triangle == 3
            assert_allclose(new_pos1, 0, atol=1e-8)
            assert_allclose(new_pos2, 1 - pos)

            new_triangle, new_pos1, new_pos2 = track.surface_step(new_triangle, new_pos1, new_pos2,
                                                                  surf.faces.T, flat_orient)
            assert new_triangle == 3
            assert_allclose(new_pos1, 0, atol=1e-8)
            assert_allclose(new_pos2, 1 - pos)

            new_triangle, new_pos1, new_pos2 = track.track_to_maximum(0, 0, pos, surf.faces.T, flat_orient)
            assert new_triangle == 3
            assert_allclose(new_pos1, 0, atol=1e-8)
            assert_allclose(new_pos2, 1 - pos)

            print('testing', pos)
            new_triangle, new_pos1, new_pos2 = track.track_to_maximum(0, pos, 0, surf.faces.T, flat_orient)
            assert new_triangle == 3
            assert_allclose(new_pos1, 0, atol=1e-8)
            assert_allclose(new_pos2, 1)

            to_extract = np.arange(surf.nvertices)
            extracted = track.extract_ridge_values(surf, orientation, to_extract)
            assert not np.isfinite(extracted[[2, 4]]).any()
            assert_allclose(extracted[[0, 1, 3, 5]], to_extract[4])
