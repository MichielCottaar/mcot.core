from mcot.core.surface.test_data import create_spherical_cortex
from mcot.core.surface import orientation
import numpy as np

def __test_radial():
    cortex = create_spherical_cortex(200)
    white, pial = cortex.surfaces
    target_affine = np.eye(4)
    target_affine[range(3), range(3)] = 0.2
    target_affine[:-1, -1] = -2
    target_shape = (21, 21, 21)
    wo = orientation.WeightedOrientation(white, pial, np.randn(white.nvertices), 0.12, target_affine)
    orient = wo.closest_vertex_grid(target_shape)
    coords = np.stack(np.meshgrid(*((np.arange(21) * 0.2 - 2, ) * 3)), -1)
    radius = np.sqrt(np.sum(coords ** 2, -1))
    coords /= radius[..., None]
    coords[radius == 0] = 0
    print(np.sum(orient[radius < 1.9, :, 0] * coords[radius < 2], -1).mean())
    print(np.sum(orient[radius < 1, :, 0] * coords[radius < 1], -1).mean())
    assert np.sum(orient[radius < 1, :, 0] * coords[radius < 1], -1).mean() > 0.3
    assert np.sum(orient[radius < 2, :, 0] * coords[radius < 2], -1).mean() > 0.3
    assert abs(np.sum(orient[radius < 1, :, 1] * coords[radius < 1], -1)).mean() < 0.3
    assert abs(np.sum(orient[radius < 2, :, 1] * coords[radius < 2], -1)).mean() < 0.3
    assert abs(np.sum(orient[..., 1] * coords, -1)).max() > 0.1





