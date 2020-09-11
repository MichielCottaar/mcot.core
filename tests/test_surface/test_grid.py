from mcot.core.surface.test_data import rectangle
from mcot.core.surface.grid import closest_surface, intersect, intersect_resolution
import numpy as np


def generate_closest(mesh, arrc, slices, notslices=None):
    print(arrc)
    if notslices is None:
        notslices = (slice(0, 0), slice(0, 0))
    eslices = list(zip(np.arange(2) * 4 + 0.5, slices, notslices))
    for x, xgood, xbad in eslices:
        for y, ygood, ybad in eslices:
            for z, zgood, zbad in eslices:
                goodarr = arrc[xgood][:, ygood][:, :, zgood]
                badarr = arrc[xbad][:, ybad][:, :, zbad]
                getp = np.all(mesh.vertices == np.array([x, y, z])[:, None], 0)
                assert np.sum(getp) == 1
                ixp = np.where(getp)[0][0]
                print(ixp, goodarr, badarr)
                assert (goodarr == ixp).all()
                assert (badarr != ixp).all()


def test_closest_surface():
    mesh = rectangle(np.ones(3) * 4, inward_normals=True)
    mesh.vertices += 0.5
    closest = closest_surface(mesh, np.ones((6, 6, 6)))
    generate_closest(mesh, closest, (slice(None, 3), slice(3, None)))
    closest_dist = closest_surface(mesh, np.ones((6, 6, 6)), max_dist=2.1)
    pos = np.array(np.meshgrid(*tuple([np.arange(6)] * 3)))
    still_good = np.sqrt(np.amin(np.sum((pos[:, None, ...] - mesh.vertices[:, :, None, None, None]) ** 2, 0), 0)) < 2.1
    print(closest_dist)
    assert (closest_dist[still_good] == closest[still_good]).all()
    assert (closest_dist[~still_good] == -1).all()
    inner = (slice(1, 3), slice(3, -1))
    outer = (slice(None, 1), slice(-1, None))
    closest = closest_surface(mesh, np.ones((6, 6, 6)), pos_inpr=True)
    generate_closest(mesh, closest, inner, outer)
    closest = closest_surface(mesh, np.ones((6, 6, 6)), pos_inpr=False)
    generate_closest(mesh, closest, outer, inner)
    flip = np.eye(4)
    flip[0, 0] = -1
    flip[0, -1] = 5
    mesh_flipped = mesh.apply_affine(flip)
    print(np.unique(mesh_flipped.vertices[0, :]), np.unique(mesh.vertices[0, :]))
    assert (np.unique(mesh_flipped.vertices[0, :]) == np.unique(mesh.vertices[0, :])).all()
    print(mesh_flipped.vertices)
    closest = closest_surface(mesh_flipped, np.ones((6, 6, 6)), pos_inpr=True)
    generate_closest(mesh_flipped, closest, inner, outer)
    closest = closest_surface(mesh_flipped, np.ones((6, 6, 6)), pos_inpr=False)
    generate_closest(mesh_flipped, closest, outer, inner)


def test_intersect():
    mesh = rectangle(np.ones(3) * 4, inward_normals=True)
    mesh.vertices += 1
    inter = intersect(mesh, (7, 7, 7))
    on_grid = np.zeros((7, 7, 7), dtype='bool')
    for ixdim in range(3):
        slices = [slice(1, -1)] * 3
        slices[ixdim] = 1
        on_grid[tuple(slices)] = True
        slices[ixdim] = -2
        on_grid[tuple(slices)] = True
    assert (inter.has_hit[on_grid] != -1).all()
    assert (inter.has_hit[~on_grid] == -1).all()

    assert (inter.nhit()[~on_grid] == 0).all()
    assert (inter.nhit()[on_grid] > 0).all()
    for x in (1, -2):
        for y in (1, -2):
            for z in (1, -2):
                assert inter.nhit()[x, y, z] >= 3
    for dim in range(3):
        for ixdim, ixother in [(1, 2), (-2, -3)]:
            select = [ixother] * 3
            select[dim] = ixdim
            assert inter.nhit()[tuple(select)] in {1, 2}

    fit_res = intersect_resolution(mesh, 0.1)
    nofit_res = intersect_resolution(mesh, 0.131)
    for ixtria in range(mesh.nfaces):
        assert ixtria in fit_res.vertex_hit
        assert ixtria in nofit_res.vertex_hit

def check_target(wo, pos, correct_vertices, norient=300):
    orientation = np.random.randn(norient, 3)
    orientation /= np.sqrt(np.sum(orientation ** 2, 1))[:, None]
    index_hit, pos_hit = wo.ray_intersect(pos - orientation, orientation)
    for idx_orient in range(orientation.shape[1]):
        assert abs(pos_hit[idx_orient] - pos).max() < 1e-3
        assert index_hit[idx_orient] in correct_vertices


def test_ray_intersect():
    np.random.seed(12345)
    cube = rectangle((10, 10, 10))
    wo = intersect_resolution(cube, 0.1)
    for ixtria in range(cube.nfaces):
        pos = np.mean(cube.vertices[:, cube.faces[:, ixtria]], -1)
        check_target(wo, pos, [ixtria])
    for ixpoint in range(cube.nvertices):
        check_target(wo, cube.vertices[:, ixpoint], np.where((cube.faces == ixpoint).any(0))[0])

    norient = 300
    for start in ([1, 1, 1], [5, 5, 5], [9, 9, 9], [np.pi, np.sqrt(2), 0.981]):
        for orientation in [np.random.randn(norient, 3), np.eye(3), -np.eye(3)]:
            orientation /= np.sqrt(np.sum(orientation ** 2, 1))[:, None]
            index, pos = wo.ray_intersect(np.array(start, dtype='f4'), orientation)
            offset = (pos - start) / orientation
            assert (index != -1).all()
            assert (((pos == 0) | (pos == 10)).sum(-1) == 1).all()
            assert (offset[orientation != 0] > 0).all()
            offset[orientation == 0] = np.nan
            diff = offset - np.nanmean(offset, -1)[:, None]
            print(diff[orientation != 0].max())
            assert abs(diff[orientation != 0]).max() < 1e-3


