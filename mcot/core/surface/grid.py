"""Defines many functions that take both a grid and a mesh as input
"""
import numba
import numpy as np
from .mesh import Mesh2D
import tempfile
import subprocess
import nibabel
import os
from numpy import linalg
from . import utils


class GridSurfaceIntersection(object):
    """
    Represents the intersections between a surface and a grid.
    """
    def __init__(self, surface, affine, has_hit, vertex_hit):
        """
        Computes the intersection of the surface with the grid defined by `shape` and `affine`.

        This object will usually be created through grid.intersect or grid.intersect_resolution

        Arguments:
        :param surface: 2D surface in a 3D space that intersects with the grid.
        :type surface: Mesh2D
        :param affine: (4, 4) array describing the transformation from voxel space to the mm space that the surface is in
        :type affine: np.ndarray
        :param has_hit: (nx, ny, nz) array which is -1 for all voxels not hit and indexes `vertex_hit` for all other voxels
        :type has_hit: np.ndarray
        :param vertex_hit: (Nhit, 30) array which in the first M entries lists the M faces hitting the voxel (filled with -1 afterwards)
        :type vertex_hit: np.ndarray
        """
        self.surface = surface
        self.affine = affine
        self.has_hit = has_hit
        self.vertex_hit = vertex_hit
        self.surf_vox = self.surface.apply_affine(self.affine)

    def nhit(self, ):
        """
        Computes how much vertices intersect with each voxel

        :return: (nx, ny, nz) int array with the number of vertices intersecting any voxel
        """
        res = np.zeros(self.has_hit.shape, dtype='i4')
        use = self.has_hit != -1
        res[use] = np.sum(self.vertex_hit[self.has_hit[use], :] != -1, -1)
        return res

    def ray_intersect(self, start, orientation, pos_inpr=0, max_dist=9e9):
        """
        Computes the intersection of the ray starting from `start` in direction `hemisphere`.

        :param start: (N, 3) initial positions (in mm coordinates)
        :param orientation: (N, 3) velocities (in mm coordinates)
        :param pos_inpr: negative to ensure the hemisphere is misaligned with the normal, positive to ensure alignment (advisable to set, especially for the pial surface).
        :return: (index of the intersected triangle, position of the intersection)
        """
        start_vox = utils.affine_mult(linalg.inv(self.affine), start)
        orientation_vox = np.dot(orientation, self.affine[:3, :3])
        pos, orient = np.broadcast_arrays(np.atleast_2d(start_vox).astype('f4'),
                                          np.atleast_2d(orientation_vox).astype('f4'))
        orient = orient / np.sqrt(np.sum(orient ** 2, -1))[..., None]
        index_hit = np.zeros(pos.shape[:-1], dtype='i4')
        pos_hit = np.zeros(pos.shape, dtype='f4')
        _numba_ray_intersect(index_hit, pos_hit, pos, orient, self.surf_vox.vertices[:, self.surf_vox.faces].T,
                             self.has_hit, self.vertex_hit, self.surf_vox.normal().T,
                             pos_inpr=pos_inpr, max_dist=max_dist)
        return index_hit, utils.affine_mult(self.affine, pos_hit)

    def edge_intersect(self, ):
        """
        Finds the edges that intersect with the surface.

        - draw a line segment for 1 voxel length from (nx - 0.5, ny - 0.5, nz -0.5) in the direction `dim` and in the positive direction (orient=0) or the negative direction (orient=1)
        - if this line segment intersects with any surface and moves from inside the surface to outside, set edges[nx, ny, nz, dim, orient] to True; otherwise it is False

        Note that edge (i, j, k, ...) correponds to the lower-left corner of the voxel (i, j, k)

        see notebooks/gyral_structure/Development/split_grid.ipynb for a visual test

        :return: (nx + 1, ny + 1, nz + 1, 3, 2) boolean array which is True:
        """
        res = np.zeros(tuple(sz + 1 for sz in self.has_hit.shape) + (3, 2), dtype='bool')
        _mark_edges(res, self.has_hit, self.vertex_hit, self.surf_vox.vertices[:, self.surf_vox.faces].T,
                    self.surf_vox.normal().T, np.zeros(3), np.zeros(3))
        return res


@numba.jit(nopython=True)
def _mark_edges(edges, has_hit, vertex_hit, faces, normal, tmp_pos, tmp_orient):
    """
    Computes which edges intersect with the provided surface

    Helper function for GridSurfaceIntersection.edge_intersect
    """
    for x in range(has_hit.shape[0]):
        for y in range(has_hit.shape[1]):
            for z in range(has_hit.shape[2]):
                idx_hit = has_hit[x, y, z]
                if idx_hit != -1:
                    for idx_which_hit in range(vertex_hit.shape[1]):
                        idx_vert = vertex_hit[idx_hit, idx_which_hit]
                        if idx_vert == -1:
                            break
                        for orientation in range(3):
                            tmp_pos[0] = x - 0.5
                            tmp_pos[1] = y - 0.5
                            tmp_pos[2] = z - 0.5
                            tmp_orient[:] = 0.
                            tmp_orient[orientation] = 1.
                            if _test_intersection(tmp_orient, tmp_pos, tmp_orient, faces, normal, idx_vert, pos_inpr=1) == 0:
                                if abs(tmp_orient[orientation] - tmp_pos[orientation]) < 1:
                                    edges[x, y, z, orientation, 0] = True
                            tmp_orient[:] = 0.
                            tmp_orient[orientation] = -1.
                            tmp_pos[orientation] += 1
                            if _test_intersection(tmp_orient, tmp_pos, tmp_orient, faces, normal, idx_vert, pos_inpr=1) == 0:
                                if abs(tmp_orient[orientation] - tmp_pos[orientation]) < 1:
                                    i, j, k = x, y, z
                                    if orientation == 0:
                                        i += 1
                                    if orientation == 1:
                                        j += 1
                                    if orientation == 2:
                                        k += 1
                                    edges[i, j, k, orientation, 1] = True


@numba.jit(nopython=True)
def _numba_ray_intersect(index_hit, pos_hit, start_all, orientation_all, faces, voxel_hit, vertices,
                         normals, pos_inpr=0, stop_wrong_inpr=False, max_dist=9e9):
    """
    Identifies the intersection of a ray from `start` in direction given by `hemisphere`.

    Helper function for GridSurfaceIntersection.ray_intersect

    :param index_hit: (K, ) output array, which will contain the vertex indices of the intersection (-1 for no intersection)
    :param pos_hit: (K, 3) output array, which will contain the 3D location of the intersection (-m primor.gyral_orientation L_white.surf.gii L_pial.surf.gii segmentation.nii.gz index_line.nii.gz --distance_filename dist_line.nii.gznp.nan for no intersection)
    :param start_all: (K, 3) array of the initial point
    :param orientation_all: (K, 3) array of the ray hemisphere
    :param faces: (N faces, 3 vertices per triangle, 3 dimensions) array of the surface faces
    :param voxel_hit: (Nx, Ny, Nz) integer array which contains -1 for voxels with no intersection and otherwise the index of the vertices stored in `vertices`
    :param vertices: (M, 30) integer array of the faces crossing through each voxel which has an intersection (filled with -1)
    :param normals: (N, 3) array of face normals (pointing inwards)
    :param pos_inpr: negative to ensure the hemisphere is misaligned with the normal, positive to ensure alignment.
    :param stop_wrong_inpr: stop if the innner product test (above) fails (can have surprising effects if vertices with opposite normals are in the same voxel)
    :param max_dist: float; maximum distance to track
    :return: (-1, -1.) if no vertex is hit within `max_dist` otherwise a tuple of (index of vertex hit, distance from start to vertex)
    """
    dist_hit = np.empty(3)
    position = np.empty(3)
    orientation = np.empty(3)
    start = np.empty(3)
    res_pos = np.empty(3)
    shape = voxel_hit.shape
    for ixray in range(index_hit.size):
        index_hit[ixray] = -1
        for ixdim in range(3):
            pos_hit[ixray, ixdim] = np.nan
            start[ixdim] = start_all[ixray, ixdim]
            orientation[ixdim] = orientation_all[ixray, ixdim]
            if orientation_all[ixray, ixdim] == 0.:
                dist_hit[ixdim] = np.inf
            else:
                dist_hit[ixdim] = abs((1 - ((start[ixdim] - 0.5) % 1)) / orientation[ixdim])
            position[ixdim] = start[ixdim]
        distance = 0.
        count = 0
        while distance < max_dist:
            count += 1
            if count > 1000:
                1/0
            x = int(round(position[0] + 1e-5 * orientation[0]))
            y = int(round(position[1] + 1e-5 * orientation[1]))
            z = int(round(position[2] + 1e-5 * orientation[2]))
            index = -1 # no intersections outside of the bounding box
            if x >= shape[0] or x <= -1 or y >= shape[1] or y <= -1 or z >= shape[2] or z <= -1:
                if ( (x >= shape[0] and orientation[0] >= 0) or (x <= -1 and orientation[0] <= 0) or
                     (y >= shape[1] and orientation[1] >= 0) or (y <= -1 and orientation[1] <= 0) or
                     (z >= shape[2] and orientation[2] >= 0) or (z <= -1 and orientation[2] <= 0) ):
                    break  # left the surface bounding box for good
            else:
                index = voxel_hit[x, y, z]
            if index != -1:
                for ixvertex in range(vertices.shape[1]):
                    ixface = vertices[index, ixvertex]
                    if ixface == -1:
                        break
                    hit = _test_intersection(res_pos, start, orientation, faces, normals, ixface, pos_inpr=pos_inpr)
                    if hit == 0:
                        for ixdim in range(3):
                            pos_hit[ixray, ixdim] = res_pos[ixdim]
                        index_hit[ixray] = ixface
                        break
                    if stop_wrong_inpr and hit == 3:
                        break
            if (index_hit[ixray] != -1) or (stop_wrong_inpr and hit == 3):
                break # found an interesection
            ixstep = np.argmin(dist_hit)
            for ixdim in range(3):
                position[ixdim] += orientation[ixdim] * dist_hit[ixstep]
            distance += dist_hit[ixstep]
            for ixdim in range(3):
                if ixdim != ixstep:
                    dist_hit[ixdim] -= dist_hit[ixstep]
            dist_hit[ixstep] = 1 / abs(orientation[ixstep])


@numba.jit(nopython=True)
def _test_intersection(new_pos, position, orientation, faces, normal, index, pos_inpr=0):
    """
    Tests of ray from `position` in direction `hemisphere` crosses `faces[index]`.

    Helper function used by _numba_ray_intersect (which is used by GridSurfaceIntersection.ray_intersect)

    :param new_pos: (3, ) output array of the intersection position (will not be touched if intersection not found)
    :param position: (3, ) array of the position of the point of interest
    :param orientation: (3, ) array of the propogation direction of the ray
    :param faces: (N faces, 3 vertices per triangle, 3 dimensions) array of the surface faces
    :param normals: (N, 3) array of the face normals
    :param index: integer; index of the face with which the intersection will be tested
    :param pos_inpr: negative to ensure the hemisphere is misaligned with the normal, positive to ensure alignment.
    :returns: integer describing the result:
    - 0 if an intersection was succesfully found,
    - 1 if there is no intersection,
    - 2 if there is an intersection, but in the negative hemisphere direction,
    - 3 if there is an intersection but the inner product test failed.
    """
    base_point = np.empty(3)
    edge1 = np.empty(3)
    edge2 = np.empty(3)
    res_cross = np.empty(3)
    for ixdim in range(3):
        base_point[ixdim] = faces[index, 0, ixdim]
        edge1[ixdim] = faces[index, 1, ixdim] - base_point[ixdim]
        edge2[ixdim] = faces[index, 2, ixdim] - base_point[ixdim]
    offset = position - base_point

    _cross(orientation, edge2, res_cross)
    inv_det = 1. / np.sum(edge1 * res_cross)
    intercept1 = np.sum(res_cross * offset) * inv_det
    if intercept1 < -1e-5:
        return 1
    _cross(offset, edge1, res_cross)
    intercept2 = np.sum(res_cross * orientation) * inv_det
    intercepts = (intercept2 >= -1e-5) & ((intercept1 + intercept2) <= 1 + 2e-5)
    if not intercepts:
        return 1
    offset = position - (base_point + intercept1 * edge1 + intercept2 * edge2)
    inpr = 0.
    for ixdim in range(3):
        inpr += orientation[ixdim] * offset[ixdim]
    if inpr > 0:
        return 2
    inpr = 0.
    for ixdim in range(3):
        inpr += orientation[ixdim] * normal[index, ixdim]
    if ((pos_inpr > 0) and (inpr < 0)) or ((pos_inpr < 0) and (inpr > 0)):
        return 3
    for ixdim in range(3):
        new_pos[ixdim] = base_point[ixdim] + intercept1 * edge1[ixdim] + intercept2 * edge2[ixdim]
    return 0


@numba.jit(nopython=True)
def _cross(arr1, arr2, res):
    """
    Helper function to compute the cross product in numba.

    After running res will contain arr1 x arr2.

    :param arr1: (3, ) array with first input vector
    :param arr2: (3, ) array with second input vector
    :param res: (3, ) array, which will contain the cross-product of the input vectors
    """
    res[0] = arr1[1] * arr2[2] - arr1[2] * arr2[1]
    res[1] = arr1[2] * arr2[0] - arr1[0] * arr2[2]
    res[2] = arr1[0] * arr2[1] - arr1[1] * arr2[0]


def bounding(surface, resolution=1):
    """
    Computes the shape and affine of a grid spanning the bounding box with given voxel size

    :param surface: 2D surface in a 3D space
    :type surface: Mesh2D
    :param resolution: voxel size
    :type resolution: float
    :return: ((nx, ny, nz), 4x4 affine transformation)
    """
    offset = np.amin(surface.vertices, -1)
    affine = np.eye(4)
    affine[np.arange(3), np.arange(3)] = resolution
    affine[:3, -1] = offset
    shape = tuple(1 + np.ceil((np.amax(surface.vertices, -1) - offset) / resolution).astype('i4'))
    return shape, affine


def intersect_resolution(surface, resolution=1):
    """
    Computes the intersection between the surface and a grid spanning the bounding box with given voxel size.

    :param surface: 2D surface in a 3D space
    :type surface: Mesh2D
    :param resolution: voxel size
    :type resolution: float
    :return: GridSurfaceIntersection object
    """
    shape, affine = bounding(surface, resolution)
    return intersect(surface, shape, affine)


def intersect(surface, shape, affine=np.eye(4)):
    """
    Computes the intersection between surface and a grid.

    :param surface: 2D surface in a 3D space that intersects with the grid.
    :type surface: Mesh2D
    :param shape: tuple of volume shape (nx, ny, nz)
    :type shape: tuple
    :param affine: (4, 4) array describing the transformation from voxel space to the mm space that the surface is in
    :type affine: np.ndarray
    :return: GridSurfaceIntersection object
    """
    surf_vox = surface.apply_affine(affine)
    faces = surf_vox.vertices[:, surf_vox.faces]
    has_hit = -np.ones(shape, dtype='i4')
    vertex_hit = -np.ones((int(np.mean(shape) * 1000), 30), dtype='i4')
    minpos = np.floor(np.amin(faces, 1) + 0.5).astype('i4')
    maxpos = np.ceil(np.amax(faces, 1) - 0.5).astype('i4')
    face_shift = np.zeros((3, 3))
    edge = np.zeros(3)
    ixc = -2
    while ixc < 0:
        ixc = _intersect_volume_surface_jit(faces, has_hit, vertex_hit, minpos, maxpos,
                                            face_shift, edge, surface.normal().T)
        if ixc == -1:
            vertex_hit = -np.ones((vertex_hit.shape[0] * 2, vertex_hit.shape[1]), vertex_hit.dtype)
        elif ixc == -2:
            vertex_hit = -np.ones((vertex_hit.shape[0], vertex_hit.shape[1] * 2), vertex_hit.dtype)
    return GridSurfaceIntersection(surface, affine, has_hit, vertex_hit[:ixc, :])


@numba.jit(nopython=True)
def _intersect_volume_surface_jit(faces, has_hit, vertex_hit, minpos, maxpos, face_shift, edge, normals):
    """
    jit helper function for intersect_volume_surface. Do not call this directly
    """
    ixc = 0
    for ixface in range(faces.shape[2]):
        face = faces[:, :, ixface]
        for x in range(minpos[0, ixface], maxpos[0, ixface] + 1):
            if (x < 0) or (x >= has_hit.shape[0]):
                continue
            for ixpoint in range(3):
                face_shift[0, ixpoint] = face[0, ixpoint] - x
            for y in range(minpos[1, ixface], maxpos[1, ixface] + 1):
                if (y < 0) or (y >= has_hit.shape[1]):
                    continue
                for ixpoint in range(3):
                    face_shift[1, ixpoint] = face[1, ixpoint] - y
                for z in range(minpos[2, ixface], maxpos[2, ixface] + 1):
                    if (z < 0) or (z >= has_hit.shape[2]):
                        continue
                    for ixpoint in range(3):
                        face_shift[2, ixpoint] = face[2, ixpoint] - z
                    # test projection onto cross product of edges with
                    if _intersect_cube_triangle(face_shift.T, normals[ixface], edge):
                        index = has_hit[x, y, z]
                        if index == -1:
                            if ixc == vertex_hit.shape[0]:
                                return -1
                            has_hit[x, y, z] = ixc
                            vertex_hit[ixc, 0] = ixface
                            ixc += 1
                        else:
                            for ixhit in range(vertex_hit.shape[1]):
                                if vertex_hit[index, ixhit] == -1:
                                    vertex_hit[index, ixhit] = ixface
                                    break
                                if ixhit == vertex_hit.shape[1] - 1:
                                    return -2
    return ixc


@numba.jit(nopython=True)
def _intersect_cube_triangle(face, normal, edge):
    """
    Tests if `face` intersects with a unit cube centered on (0, 0, 0)

    Helper function for _intersect_volume_surface_jit

    :param face: (3 corners, 3 dimensions) array
    :param normal: (3, ) array of the pre-computed face normal
    :param edge: (3, ) array that will be used to store temporary values
    :return: True if intersects
    """
    # test the projection onto the cross product of the edges with the (x, y, z) orientations
    for ix_noedge in range(3):
        ixe1 = (ix_noedge + 1) % 3
        ixe2 = (ix_noedge + 2) % 3
        for ixdim in range(3):
            edge[ixdim] = face[ixe1, ixdim] - face[ixe2, ixdim]
        for ixdim_cross in range(3):
            ixd1 = (ixdim_cross + 1) % 3
            ixd2 = (ixdim_cross + 2) % 3
            pos0 = face[ixe1, ixd1] * edge[ixd2] - face[ixe1, ixd2] * edge[ixd1]
            pos1 = face[ix_noedge, ixd1] * edge[ixd2] - face[ix_noedge, ixd2] * edge[ixd1]
            if pos0 < pos1:
                min_pos = pos0
                max_pos = pos1
            else:
                min_pos = pos1
                max_pos = pos0
            rad = (abs(edge[ixd1]) + abs(edge[ixd2])) * 0.5
            if min_pos > rad or max_pos < -rad:
                return False
    rad = 0.5 * (abs(normal[0]) + abs(normal[1]) + abs(normal[2]))
    min_pos = rad * 2
    max_pos = -rad * 2
    for ixedge in range(3):
        pos = np.sum(face[ixedge, :] * normal)
        if pos < min_pos:
            min_pos = pos
        if pos > max_pos:
            max_pos = pos
    if min_pos > rad or max_pos < -rad:
        return False
    return True


def closest_surface(surface, grid, affine=None, pos_inpr=None, max_dist=np.inf):
    """
    Find for every non-zero point in the grid the closest point on the surface.

    :param surface: target surface
    :type surface: Mesh2D
    :param grid: (Nx, Ny, Nz) array, where for all non-zero elements the closest element on the target surface will be computed
    :param affine: (4, 4) array with the transformation from voxel space to the mm space in which the surface has been defined (default: no transformation)
    :param pos_inpr: whether the normal of the surface vertices should point towards (True) or away (False) from the grid point (default: no such constraint)
    :param max_dist: maximum distance from the surface to consider (in mm). Points beyond this maximum are set to -1.
    :return: (Nx, Ny, Nz) integer array, which is -1 for all zero elements in `grid` and contains the indices of the closest surface vertices otherwise
    """
    res_arr = np.zeros(grid.shape, dtype='i4')
    res_arr[grid == 0] = -1
    if affine is not None:
        surface = surface.apply_affine(affine)
        step_size = np.sqrt(np.sum(affine[:-1, :-1] ** 2, -1))
        max_dist /= np.mean(step_size)
    normal = surface.vertices if pos_inpr is None else (1 if pos_inpr else -1) * surface.normal(atpoint=True)
    closest_point(surface.vertices, res_arr, normal, test_orientation=pos_inpr is not None, max_dist=max_dist)
    return res_arr


@numba.jit(nopython=True)
def closest_point(points, grid, orientation, test_orientation=False, max_dist=np.inf):
    """
    Find for every point in the grid the closest point on the surface.

    :param points: (3, N) array of points in the grid in voxel coordinates
    :param grid: (Nx, Ny, Nz) array, where all elements that are not -1 will be replaced by the index of the closest point.
    :param orientation: (3, N) array of the hemisphere of each of these vertices (only grid vertices which are in the semi-hemisphere defined by this hemisphere will be considered
    :param max_dist: maximum distance from the surface to consider. Points beyond this maximum are set to -1.
    :param test_orientation: whether to test the hemisphere
    """
    for x in range(grid.shape[0]):
        for y in range(grid.shape[1]):
            for z in range(grid.shape[2]):
                if grid[x, y, z] != -1:
                    min_distsq = max_dist ** 2
                    ixpoint = -1
                    for ixp in range(points.shape[1]):
                        distsq = (points[0, ixp] - x) ** 2 + (points[1, ixp] - y) ** 2 + (points[2, ixp] - z) ** 2
                        if distsq < min_distsq:
                            if not test_orientation or (orientation[0, ixp] * (points[0, ixp] - x) +
                                                        orientation[1, ixp] * (points[1, ixp] - y) +
                                                        orientation[2, ixp] * (points[2, ixp] - z)) < 0:
                                ixpoint = ixp
                                min_distsq = distsq
                    grid[x, y, z] = ixpoint


def signed_distance(surface, shape, affine=None):
    """
    Signed distance from the surface to a grid with size `shape` in mm.

    Calls out to workbench to do the actual calculation.

    :param surface: reference surface
    :type surface: Mesh2D
    :param shape: (Nx, Ny, Nz) tuple defining the shape of the grid
    :param affine: (4, 4) array transforming the voxel to mm coordinates (default: no transformation)
    :return: (Nx, Ny, Nz)-array with the signed distance from the surface (negative inside the brain)
    """
    if affine is None:
        affine = np.eye(4)
    surf_filename = surface.as_temp_file()
    vol = tempfile.NamedTemporaryFile(suffix='.nii.gz', delete=False)
    vol.close()
    nibabel.Nifti1Image(np.zeros(shape), affine).to_filename(vol.name)
    output = tempfile.NamedTemporaryFile(suffix='.nii.gz', delete=False)
    output.close()
    subprocess.check_output('wb_command -create-signed-distance-volume %s %s %s' % (surf_filename, vol.name, output.name),
                            shell=True, stderr=subprocess.STDOUT)
    res = nibabel.load(output.name).get_data()
    for filename in [surf_filename, output.name, vol.name]:
        os.remove(filename)
    return res
