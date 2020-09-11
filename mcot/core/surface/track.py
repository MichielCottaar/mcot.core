"""
Run deterministic tractography along the surface
"""

import numba
from .mesh import Mesh2D
import numpy as np


@numba.jit(nopython=True)
def surface_step(idx_triangle, pos1, pos2, triangles, directions):
    """
    Takes a step across the given triangle to new position

    Location is defined by two vertices and the rel_pos between them

    :param idx_triangle: first vertex of current location
    :param pos1: position along first triangle axis
    :param pos2: position along second triangle axis
    :param triangles: (M, 3) array of triangles indices
    :param directions: (M, 2) array of gradients on the triangle surface
    :return: new triangle and relative position (int, float, float)
    """
    assert pos1 + pos2 <= 1
    orient = directions[idx_triangle]

    # distance to each triangle border
    if abs(orient[1]) < 1e-8:
        step_first = 0
    else:
        step_first = -pos2 / orient[1]
    if abs(orient[0]) < 1e-8:
        step_second = 0
    else:
        step_second = -pos1 / orient[0]
    if abs(orient[0] + orient[1]) < 1e-8:
        step_third = 0
    else:
        step_third = (1 - pos1 - pos2) / (orient[0] + orient[1])

    # determine which border got hit
    surface_hit = -1
    step_size = 1e3
    if step_first > 1e-8:
        surface_hit = 1
        step_size = step_first
    if 1e-8 < step_second < step_size:
        surface_hit = 2
        step_size = step_second
    if 1e-8 < step_third < step_size:
        surface_hit = 3
        step_size = step_third

    if surface_hit == -1:
        return idx_triangle, pos1, pos2

    # move particle
    new_pos1 = pos1 + orient[0] * step_size
    new_pos2 = pos2 + orient[1] * step_size
    new_idx1 = triangles[idx_triangle, 1 if surface_hit == 3 else 0]
    new_idx2 = triangles[idx_triangle, 1 if surface_hit == 1 else 2]

    # where along edge are we
    if surface_hit == 1:
        rel_pos = new_pos1
    else:
        rel_pos = new_pos2

    # correct for numeric drift
    if rel_pos > 1:
        rel_pos = 1
    if rel_pos < 0:
        rel_pos = 0

    # what triangle are we transitioning to
    for next_triangle in range(triangles.shape[0]):
        if next_triangle != idx_triangle:
            # find position on neighbouring triangle
            if triangles[next_triangle, 0] == new_idx1:
                if triangles[next_triangle, 1] == new_idx2:
                    return next_triangle, rel_pos, 0
                if triangles[next_triangle, 2] == new_idx2:
                    return next_triangle, 0, rel_pos
            elif triangles[next_triangle, 1] == new_idx1:
                if triangles[next_triangle, 2] == new_idx2:
                    return next_triangle, 1 - rel_pos, rel_pos
                if triangles[next_triangle, 0] == new_idx2:
                    return next_triangle, 1 - rel_pos, 0
            elif triangles[next_triangle, 2] == new_idx1:
                if triangles[next_triangle, 0] == new_idx2:
                    return next_triangle, 0, 1 - rel_pos
                if triangles[next_triangle, 1] == new_idx2:
                    return next_triangle, rel_pos, 1 - rel_pos
    return idx_triangle, new_pos1, new_pos2


@numba.jit(nopython=True)
def track_to_maximum(idx_triangle, pos1, pos2, triangles, directions):
    """
    Keep tracking until a ridge is found in the gradient

    :param idx_triangle: first vertex of current location
    :param pos1: position along first triangle axis
    :param pos2: position along second triangle axis
    :param triangles: (M, 3) array of triangles indices
    :param directions: (M, 2) array of gradients on the triangle surface
    :return: new triangle and relative position (int, float, float)
    """
    for _ in range(1000):
        idx_old = idx_triangle
        idx_triangle, pos1, pos2 = surface_step(idx_triangle, pos1, pos2, triangles, directions)
        if idx_old == idx_triangle:
            return idx_triangle, pos1, pos2
    raise ValueError("Maximum not found")


@numba.jit(nopython=True)
def _extract_ridge_values_helper(total, count, to_extract, triangles, directions):
    """
    Extracts ridge values from `to_extract`

    updates total and count (which should start at zeros)

    :param total: current total (starts at 0)
    :param count: current number of contributors (starts at 0)
    :param to_extract: (Nvertex, ) array of values to extract
    :param triangles: (Ntriangle, 3) index array into the vertices
    :param directions: (Ntriangles, 2) array of directions to track in
    """
    for idx_triangle in range(triangles.shape[0]):
        for internal_idx, pos1, pos2 in [
            (0, 0, 0),
            (1, 1, 0),
            (2, 0, 1),
        ]:
            idx_vertex = triangles[idx_triangle, internal_idx]
            max_idx, max_pos1, max_pos2 = track_to_maximum(idx_triangle, pos1, pos2, triangles, directions)

            x = to_extract[triangles[max_idx, 0]]
            if triangles[max_idx, 0] == idx_vertex and (1 - max_pos1 - max_pos2) > 1e-8:
                continue
            y = to_extract[triangles[max_idx, 1]]
            if triangles[max_idx, 1] == idx_vertex and max_pos1 > 1e-8:
                continue
            z = to_extract[triangles[max_idx, 2]]
            if triangles[max_idx, 2] == idx_vertex and max_pos2 > 1e-8:
                continue
            count[idx_vertex] += 1
            total[idx_vertex] += x * (1 - max_pos1 - max_pos2) + y * max_pos1 + z * max_pos2


def extract_ridge_values(surface: Mesh2D, orientations, to_extract):
    """
    Extract ridge values from `to_extract`

    :param surface: Mesh surface
    :param orientations: (N, 3) array of orientations in 3D space (N = number of faces)
    :param to_extract: (M, ) array of values to extract (M = number of vertices)
    :return: (M, ) array of extracted values
    """
    assert to_extract.shape == (surface.nvertices, )
    if orientations.shape[1] == 3:
        orientations = flatten_gradient(surface, orientations)
    total = np.zeros(surface.nvertices, dtype='float')
    count = np.zeros(surface.nvertices, dtype='int')

    _extract_ridge_values_helper(total, count, to_extract, surface.faces.T, orientations)
    return total / count


def flatten_gradient(surface: Mesh2D, orientations):
    """
    Redefine the orientations from 3D cartesian space to surface space

    :param surface: Mesh surface
    :param orientations: (N, 3) array of orientations in 3D space (N = number of faces)
    :return: (N, 2) orientations on triangular surface (N = number of faces)
    """
    normed_orient = orientations / np.sqrt(np.sum(orientations ** 2, 1))[:, None]
    all_pos = surface.vertices[:, surface.faces]
    assert (all_pos[0] == surface.vertices[0][surface.faces]).all()
    offset = all_pos[:, 1:, :] - all_pos[:, 0, None, :]

    length = np.sum(offset ** 2, 0)  # (2, Nfaces) length squared arrays
    cross = np.sum(offset[:, 0, :] * offset[:, 1, :], 0)  # (Nfaces, ) cross product for basis system
    alignment = np.sum(normed_orient.T[:, None, :] * offset, 0)  # (2, Nfaces) array with alignment

    b = (cross * alignment[0] - alignment[1] * length[0]) / (cross ** 2 - length[0] * length[1])
    a = (alignment[0] - b * cross) / length[0]
    return np.stack((a, b), -1) / np.sqrt(a ** 2 + b ** 2)[:, None]
