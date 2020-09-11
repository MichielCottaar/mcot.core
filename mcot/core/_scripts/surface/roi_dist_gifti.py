#!/usr/bin/env python
"""
For every vertex computes the distance to the closest point in a ROI
"""
from mcot.core.surface import CorticalMesh
from mcot.core import write_gifti
import nibabel as nib
import numpy as np
from scipy.sparse import csgraph
from scipy import sparse


def sparse_min(sparse_matrix, axis=None):
    """
    Computes the minium of a sparse matrix ignoring the zero elements

    :param sparse_matrix:
    :return:
    """
    coo_mat = sparse_matrix.tocoo()
    if axis is None:
        return coo_mat.data.min()
    max_val = coo_mat.data.max()
    coo_mat.data -= max_val + 1  # make sure all numbers are negative
    arr = coo_mat.min(axis=axis).toarray().flatten()
    idx = np.array(coo_mat.argmin(axis=axis)).flatten()
    arr[arr == 0] = np.nan
    idx[arr == 0] = -1
    arr += max_val + 1
    coo_mat.data += max_val + 1  # reset values to restore matrix if tocoo() did not make a copy
    return arr, idx


def run(surface: CorticalMesh, roi: np.ndarray):
    """
    Finds the shortest route to the ROI for every vertex

    :param surface: cortical surface
    :param roi: region of interest
    :return:
    """
    graph = surface.graph_point_point('distance', dtype='float')
    roi_dist, roi_idx = sparse_min(graph[roi, :], 0)
    stack1 = sparse.vstack((roi_dist[~roi], graph[~roi, :][:, ~roi])).tocsr()
    full_mat = sparse.hstack((np.append(0, roi_dist[~roi])[:, None], stack1))

    dist_matrix, predecessors = csgraph.dijkstra(full_mat, indices=0, directed=False,
                                                 return_predecessors=True)
    full_dist_matrix = np.zeros(surface.nvertices)
    full_dist_matrix[~roi] = dist_matrix[1:]

    original_predecessors = np.zeros(predecessors.shape)
    original_idx = np.arange(surface.nvertices)
    original_predecessors[predecessors != 0] = original_idx[~roi][predecessors[predecessors != 0] - 1]

    original_predecessors[predecessors == 0] = original_idx[roi][roi_idx[predecessors[predecessors == 0] - 1]]
    full_predecessors = -np.ones(surface.nvertices, dtype='i4')
    full_predecessors[~roi] = original_predecessors[1:]

    closest_vertex = full_predecessors.copy()
    closest_vertex[roi] = original_idx[roi]
    for _ in range(surface.nvertices):
        replace = (~roi[closest_vertex]) & (closest_vertex != -1)
        if not replace.any():
            break
        closest_vertex[replace] = full_predecessors[closest_vertex[replace]]

    return full_dist_matrix, full_predecessors, closest_vertex


def run_from_args(args):
    """
    Runs the script based on a Namespace containing the command line arguments
    """
    if (
            args.output_dist is None and
            args.output_closest_vertex is None and
            args.output_predecessor_vertex is None and
            args.project is None
    ):
        raise ValueError("No output files requested, set at least one of -p, -od, -orv, or -onv")
    surface = CorticalMesh.read(args.surface)
    roi = nib.load(args.roi).darrays[0].data != 0
    if args.project is not None:
        to_project = nib.load(args.project[0]).darrays[0].data
    distance, predecessor, closest_vertex = run(
            surface, roi
    )
    if args.output_dist is not None:
        write_gifti(args.output_dist, [distance], brain_structure=surface.anatomy)
    if args.output_closest_vertex is not None:
        write_gifti(args.output_closest_vertex, [closest_vertex], brain_structure=surface.anatomy)
    if args.output_predecessor_vertex is not None:
        write_gifti(args.output_predecessor_vertex, [predecessor], brain_structure=surface.anatomy)
    if args.project is not None:
        projected = to_project[closest_vertex]
        write_gifti(args.project[1], [projected], brain_structure=surface.anatomy)


def add_to_parser(parser=None):
    """
    Creates the parser of the command line arguments
    """
    parser.add_argument('surface', help='.surf.gii input surface along which the distance will be computed')
    parser.add_argument('roi', help='.shape.gii with the target ROI as non-zeros')
    parser.add_argument('-od', '--output_dist', help='.shape.gii with the distance to the ROI (0 in ROI)')
    parser.add_argument('-ocv', '--output_closest_vertex', help='.shape.gii with the index of the closest vertex')
    parser.add_argument('-opv', '--output_predecessor_vertex',
                        help='.shape.gii with the index the neighbouring vertex that is on the shortest ' +
                             'route to the ROI')
    parser.add_argument('-p', '--project', nargs=2, help='<input shape.gii> <output.shape.gii> projects data from '
                                                         'one ROI onto the full brain based on the closest vertex')
