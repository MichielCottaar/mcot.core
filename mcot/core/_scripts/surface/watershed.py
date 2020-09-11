#!/usr/bin/env python
"""
Creates maps of gyri or sulci using watershed algorithm

The watershed algorithm sorts all vertices based on sulcal depth (or some other metric) and then visits them one by one:

    - if none of the neighbour have been visited yet (i.e., the vertex is a local minimum) the vertex is given a new label
    - if only one neighbour has a label (or all neighbours have the same label), the vertex is given that label
    - if different neigbours of the vertex have different labels one of the following two actions is taken:

        - if either of the masks is below the minumum size (in mm^2) or below the minimum depth
          (i.e., the difference between the lowest value of the metric and the value in the current vertex)
          then the two parcels are merged (and the vertex is given the value of the new parcel)
        - otherwise both parcels are kept separate and the vertex is identified as being on the ridge between these
          parcels.
"""
from loguru import logger
from mcot.core.surface import CorticalMesh
from mcot.core import write_gifti
import nibabel as nib
import numpy as np
import numba
from scipy import optimize


@numba.jit(nopython=True)
def _watershed(segment, neighbours, depth, segment_depth, min_depth, size, segment_size, min_size, fill_ridge):
    """
    Actually runs watershed based on a sorted neighbourhood graph

    :param segment: (N, ) output array to which the segmentation will be written
    :param neighbours: (N, M) index array with the neighbours for each vertex (padded with -1)
    :param depth: (N, ) array which will contains the depth for each vertex
    :param segment_depth: (N, ) array which will contain the minimal depth for each segment
    :param min_depth: minimal depth of the groups
    :param size: (N, ) array which will contains the size of each vertex
    :param segment_size: (N, ) array which will contains the size of each segment
    :param min_size: minimal depth of the groups
    :param fill_ridge: if True, fills the ridge with the value of the smaller of the neighbouring parcels
    """
    new_segment = 1  # index of new parcel
    for idx in range(segment.size):
        segment[idx] = -1  # -1 means still to be assigned or ridge
    for idx in range(segment.size):
        proposed_segment = -2  # -2 means no neighbours are part of an existing parcel
        for idx2 in range(neighbours.shape[1]):
            if neighbours[idx, idx2] == -1:
                # reached the padding, so can stop now
                break
            other_idx = neighbours[idx, idx2]
            if segment[other_idx] == -1:
                # unassigned neighbours are irrelevant
                continue
            if proposed_segment == -2:
                # first neighbour of existing parcel; just store it
                proposed_segment = segment[other_idx]
            elif proposed_segment != segment[other_idx]:
                # another neighbour has a different parcel
                if (
                        (depth[idx] - segment_depth[proposed_segment]) < min_depth or
                        (depth[idx] - segment_depth[segment[other_idx]]) < min_depth or
                        (segment_size[proposed_segment] < min_size) or
                        (segment_size[segment[other_idx]] < min_size)
                ):
                    # merges the two segments
                    to_replace = segment[other_idx]
                    for idx_replace in range(idx):
                        if segment[idx_replace] == to_replace:
                            segment[idx_replace] = proposed_segment
                    segment_size[proposed_segment] += segment_size[to_replace]
                    segment_depth[proposed_segment] = min(segment_depth[proposed_segment],
                                                          segment_depth[to_replace])
                elif not fill_ridge:
                    # on the ridge
                    proposed_segment = -1
                    break
                elif segment_size[segment[other_idx]] < segment_size[proposed_segment]:
                    # join the smaller segment; continue checking if this vertex neighbours an even smaller segment
                    proposed_segment = segment[other_idx]

        # proposed_segment is now one of:
        # -2) no neighbour found
        # -1) neighbours belong to different segments (i.e., ridge)
        # 1--N) only neighbours from the same segment found
        if proposed_segment == -2:
            segment[idx] = new_segment
            segment_depth[new_segment] = depth[idx]
            segment_size[new_segment] = size[idx]
            new_segment += 1
        else:
            segment[idx] = proposed_segment
            if proposed_segment != -1:
                segment_size[proposed_segment] += size[idx]


def run(surface: CorticalMesh, depth, min_depth=0, min_size=0, fill_ridge=False):
    """
    Runs watershed on the graph generated on the surface

    :param surface: mesh representing cortical surface
    :param depth: (N, ) array where N is number of vertices
    :param min_depth: minimum offset between minimum along ridge and minimum depth for each segment
    :param min_size: minimal size of each segment in mm^2
    :param fill_ridge: if True, fills the ridge with the value of the smaller of the neighbouring parcels
    :return: (N, ) array with segmentation (-1 on edges, 1-M within segments, where M is the number of segments)
    """
    graph = surface.graph_point_point().tocsr()
    max_edges = graph.sum(-1).max()
    neighbours = -np.ones((surface.nvertices, max_edges), dtype='i4')
    sorting = np.argsort(depth)
    sorted_graph = graph[:, sorting][sorting, :]
    for idx in range(surface.nvertices):
        ptrs = sorted_graph.indptr[idx:]
        neighbours[idx, :ptrs[1] - ptrs[0]] = sorted_graph.indices[ptrs[0]:ptrs[1]]

    segment = np.zeros(surface.nvertices, dtype='i4')
    _watershed(
            segment, neighbours,
            depth[sorting], np.zeros(surface.nvertices, dtype=depth.dtype), min_depth,
            surface.size_vertices()[sorting], np.zeros(surface.nvertices), min_size, fill_ridge
    )

    cleaned_segment = np.zeros(surface.nvertices, dtype='i4')
    cleaned_segment[segment >= 0] = np.unique(segment[segment >= 0], return_inverse=True)[1] + 1
    cleaned_segment[segment == -1] = -1
    cleaned_segment[segment == -2] = -2

    orig_segment = np.zeros(surface.nvertices, dtype='i4')
    orig_segment[sorting] = cleaned_segment
    logger.info(f'{np.unique(segment[segment > 0]).size} segments found')
    return orig_segment


def run_nparcels(surface: CorticalMesh, depth, nparcels, min_size=0, fill_ridge=False):
    """
    Returns a parcellation with requested number of parcels based on watershed algorithm

    min_depth of the parcels in the watershed algorithm is determined to match the number of samples

    :param surface: mesh representing cortical surface
    :param depth: (N, ) array where N is number of vertices
    :param nparcels: number of parcels in the output (not counting edges)
    :param min_size: minimal size of each segment in mm^2
    :param fill_ridge: if True, fills the ridge with the value of the smaller of the neighbouring parcels
    :return: (N, ) array with segmentation (-1 on edges, 1-`nparcels` within segments)
    """
    res = optimize.root_scalar(
        lambda min_depth: run(surface, depth, min_depth, min_size, fill_ridge).max() - nparcels,
        bracket=(depth.min(), depth.max()), method='bisect',
    )
    return run(surface, depth, res.root, min_size, fill_ridge=fill_ridge)


def run_from_args(args):
    """
    Runs the script based on a Namespace containing the command line arguments
    """
    surface = CorticalMesh.read(args.surface)
    arr = nib.load(args.depth).darrays[0].data
    if args.flip:
        arr *= -1
    use = np.isfinite(arr) & (arr != 0)
    if args.Nparcels is not None:
        segments = run(
            surface[use],
            arr[use],
            min_depth=args.min_depth,
            min_size=args.min_size,
            fill_ridge=args.fill_ridge,
        )
    else:
        segments = run_nparcels(
            surface[use],
            arr[use],
            nparcels=args.Nparcels,
            min_size=args.min_size,
            fill_ridge=args.fill_ridge,
        )
    full_segment = np.zeros(arr.size, dtype='i4')
    full_segment[use] = segments
    write_gifti(args.out, [full_segment], color_map='default',
                brain_structure=surface.anatomy)


def add_to_parser(parser):
    """
    Creates the parser of the command line arguments
    """
    parser.add_argument('surface', help='.surf.gii file with the surface')
    parser.add_argument('depth', help='.shape.gii with array to drive segmentation (e.g., sulcal depth or curvature)')
    parser.add_argument('out', help='.label.gii output filename that will contain the segmentation')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('-md', '--min_depth', type=float, default=0,
                       help='Minimum offset between minimal depth and lowest depth on ridge for each segment')
    group.add_argument('-N', '--Nparcels', type=int,
                       help='Target number of parcels (if set will automatically determine the required min_depth)')
    parser.add_argument('-ms', '--min_size', default=0., type=float,
                        help='Minimum size for each segment')
    parser.add_argument('-f', '--flip',
                        help='Flips the sign of the metric (so that the watershed runs in the opposite direction)')
    parser.add_argument('-r', '--fill_ridge', action='store_true',
                        help='If True fills the edges between the parcels with values from one of the parcels (preferring growing the smaller parcels)')
