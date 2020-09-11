#!/usr/bin/env python
"""Randomly parcellates a surface"""
from mcot.core.surface import CorticalMesh
import nibabel as nib
import numpy as np
from mcot.core import kmedoids, write_gifti
from scipy.sparse import csgraph
import os.path as op
from loguru import logger


def uniform_centroids(dist_map, n_centroids):
    """
    Uniformly space `n_centroids` seeds in a naive way

    :param dist_map: sparse distance map
    :param n_centroids: number of seeds to place
    :return: (n_centroids, ) integer arrays with the indices of the seeds
    """
    def get_dist(idx_vertex):
        return csgraph.dijkstra(dist_map, indices=idx_vertex, directed=False)

    res = np.zeros(n_centroids, dtype='i4')
    res[0] = np.random.randint(0, dist_map.shape[0])
    dist = get_dist(res[0])
    for idx in range(1, n_centroids):
        res[idx] = np.argmax(dist)
        np.minimum(dist, get_dist(res[idx]), out=dist)
    return res


def run(surface: CorticalMesh, ncluster: int, max_iter=0):
    """
    Creates a random parcellation of the surface based on the distance between surface elements

    :param surface: cortical surface with N vertices
    :param ncluster: number of clusters
    :param max_iter: maximum number of iterations in the k-medoids (if 0 a fast voronoi evaluation is used)
    :return: (N, ) integer array with values from 0 to N-1
    """
    sparse_dist_map = surface.graph_point_point(weight='distance', dtype='f8')
    if max_iter == 0:
        centroids = uniform_centroids(sparse_dist_map, ncluster)
        dens_dist_map = csgraph.dijkstra(sparse_dist_map, indices=centroids, directed=False)
        return dens_dist_map.argmin(0)
    else:
        dense_dist_map = csgraph.dijkstra(sparse_dist_map, directed=False)
        return kmedoids.kmedoids(dense_dist_map, ncluster, max_loops=max_iter)[1]


def run_from_args(args):
    """
    Runs the script based on a Namespace containing the command line arguments
    """
    logger.info('starting %s', op.basename(__file__))
    mask = nib.load(args.mask).darrays[0].data > 0
    full_surface = CorticalMesh.read(args.surface)
    labels = run(
            surface=full_surface[mask],
            ncluster=args.ncluster,
            max_iter=args.iter,
    )
    full_labels = np.zeros(full_surface.nvertices, dtype='i4')
    full_labels[mask] = labels + 1

    write_gifti.write_gifti(args.out, [full_labels], brain_structure=full_surface.anatomy,
                            color_map={0: ('unk', (0, 0, 0, 0))})
    logger.info('ending %s', op.basename(__file__))


def add_to_parser(parser):
    """
    Creates the parser of the command line arguments
    """
    parser.add_argument("surface", help='.surf.gii file with the surface')
    parser.add_argument("mask", help='.shape.gii file with the surface mask')
    parser.add_argument("ncluster", type=int, help='number of clusters')
    parser.add_argument("out", help='.dlabel.gii dense GIFTI label output (0 outside of mask)')
    parser.add_argument("--iter", default=0, type=int, help='maximum number of iterations (default: 0)')
