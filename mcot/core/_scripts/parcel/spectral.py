#!/usr/bin/env python
"""Parcellates the surface based on clustering similar fingerprints"""
from mcot.core.surface import CorticalMesh
import numpy as np
from loguru import logger
from ..surface import gradient
from ..parcel import cluster, random
from mcot.core import write_gifti
from mcot.core.greyordinate import GreyOrdinates


def run(surface: CorticalMesh, features, n_iter=20, metric='spearman', method='spectral'):
    """
    Repeatedly identify the border between clusters in a spotlight fashion

    :param surface: anatomical surface with N vertices
    :param features: (N, M) array of surface features
    :param n_iter: number of  iterations
    :param metric: which metric to use for the similarity between neighbouring vertices
    :param method: clustering method ('spectral', 'affinity', or 'DBSCAN')
    :return: (N, ) array with fraction of times a vertex was on the border between the clusters
    """
    sim_mat = gradient.distance(surface.graph_point_point(include_diagonal=False), features, metric=metric)
    on_border = np.zeros(surface.nvertices)
    for n_clusters in np.linspace(10, 100, n_iter):
        logger.info('Randomly parcellating cortex in %d clusters', n_clusters)
        parcellation = random.run(surface, int(n_clusters))
        logger.info('Running %s clustering on each cluster', method)
        for idx_cluster in range(int(n_clusters)):
            logger.debug('Processing cluster %d (out of %d)', idx_cluster, int(n_clusters))
            mask = parcellation == idx_cluster
            labels = cluster.cluster(sim_mat[mask, :][:, mask], method=method, n_clusters=2)
            graph = surface[mask].graph_point_point()
            smooth_labels = (graph.dot(labels) + labels) / (graph.dot(np.ones(mask.sum())) + 1)
            # edge detection only guaranteed to work properly for 2 clusters
            on_border[mask] += (smooth_labels % 1.) != 0.
    return on_border / n_iter


def run_from_args(args):
    """
    Runs the script based on a Namespace containing the command line arguments
    """
    surf = CorticalMesh.read(args.surface)
    go = GreyOrdinates.from_filename(args.features)
    features = go.surface(surf.anatomy)
    mask = np.isfinite(features)
    res = run(
            surf[mask],
            features,
            n_iter=args.iter,
            metric=args.metric,
            method=args.cluster,
    )
    full_res = np.zeros(surf.nvertices)
    full_res[mask] = res
    write_gifti(args.output, [full_res], surf.anatomy)


def add_to_parser(parser):
    """
    Creates the parser of the command line arguments
    """
    parser.add_argument("surface", help='.surf.gii file describing the surface (WM/GM boundary, mid, or pial)')
    parser.add_argument("features", help='GIFTI/CIFTI file with the features of interest')
    parser.add_argument("output", help='GIFTI file of the most consistent edge locations (same type as input)')
    parser.add_argument("--iter", type=int, default=20, help='number of searchlights for every vertex')
    parser.add_argument("-m", '--metric', choices=gradient.dist_options, default='euclidean',
                        help='how to compute the similarity with the neighbours (default: euclidean)')
    parser.add_argument("-c", '--cluster', choices=('spectral', 'affinity', 'dbscan'), default='spectral',
                        help='Clustering algorithm to use')
