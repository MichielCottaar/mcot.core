#!/usr/bin/env python
"""Clusters the surface based on provided features"""
from loguru import logger
from ..surface import gradient
from mcot.core.surface import CorticalMesh
from mcot.core import write_gifti
import numpy as np
from sklearn import cluster as skcluster
from mcot.core.greyordinate import GreyOrdinates


def cluster(similarity, method='spectral', n_clusters=10, **kwargs):
    """
    Cluster based on a similarity graph

    :param similarity: (N, N) sparse similarity matrix
    :param method: clustering method ('spectral', 'affinity', or 'DBSCAN')
    :param n_clusters: number of clusters
    :return: (N, ) integer array of labels
    """
    logger.info("Starting to cluster %s array using %s clustering", similarity.shape, method)
    if method.lower() == 'spectral':
        algorithm = skcluster.SpectralClustering(n_clusters=n_clusters, affinity='precomputed', **kwargs)
    elif method.lower() == 'affinity':
        algorithm = skcluster.AffinityPropagation(affinity='precomputed', preference=np.mean(similarity), **kwargs)
    elif method.lower() == 'dbscan':
        algorithm = skcluster.DBSCAN(metric='precomputed', n_clusters=n_clusters, **kwargs)
    return algorithm.fit_predict(similarity)


def run(surface, features, metric='spearman', method='spectral', n_clusters=10):
    """
    Clusters the surface based on the provided features

    :param surface: anatomical surface with N vertices
    :param features: (M, N) array of surface features
    :param metric: which metric to use for the similarity between neighbouring vertices
    :param method: clustering method ('spectral', 'affinity', or 'DBSCAN')
    :param n_clusters: number of clusters
    :return: (N, ) integer array of labels
    """
    sim_graph = gradient.distance(surface.graph_point_point(include_diagonal=False), features, metric=metric)
    return cluster(sim_graph.tocsr(), method=method, n_clusters=n_clusters)


def run_from_args(args):
    """
    Runs the script based on a Namespace containing the command line arguments
    """
    surface = CorticalMesh.read(args.surface)
    go = GreyOrdinates.from_filename(args.features)
    features = go.surface(surface.anatomy)
    mask = np.isfinite(features)
    labels = run(
            surface[mask],
            features[..., mask],
            metric=args.metric,
            method=args.cluster,
            n_clusters=args.n_clusters,
    )
    full_labels = -np.ones(surface.nvertices, dtype=labels.dtype)
    full_labels[mask] = labels
    write_gifti(args.output, [full_labels],
                brain_structure=surface.anatomy,
                color_map={-1: ('???', (0, 0, 0, 0))})


def add_to_parser(parser):
    """
    Creates the parser of the command line arguments
    """
    parser.add_argument("surface", help='.surf.gii file describing the surface (WM/GM boundary, mid, or pial)')
    parser.add_argument("features", help='GIFTI/CIFTI file with the features of interest')
    parser.add_argument("output", help='.label.gii GIFTI file with the cluster labels')
    parser.add_argument("-m", '--metric', choices=gradient.dist_options, default='euclidean',
                        help='how to compute the similarity with the neighbours (default: euclidean)')
    parser.add_argument("-c", '--cluster', choices=('spectral', 'affinity', 'dbscan'), default='spectral',
                        help='Clustering algorithm to use')
    parser.add_argument("-n", "--n_clusters", type=int, default=10,
                        help='number of clusters to fit (ignored by affinity)')
