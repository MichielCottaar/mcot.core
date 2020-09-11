#!/usr/bin/env python
"""
Computes the feature similarity with the neighbouring vertices

All relevant distance metric from scipy are included (https://docs.scipy.org/doc/scipy/reference/spatial.distance.html)
as well as histogram intersection (http://blog.datadive.net/histogram-intersection-for-change-detection/).
"""
import numpy as np
from scipy import spatial, sparse
from mcot.core.surface import CorticalMesh
from mcot.core.greyordinate import GreyOrdinates
from concurrent import futures
from nibabel import cifti2


scipy_options = (
    'braycurtis', 'canberra', 'cityblock', 'correlation',
    'cosine', 'euclidean', 'hamming', 'jaccard', 'jensenshannon',
    'minkowski', 'sqeuclidean',
)

dist_options = ('histogram_intersection', ) + scipy_options


def _histogram_intersection(arr1, arr2):
    normed1 = np.array(arr1, dtype='float')
    normed1 /= normed1.sum()
    normed2 = np.array(arr2, dtype='float')
    normed2 /= normed2.sum()
    return np.minimum(normed1, normed2).sum()


def distance(graph, features, metric='euclidean'):
    """
    Computes the distance between neighbouring feature sets

    :param graph: (N, N) sparse matrix indicating which features are neighbouring
    :param features: (N, M) for M features defined at N locations
    :param metric: which distance operator to use (one of `dist_options`)
    :return: (N, N) sparse matrix with the distance between each feature pair (same structure as `graph`)
    """
    executor = futures.ThreadPoolExecutor(10)
    metric = metric.lower()
    if metric not in dist_options:
        raise ValueError(f"{metric} is not a valid distance metric, should be one of: {', '.join(dist_options)}")
    if metric == 'histogram_intersection':
        dist_func = _histogram_intersection
    else:
        dist_func = getattr(spatial.distance, metric)
    coo = graph.tocoo()

    data = np.zeros(coo.data.size)

    def to_run(idx):
        data[idx] = dist_func(features[coo.row[idx], :], features[coo.col[idx], :])
    list(executor.map(to_run, range(coo.data.size), chunksize=100))
    return sparse.coo_matrix((data, (coo.row, coo.col)), shape=graph.shape)


def reduce(distance, method='median'):
    """
    Summarises a NxN sparse matrix of distances as a (N, ) array of distances

    :param distance: (N, N) sparse matrix of distances between neighbouring vertices
    :return: (N, ) array with the summarised distance
    """
    if method not in ('min', 'max', 'median', 'mean'):
        raise ValueError("Reduction method {method} has not been recognised")
    as_mat = distance.tocsr()
    res = np.zeros(distance.shape[0])
    for idx, i1, i2 in zip(range(distance.shape[0]), as_mat.indptr[:-1], as_mat.indptr[1:]):
        if i1 == i2:
            res[idx] = np.nan
        else:
            res[idx] = getattr(np, method)(as_mat.data[i1:i2])
    return res


def run(surface: CorticalMesh, arr, distance_metric='euclidean', reduction='median'):
    """
    Computes the reduced distance between neighbouring vertices

    :param surface: surface array
    :param arr: (N, M) array for M features on N vertices
    :param distance_metric: distance metric to use
    :param reduction: reduction method to use to average over all connections with a vertex
    :return: (N, ) array with the summarised distance
    """
    if arr.shape[0] != surface.nvertices:
        raise ValueError(f"Expected ({surface.nvertices}, N) array, got {arr.shape} array")
    graph = surface.graph_point_point(dtype=float, include_diagonal=False)
    dist = distance(graph, arr, metric=distance_metric)
    return reduce(dist, method=reduction)


def run_from_args(args):
    """
    Runs the script based on a Namespace containing the command line arguments
    """
    go = GreyOrdinates.from_filename(args.features)
    if args.transpose:
        go = go.transpose()
    surface = CorticalMesh.read(args.surface)
    surface_mask, features = go.surface(surface.anatomy, partial=True)
    if args.exclude_same_region:
        features = features[go.other_axes[0].name != surface.anatomy.cifti]
    res = run(surface[surface_mask], features.T, args.distance, args.reduction)
    new_bm = cifti2.BrainModelAxis.from_surface(surface_mask, nvertex=surface.nvertices, name=surface.anatomy.cifti)
    GreyOrdinates(res, new_bm).to_filename(args.output)


def add_to_parser(parser):
    """
    Creates the parser of the command line arguments
    """
    parser.add_argument('features', help='GIFTI or CIFTI file with the features to select on')
    parser.add_argument('surface', help='surface file')
    parser.add_argument('output', help='GIFTI or CIFTI output file')
    parser.add_argument('-d', '--distance', choices=dist_options, default='euclidean',
                        help='distance metric to use (default: euclidean)')
    parser.add_argument('-r', '--reduction', choices=('min', 'max', 'median', 'mean'), default='median',
                        help='reduction method to summarise all the edges of one vertex (default: median)')
    parser.add_argument('-T', '--transpose', action='store_true',
                        help='Transpose the features array before extracting features (only makes sense for dconn)')
    parser.add_argument('--exclude-same-region', action='store_true',
                        help="in a dconn file do not include connections to the same cortex as features")
