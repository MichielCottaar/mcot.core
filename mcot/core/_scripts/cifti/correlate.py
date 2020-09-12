#!/usr/bin/env python
"""Correlates two CIFTI files"""
from nibabel import cifti2
import numpy as np
from scipy import stats
from loguru import logger
import os.path as op
import time


def correlate_arr(arr1, arr2, collapse=False):
    """
    Computes the Pearson r correlation for two arrays

    :param arr1: (M, N) array for N observations of M variables
    :param arr2: (M, K) array for K obsevations of M variables
    :param collapse: assume N==K and only return (N, ) array with the r correlation
    :return: (N, K) array with the r correlation (or (N, ) array if collapse is set)
    """
    arr1 = np.atleast_2d(arr1)
    arr2 = np.atleast_2d(arr2)
    if collapse:
        assert arr1.shape == arr2.shape
    demeaned1 = arr1 - arr1.mean(0)
    demeaned2 = arr2 - arr2.mean(0)
    if collapse:
        r_num = np.einsum('ij,ij->j', demeaned1, demeaned2)
        r_den = np.sqrt(np.sum(demeaned1 ** 2, 0) * np.sum(demeaned2 ** 2, 0))
    else:
        r_num = demeaned1.T.dot(demeaned2)
        r_den = np.sqrt(
                np.sum(demeaned1 ** 2, 0)[:, None] * np.sum(demeaned2 ** 2, 0)[None, :]
        )
    return r_num / r_den


class FakeDConn(object):
    """
    Pretends to be a nibabel image of the dconn file
    """
    def __init__(self, dtseries):
        self.dataobj = FakeDConnDataObject(dtseries)


class FakeDConnDataObject(object):
    """
    Pretends to be a nibabel dataobject of a dconn file
    """
    def __init__(self, dtseries):
        self.dtseries = dtseries

    def __getitem__(self, item):
        slc1, slc2 = item
        return correlate_arr(
                self.dtseries.dataobj[:, slc1],
                self.dtseries.dataobj[:, slc2]
        )


def correlate(dconn1, dconn2, select_features=slice(None)):
    """
    Correlates two memory mapped arrays

    :param dconn1: (M, N) nibabel Cifti2Image (M features for N greyordinates)
    :param dconn2: (M, N) nibabel Cifti2Image (M features for N greyordinates)
    :param select_features: which features to select
    :return: (N, ) in-memory array
    """
    logger.info('Correlating {}'.format(select_features))
    ngrey = dconn1.shape[1]
    step = int(np.ceil(1e8 / ngrey))
    res = np.zeros(ngrey)
    loading_time = 0
    correlate_time = 0
    for idx in range(0, ngrey, step):
        logger.debug('Calculating correlation for grayordinates between {} and {}'.format(idx, idx + step))
        start_time = time.time()
        if isinstance(select_features, list) or isinstance(select_features, tuple):
            local1 = np.concatenate([dconn1.dataobj[sf, idx:idx + step] for sf in select_features], 0)
            local2 = np.concatenate([dconn2.dataobj[sf, idx:idx + step] for sf in select_features], 0)
        else:
            local1 = np.array(dconn1.dataobj[select_features, idx:idx + step])
            local2 = np.array(dconn2.dataobj[select_features, idx:idx + step])
        mid_time = time.time()
        loading_time += mid_time - start_time
        for sub_idx in range(local1.shape[1]):
            res[idx + sub_idx] = stats.spearmanr(local1[:, sub_idx], local2[:, sub_idx])[0]
        correlate_time += time.time() - mid_time
    logger.info('Loading data took {} minutes'.format(loading_time / 60))
    logger.info('Correlating the data took {} minutes'.format(correlate_time / 60))
    return res


def run(dconn1, dconn2, split_greyordinates: cifti2.BrainModelAxis=None):
    """
    Computes the correlation between dense connectomes

    :param dconn1: (M, N) nibabel Cifti2Image
    :param dconn2: (M, N) nibabel Cifti2Image
    :param split_greyordinates: defines M voxels/vertices to select sub-set of features the features
    :return: dictionary mapping names of target regions to (N, ) arrays
    """
    logger.info('Computing full correlation')

    res = {'full': correlate(dconn1, dconn2)}
    if split_greyordinates is None:
        return res

    cortex = []
    sub_cortical = []
    left = []
    right = []
    for struc_name, struc_slice, struc_bm in split_greyordinates.iter_structures():
        short_name = struc_name.split('_', maxsplit=2)[2]
        if 'CORTEX' in short_name:
            logger.info('Computing correlation relative to %s', short_name)
            res[short_name] = correlate(dconn1, dconn2, struc_slice)

        if struc_bm.is_surface.all():
            cortex.append(struc_slice)
        else:
            sub_cortical.append(struc_slice)
        if 'LEFT' in short_name:
            left.append(struc_slice)
        if 'RIGHT' in short_name:
            right.append(struc_slice)

    for name in ('cortex', 'sub_cortical', 'left', 'right'):
        logger.info('Computing correlation relative to all %s brain structures', name)
        res[name] = correlate(dconn1, dconn2, locals()[name])

    return res


def run_from_args(args):
    """
    Runs the script based on a Namespace containing the command line arguments
    """
    logger.info('starting %s', op.basename(__file__))
    dconn1 = cifti2.load(args.input)
    axes1 = dconn1.header.get_axis(dconn1)
    dconn2 = cifti2.load(args.reference)
    axes2 = dconn2.header.get_axis(dconn2)
    if not (isinstance(axes1[1], cifti2.BrainModelAxis) or isinstance(axes1[1], cifti2.ParcelsAxis)):
        raise ValueError("Columns should have greyordinates or parcels")
    if axes1[1] != axes2[1]:
        raise ValueError("Compared CIFTI files should have the same greyordinates/parcels along the columns")
    if args.as_dconn:
        if isinstance(axes1[0], cifti2.SeriesAxis):
            axes1 = (axes1[1], ) * 2
            dconn1 = FakeDConn(dconn1)
        elif isinstance(axes2[0], cifti2.SeriesAxis):
            axes2 = (axes2[1], ) * 2
            dconn2 = FakeDConn(dconn2)
    if args.split_dconn and not isinstance(axes1[0], cifti2.BrainModelAxis):
        raise ValueError("Rows should be greyordinates when using the option --split_dconn")
    if axes1[0] != axes2[0]:
        raise ValueError("Compared CIFTI files should have the same features along the rows")
    as_dict = run(
            dconn1=dconn1,
            dconn2=dconn2,
            split_greyordinates=axes1[0] if args.split_dconn else None,
    )
    scalar = cifti2.ScalarAxis(list(as_dict.keys()))
    arr = np.stack(list(as_dict.values()), -1)
    cifti2.Cifti2Image(arr.T, header=(scalar, axes1[1])).to_filename(args.output)
    logger.info('ending %s', op.basename(__file__))


def add_to_parser(parser):
    """
    Creates the parser of the command line arguments
    """
    parser.add_argument("-i", "--input", required=True,
                        help="input parcellated or dense CIFTI file")
    parser.add_argument("-r", "--reference", required=True,
                        help="reference dense connectome (should have the same greyordinates or parcels)")
    parser.add_argument("-o", "--output", required=True,
                        help="output dscalar or pscalar file with the correlations")
    parser.add_argument("--split_dconn", action='store_true',
                        help='If set splits the feature greyordinates into different groups (works for dconn or pdconn)')
    parser.add_argument("--as_dconn", action='store_true',
                        help="If set treats dense timeseries files and dense connectome files " +
                             "using the Pearson r correlation")
