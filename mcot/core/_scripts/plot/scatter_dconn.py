#!/usr/bin/env python
"""Scatter plot of dense connectomes"""
import cifti
import matplotlib.pyplot as plt
from mcutils.plot import density_scatter
from mcutils.scripts.plot.scatter_dscalar import find_common
import numpy as np
import nibabel as nib
import os.path as op
from loguru import logger


def sub_sample(values, slice1, slice2, nmax=int(1e5)):
    if isinstance(slice1, slice):
        values = values[slice1, :]
        slice1 = None
    if isinstance(slice2, slice):
        values = values[:, slice2]
        slice2 = None
    nval = ((values.shape[0] if slice1 is None else len(slice1)) *
            (values.shape[1] if slice2 is None else len(slice2)))
    nsample = int(np.ceil(nval / nmax))

    def get_idx(idx):
        if slice1 is None and slice2 is None:
            return values[idx::nsample, idx::nsample]
        elif slice1 is None:
            return values[idx::nsample][:, slice2[::nsample]]
        elif slice2 is None:
            return values[:, idx::nsample][slice1[::nsample]]
        else:
            return values[slice1[idx::nsample, None], slice2[None, idx::nsample]]

    return np.concatenate([get_idx(idx).flat for idx in range(nsample)])


def run(x, x_bm: cifti.BrainModel, y, y_bm: cifti.BrainModel, xlog=False, ylog=False, diagonal=False):
    """
    Creates a plot of

    :param x: (N, N) nibabel Cifti2Image
    :param x_bm: Describes which voxel/vertex is covered by each element in `x`
    :param y: (M, M) nibabel Cifti2Image
    :param y_bm: Describes which voxel/vertex is covered by each element in `y`
    :param xlog: makes the x-axis logarithmic
    :param ylog: makes the y-axis logarithmic
    :param diagonal: If True plots a diagonal straight line illustrating where both axes are equal
    """
    x_structures = {s[0]: s[1:] for s in x_bm.iter_structures()}
    y_structures = {s[0]: s[1:] for s in y_bm.iter_structures()}
    shared_structures = np.intersect1d(list(x_structures.keys()), list(y_structures.keys()))

    sz = shared_structures.size
    fig, axes = plt.subplots(sz, sz, figsize=(sz * 4, sz * 4))

    for idx1, name1 in enumerate(shared_structures):
        logger.info('processing row of plots with %s', name1)
        x_idx1, y_idx1 = find_common(x_structures[name1][1],
                                     y_structures[name1][1])
        for idx2, name2 in enumerate(shared_structures):
            if idx2 > idx1:
                fig.delaxes(axes[idx1, idx2])
                continue
            logger.debug('column %s', name2)
            x_idx2, y_idx2 = find_common(x_structures[name2][1],
                                         y_structures[name2][1])

            xmemmap_arr = x.dataobj[x_structures[name1][0], x_structures[name2][0]]
            xval = sub_sample(xmemmap_arr, x_idx1, x_idx2)

            ymemmap_arr = y.dataobj[y_structures[name1][0], y_structures[name2][0]]
            yval = sub_sample(ymemmap_arr, y_idx1, y_idx2)
            use = np.isfinite(xval) & np.isfinite(yval)
            if xlog:
                replacement = np.nanmin(xval[xval != 0])
                xval[xval == 0] = replacement
            if ylog:
                replacement = np.nanmin(yval[yval != 0])
                yval[yval == 0] = replacement

            if idx1 == idx2:
                axes[idx1, idx2].set_title(name2.split('_', maxsplit=2)[2])
            density_scatter(xval[use], yval[use], axes=axes[idx1, idx2],
                            s=2, alpha=0.3, xlog=xlog, ylog=ylog, diagonal=diagonal)
    return fig


def run_from_args(args):
    """
    Runs the script based on a Namespace containing the command line arguments
    """
    logger.info('starting %s', op.basename(__file__))
    x_bm, x_bm2 = cifti.get_axes(args.x_dconn)
    assert x_bm == x_bm2
    y_bm, y_bm2 = cifti.get_axes(args.y_dconn)
    assert y_bm == y_bm2
    run(
            x=nib.load(args.x_dconn), x_bm=x_bm,
            y=nib.load(args.y_dconn), y_bm=y_bm,
            xlog=args.xlog, ylog=args.ylog,
            diagonal=args.diagonal,
    ).savefig(args.output)
    logger.info('ending %s', op.basename(__file__))


def add_to_parser(parser):
    """
    Creates the parser of the command line arguments
    """
    parser.add_argument('x_dconn', help='dconn plotted on the x-axis')
    parser.add_argument('y_dconn', help='dconn plotted on the y-axis')
    parser.add_argument('--xlog', action='store_true', help='makes the x-axis logarithmic')
    parser.add_argument('--ylog', action='store_true', help='makes the y-axis logarithmic')
    parser.add_argument('-d', '--diagonal', action='store_true', help='draws a diagonal illustrating equality')
    parser.add_argument('output', help='image output name')
