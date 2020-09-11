#!/usr/bin/env python
"""Scatter plot of dense scalar files"""
import cifti
import matplotlib.pyplot as plt
from mcutils.plot import density_scatter, default_grid
import numpy as np


def find_common(bm1: cifti.BrainModel, bm2: cifti.BrainModel):
    """
    Returns the indices into bm1 and bm2 that produce the common elements

    :param bm1: list of voxels and vertices
    :param bm2: list of voxels and vertices
    :return: tuple of two integer arrays or slices
    """
    if bm1 == bm2:
        return slice(None), slice(None)
    max_voxel = np.max((bm1.voxel.max(0), bm2.voxel.max(0)), 0)
    adjustment = np.append(1, np.cumprod(max_voxel))
    unique_index1 = (adjustment[:3] * bm1.voxel).sum(-1) + adjustment[-1] * bm1.vertex
    unique_index2 = (adjustment[:3] * bm2.voxel).sum(-1) + adjustment[-1] * bm2.vertex
    target_index = np.intersect1d(unique_index1, unique_index2)
    idx1 = np.argmin(abs(unique_index1[None, :] - target_index[:, None]), 1)
    idx2 = np.argmin(abs(unique_index2[None, :] - target_index[:, None]), 1)
    return idx1, idx2


def run(x, x_bm: cifti.BrainModel, y, y_bm: cifti.BrainModel, xlog=False, ylog=False, diagonal=False):
    """
    Creates a plot of

    :param x: (N, ) array with values to be plotted on the x-axis
    :param x_bm: Describes which voxel/vertex is covered by each element in `x`
    :param y: (M, ) array with values to be plotted on the y-axis
    :param y_bm: Describes which voxel/vertex is covered by each element in `y`
    :param xlog: makes the x-axis logarithmic
    :param ylog: makes the y-axis logarithmic
    :param diagonal: If True plots a diagonal straight line illustrating where both axes are equal
    """
    x_structures = {s[0]: s[1:] for s in x_bm.iter_structures()}
    y_structures = {s[0]: s[1:] for s in y_bm.iter_structures()}
    shared_structures = np.intersect1d(list(x_structures.keys()), list(y_structures.keys()))

    gs = default_grid(shared_structures.size)
    fig = plt.figure(figsize=(gs._ncols * 4, gs._nrows * 4))

    for idx, name in enumerate(shared_structures):
        print(name.split('_', maxsplit=2)[-1])

        idx1, idx2 = find_common(x_structures[name][1],
                                 y_structures[name][1])
        xval = x[x_structures[name][0]][idx1]
        yval = y[y_structures[name][0]][idx2]
        print("N =", xval.size)
        axes = fig.add_subplot(gs[idx])
        density_scatter(xval, yval, axes=axes, s=2, alpha=0.3, xlog=xlog, ylog=ylog, diagonal=diagonal)
        axes.set_title(name.split('_', maxsplit=2)[-1])
    return fig


def read_dscalar(filename, scalar_name):
    """
    Reads a row from a CIFTI dscalar file

    :param filename: CIFTI filename
    :param scalar_name: name of the row (or integer)
    :return: 1D array and the brainmodel axis
    """
    full_arr, (scalar, bm) = cifti.read(filename)
    if scalar_name in scalar.name:
        idx = list(scalar.name).index(scalar_name)
    else:
        idx = int(scalar_name)
    return full_arr[idx], bm


def run_from_args(args):
    """
    Runs the script based on a Namespace containing the command line arguments
    """
    x, x_bm = read_dscalar(*args.x_dscalar)
    y, y_bm = read_dscalar(*args.y_dscalar)
    run(
            x=x, x_bm=x_bm,
            y=y, y_bm=y_bm,
            xlog=args.xlog, ylog=args.ylog,
            diagonal=args.diagonal,
    ).savefig(args.output)


def add_to_parser(parser):
    """
    Creates the parser of the command line arguments
    """
    parser.add_argument('x_dscalar', nargs=2, help='dscalar plotted on the x-axis')
    parser.add_argument('y_dscalar', nargs=2, help='dscalar plotted on the y-axis')
    parser.add_argument('--xlog', action='store_true', help='makes the x-axis logarithmic')
    parser.add_argument('--ylog', action='store_true', help='makes the y-axis logarithmic')
    parser.add_argument('-d', '--diagonal', action='store_true', help='draws a diagonal illustrating equality')
    parser.add_argument('output', help='image output name')
