#!/usr/bin/env python
"""Scatter plot of gifti files"""
import matplotlib.pyplot as plt
from mcutils.plot import density_scatter
import nibabel as nib


def run(x, y, xlog=False, ylog=False, diagonal=False):
    """
    Creates a plot of

    :param x: (N, ) array with values to be plotted on the x-axis
    :param y: (M, ) array with values to be plotted on the y-axis
    :param xlog: makes the x-axis logarithmic
    :param ylog: makes the y-axis logarithmic
    :param diagonal: If True plots a diagonal straight line illustrating where both axes are equal
    """
    fig = plt.figure()
    density_scatter(x, y, s=2, alpha=0.3, xlog=xlog, ylog=ylog, diagonal=diagonal,
                    axes=fig.add_subplot(111))
    return fig


def run_from_args(args):
    """
    Runs the script based on a Namespace containing the command line arguments
    """
    x = nib.load(args.x_gifti).darrays[0].data
    y = nib.load(args.y_gifti).darrays[0].data
    run(
            x=x,
            y=y,
            xlog=args.xlog, ylog=args.ylog,
            diagonal=args.diagonal,
    ).savefig(args.output)


def add_to_parser(parser):
    """
    Creates the parser of the command line arguments
    """
    parser.add_argument('x_gifti', help='gifti plotted on the x-axis')
    parser.add_argument('y_gifti', help='gifti plotted on the y-axis')
    parser.add_argument('--xlog', action='store_true', help='makes the x-axis logarithmic')
    parser.add_argument('--ylog', action='store_true', help='makes the y-axis logarithmic')
    parser.add_argument('-d', '--diagonal', action='store_true', help='draws a diagonal illustrating equality')
    parser.add_argument('output', help='image output name')
