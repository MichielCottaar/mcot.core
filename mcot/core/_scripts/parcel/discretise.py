#!/usr/bin/env python
"""
Discretizes a continuous variable
"""
from loguru import logger
from mcot.core import scripts
import numpy as np
import colorcet as cc
from mcot.core.cifti import combine


def run_array(arr, nbins, bins=None, weight=None, include_zeros=False):
    """
    Returns a discretised version of the input array

    :param arr: nibabel input image
    :param nbins: number of bins to extract
    :param bins: one of the following

        - None: use weight to set the bins
        - 'number': each parcel will have the same number of elements
        - 'regular': split the range from min to max in the input into equal bins
        - 1D array: explicit boundaries

    :param weight: selects the bins so each parcel has the same sum in this image (only used if bins is None)
    :param include_zeros: if True include zeros in the analysis
    :return: tuple with:

        - array with the parcels (zero where the original array was zero)
        - (nbins + 1, ...) array with the applied boundaries
    """
    if weight is not None and arr.shape[:weight.ndim] != weight.shape:
        raise ValueError("Shape of weight image does not match input image")

    res = np.zeros(arr.shape, dtype='i4')

    all_bins = np.zeros((nbins + 1, ) + (() if weight is None else arr.shape[weight.ndim:]))

    for idx in np.ndindex(*(() if weight is None else arr.shape[weight.ndim:])):
        sub_arr = arr[(Ellipsis, ) + idx]
        mask = slice(None) if include_zeros else sub_arr != 0

        if weight is not None:
            if (np.array(idx) == 0).all():
                logger.info('Using weight file to set bins')
            idx_sorted = np.argsort(sub_arr[mask])
            values = np.append(0, np.cumsum(weight[mask][idx_sorted]))
            edges = np.floor(np.interp(
                    np.linspace(0, values[-1], nbins + 1)[1:-1],
                    values,
                    np.arange(values.size),
            )).astype('int')
            use_bins_mid = sub_arr[mask][idx_sorted][edges]
            use_bins = np.append(-np.inf, np.append(use_bins_mid, np.inf))
        elif bins == 'regular':
            if (np.array(idx) == 0).all():
                logger.info('Using regularly spaced bins')
            use_bins = np.linspace(sub_arr.min(), sub_arr.max(), nbins + 1)
        elif bins == 'number':
            if (np.array(idx) == 0).all():
                logger.info('Setting bins to have identical number of elements in parcels')
            use_bins = np.sort(sub_arr[mask])[np.around(np.linspace(0, sub_arr[mask].size - 1, nbins + 1)).astype('i4')]
        else:
            use_bins = np.array(bins)
        assert use_bins.size == nbins + 1

        logger.debug(f'Bins for {idx}: {use_bins}')

        all_bins[(Ellipsis, ) + idx] = use_bins

        use_bins[-1] += 1e-8
        res[(Ellipsis, ) + idx][mask] = np.digitize(sub_arr[mask], use_bins)
    return res, all_bins


def run_from_args(args):
    """
    Runs the script based on a Namespace containing the command line arguments
    """
    arr, axes = args.input
    bins, weight = None, None
    if args.equal_weight:
        weight, axes_weight = args.equal_weight
        bm, (idx_arr, idx_weight) = combine([axes[-1], axes_weight[-1]])
        axes = axes[:-1] + (bm, )
        arr = arr[..., idx_arr]
        weight = weight[..., idx_weight]
    elif args.equal_number:
        bins = 'number'
    elif args.equal_bin:
        bins = 'regular'
    elif args.set_bin:
        if len(args.set_bin) == args.nbins - 1:
            args.set_bin = np.append(-np.inf, np.append(args.set_bin, np.inf))
        assert len(args.set_bin) == args.nbins + 1
        bins = args.set_bin
    else:
        raise ValueError("No binning method selected")

    res, used_bins = run_array(
        arr, args.nbins, bins=bins, weight=weight, include_zeros=args.include_zeros
    )
    labels = [{int(idx): (f'{start:.2f} to {end:.2f}', c) for idx, start, end, c in zip(
            range(1, 100), used_bins[:-1], used_bins[1:], cc.glasbey)}]
    new_axes = (axes[0].to_label(labels), ) + axes[1:]
    args.output((res, new_axes))


def add_to_parser(parser):
    """
    Creates the parser of the command line arguments
    """
    parser.add_argument('input', type=scripts.greyordinate_in,
                        help='input NIFTI/GIFTI/CIFTI file')
    parser.add_argument('output', type=scripts.output,
                        help='output NIFTI/GIFTI/CIFTI files')
    parser.add_argument('nbins', type=int, help='number of bins')
    parser.add_argument('-i0', '--include_zeros', help='Include zeros in the analysis (always true for CIFTI)')
    grp = parser.add_mutually_exclusive_group()
    grp.add_argument('--equal_weight', type=scripts.greyordinate_in,
                     help='Each bin will contain the same total weight of given file')
    grp.add_argument('--equal_number', action='store_true',
                     help='Each bin will contain the same number of elements')
    grp.add_argument('--equal_bin', action='store_true',
                     help='Each bin will have the same size (from min to max value)')
    grp.add_argument('--set_bin', nargs='*', type=float,
                     help='Manually sets the edges of the bins as a sequence of numbers')
