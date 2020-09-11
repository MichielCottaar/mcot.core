#!/usr/bin/env python
"""
Combines two parcellations into a single parcellation by multiplying them
"""
import argparse
from mcot.core import scripts
from copy import deepcopy
import numpy as np
from mcot.core.cifti import combine


def run(arr1, label1, arr2, label2):
    """
    Combines the two parcellations into one by multiplying

    :param arr1: int array with first parcellation
    :param label1: dict from index to (name, RGBA colour) for first parcellation
    :param arr2: int array with second parcellation
    :param label2: dict from index to (name, RGBA colour) for second parcellation
    :return: tuple with two elements:

        - int array with final parcellation
        - dict from index to (name, RBBA colour)
    """
    arr2_values = np.array(list(label2.keys()), dtype='i4')
    frange2 = max(arr2_values.max() - arr2_values.min(), 1)
    factor = 10 ** int(np.ceil(np.log10(frange2)))
    combined = arr1 * factor + arr2 - arr2_values.min()

    table = {}
    for index1, (name1, col1) in label1.items():
        for index2, (name2, col2) in label2.items():
            idx = index1 * factor + index2 - arr2_values.min()
            if idx in combined:
                table[idx] = (
                    f'{name1} & {name2}',
                    tuple((np.array(col1) + np.array(col2)) / 2)
                )
    return combined, table


def run_from_args(args):
    """
    Runs the script based on a Namespace containing the command line arguments
    """
    arr1, axes1 = args.first_parcellation
    arr2, axes2 = args.second_parcellation

    bm, (idx1, idx2) = combine((axes1[-1], axes2[-1]))
    arr1 = arr1[..., idx1]
    arr2 = arr2[..., idx2]
    axes1 = axes1[:-1] + (bm, )
    axes2 = axes2[:-1] + (bm, )

    if arr1.shape != arr2.shape:
        raise ValueError("Input parcellations should have the same shape")

    labels = []
    res = []
    for a1, x1, a2, x2 in zip(arr1, axes1[0].label, arr2, axes2[0].label):
        combined_arr, combined_table = run(a1, x1, a2, x2)
        res.append(combined_arr)
        labels.append(combined_table)

    label_axis = deepcopy(axes1[0])
    label_axis.label = labels
    args.output((
            np.stack(res, 0),
            (label_axis, ) + axes1[1:]
    ))


def add_to_parser(parser):
    """
    Creates the parser of the command line arguments
    """
    if parser is None:
        parser = __doc__
    if isinstance(parser, str):
        parser = argparse.ArgumentParser(parser)
    parser.add_argument("first_parcellation", type=scripts.greyordinate_in,
                        help='GIFTI or CIFTI parcellation file')
    parser.add_argument("second_parcellation", type=scripts.greyordinate_in,
                        help='GIFTI or CIFTI parcellation file')
    parser.add_argument("output", type=scripts.output,
                        help='output GIFTI or CIFTI parcellation file')
