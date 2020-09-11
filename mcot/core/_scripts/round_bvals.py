#!/usr/bin/env python
"""Rounds the input b-values

Each b-value is considered to be in the same shell as each other b-value within `b_delta`.
For each shell the median b-value of all volumes is used as in the output b-value file.

Note that even volumes, which have a difference in b-value of more than `b_delta`
can still be in the same shell if there are intermediate b-values.
For example, for a `b_delta` of 100, the b-values of [0, 80, 160] are all within one shells,
but the b-values of [0, 160] are not.
"""
import numpy as np
from loguru import logger


def run(input_bvals: np.ndarray, delta_bval: int = 100) -> np.ndarray:
    """
    Rounds an array of b-values

    All b-values within `delta_bval` will be set to their median value.

    :param input_bvals: (N, ) array of input b-values
    :param delta_bval: offset of b-value to still be considered a single shell
    :return: (N, ) array of rounded b-values
    """
    assigned = np.zeros(input_bvals.size, dtype='bool')

    in_shell = abs(input_bvals[None, :] - input_bvals[:, None]) < delta_bval

    new_bvals = input_bvals.copy()

    while not assigned.all():
        idx = np.where(~assigned)[0][0]
        use = in_shell[:, idx]
        nuse = 1
        while use.sum() != nuse:
            nuse = use.sum()
            use = in_shell[:, use].any(-1)
        if assigned[use].any():
            raise ValueError("Assigning the same volume to two shells")
        median_bval = np.median(input_bvals[use])
        logger.debug('found b-shell with b-value of %.2f', median_bval)
        new_bvals[use] = median_bval

        assigned[use] = True

    return new_bvals


def run_from_args(args):
    rounded = run(
            input_bvals=np.genfromtxt(args.input_bvals, dtype='i4'),
            delta_bval=args.delta_bval
    )
    logger.info('b-shells found: %s', np.unique(rounded))
    np.savetxt(args.rounded_bvals, rounded, fmt='%d')
    return rounded


def add_to_parser(parser):
    parser.add_argument('input_bvals', help='text file with input b-values')
    parser.add_argument('rounded_bvals', help='text file with output b-values')
    parser.add_argument('delta_bval', type=int, default=100, help='offset in b-value to still be considered the same (default: 100)')
