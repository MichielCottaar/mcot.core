#!/usr/bin/env python
"""Indexes the volumes into shells based on the sidecar"""
import argparse
from loguru import logger
from mcutils.utils.sidecar import AcquisitionParams
import numpy as np

possible_parameters = [
    ('b', 'b-value (in s/mm^2)'),
    ('b_delta', 'anisotropy of b-tensor'),
    ('b_eta', 'asymmetry of b-tensor'),
    ('b_symm', 'b-value along symmetry axis'),
    ('b_perp', 'b-value perpendicular to the symmetry axis'),
    ('te', 'echo time'),
    ('tr', 'repetition time'),
    ('ti', 'inversion time'),
    ('tm', 'mixing time'),
]


def get_indices(sidecar: AcquisitionParams, group_args):
    """
    Assigns each observation to a shell

    :param sidecar: MDE XPS structure describing the data
    :param group_args: command line arguments from scripts.sidecar_index.add_index_params
    :return: integer array with the shell index
    """
    constraints = {
        var: getattr(group_args, var) for var, _ in possible_parameters if getattr(group_args, var, None) is not None
    }
    for idx, (var, value) in enumerate(constraints.items()):
        logger.debug(f'Constraint #{idx + 1}: Delta {var} <= {value}')
    indices = sidecar.get_index(**constraints, sort_by=group_args.sort)
    logger.info(f'{max(indices) + 1} shells found')
    logger.debug(f'Shell indices (0-based): {list(indices)}')
    for idx, var in enumerate(constraints.keys()):
        values = sidecar[var]
        logger.debug(f'Mean {var} per shell: {[np.mean(values[idx == indices]) for idx in range(max(indices) + 1)]}')
    return indices


def run_from_args(args):
    """
    Runs the script based on a Namespace containing the command line arguments
    """
    sidecar = AcquisitionParams.read(args.input)
    indices = get_indices(sidecar, group_args=args)
    if args.a_ind:
        sidecar.a_ind = indices
    elif args.s_ind:
        sidecar.s_ind = indices
    else:
        raise ValueError("either --a_ind or --s_ind should be set")

    out_fn = args.input if args.output is None else args.output
    sidecar.write(out_fn)


def add_index_params(parser: argparse.ArgumentParser, exclude=(), as_group=True):
    """
    Adds the arguments needed to define indexing of an XPS file on the command line

    :param parser: argument parser
    :param exclude: exclude certain parameters from the accepted parameter list
    :param as_group: add the new parameters in their own group
    """
    if as_group:
        group = parser.add_argument_group('indexing',
                                          'Parameters used to group the data into different shells.\n' +
                                          'Any observations that match for all parameters within the provided ' +
                                          'offsets will be in the same shell.')
    else:
        group = parser

    for var, description in possible_parameters:
        if var not in exclude:
            group.add_argument(f'--{var}', type=float, help='maximum offset for ' + description)

    parser.add_argument('--sort', choices=[p[0] for p in possible_parameters],
                        help='sort the output by the selected parameter')


def add_to_parser(parser):
    """
    Creates the parser of the command line arguments
    """
    parser.add_argument("input", help='Sidecar input file with acquisition parameters (.mat or .json)')
    parser.add_argument("output", nargs='?', help='Sidecar output file (.mat or .json). Defaults to the input file')
    target_group = parser.add_mutually_exclusive_group(required=True)
    target_group.add_argument('-a', '--a_ind', help='add shell index to the a_ind index', action='store_true')
    target_group.add_argument('-s', '--s_ind', help='add shell index to the s_ind index', action='store_true')
    add_index_params(parser)
