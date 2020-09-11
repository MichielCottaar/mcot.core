#!/usr/bin/env python
"""Merges multiple sidecars into one"""
import sys
from typing import List, Tuple
from mcutils.utils.sidecar import AcquisitionParams, concat


def run(*parts):
    """
    Access to the script for other python programs.

    :param parts: objects to merge. Each object is a tuple with:

        1. identifier, one of SIDE, LTE, PTE, or STE
        2. list of 1 or 2 filenames:

            - .json or .mat for SIDE
            - bvals and bvecs for LTE and PTE
            - bvals for STE
    """
    partial_sidecar = []
    for identifier, filenames in parts:
        if identifier == 'SIDE':
            partial_sidecar.append(AcquisitionParams.read(filenames[0]))
        else:
            partial_sidecar.append(AcquisitionParams.read_bvals_bvecs(*filenames, modality=identifier))
    return concat(*partial_sidecar)


def usage():
    return """
Usage: mc_script sidecar_merge <output> <input1 input2 .....>

    <output> : .json or .mat output file for the XPS output
    <input#> : input bvals/bvecs or XPS file. Each input can be one of the following:
        - <-X/--SIDE> <sidecar> : sidecar filename is a .json of .mat input file with acquisition parameters
        - <-L/--LTE> <bvals> <bvecs> : bvals & bvecs for linear tensor encoding
        - <-P/--PTE> <bvals> <bvecs> : bvals & bvecs for planar tensor encoding
        - <-S/--STE> <bvals> <bvecs> : bvals & bvecs for spherical tensor encoding
"""


def parse_args(args=None) -> Tuple[str, List[Tuple[str, List[str]]]]:
    """
    Parses the command-line arguments

    :param args: command line arguments (defaults to sys.argv[1:])
    :return: output filename and list of objects to merge. Each object is a tuple with:

        1. indentifier, one of XPS, LTE, PTE, or STE
        2. list of 1 or 2 filenames:

            - .json or .mat for XPS
            - bvals and bvecs for LTE and PTE
            - bvals for STE
    """
    if args is None:
        args = sys.argv[1:]
    if len(args) == 0 or args[0] in ('-h', '--help'):
        print(usage())
        sys.exit(0)
    output = args[0]
    if output.split('.')[-1].lower() not in ('json', 'mat'):
        print(usage())
        raise ValueError(f"Output file {output} does not have .json or .mat extension")

    options = {
        '-X': ('SIDE', 1),
        '--SIDE': ('SIDE', 1),
        '-L': ('LTE', 2),
        '--LTE': ('LTE', 2),
        '-P': ('PTE', 2),
        '--PTE': ('PTE', 2),
        '-S': ('STE', 1),
        '--STE': ('STE', 1),
    }

    split_args = []
    for element in args[1:]:
        if element in options:
            split_args.append([element])
        else:
            if len(split_args) == 0:
                print(usage())
                merged='/'.join(options.keys())
                raise ValueError(f'Expected one of {merged} after output filename, not {element}')
            split_args[-1].append(element)

    parts = []
    for args_set in split_args:
        identifier, n_fn_expected = options[args_set[0]]
        n_fn = len(args_set) - 1
        if n_fn != n_fn_expected:
            print(usage())
            raise ValueError(f'Expected {n_fn_expected} arguments for {identifier}, but got {n_fn}: {args_set[1:]}')
        parts.append((identifier, args_set[1:]))
    return output, parts


def main():
    """
    Runs the script from the command line
    """
    out_fn, parts = parse_args()

    new_sidecar = run(*parts)
    new_sidecar.write(out_fn)
