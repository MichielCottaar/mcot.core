#!/usr/bin/env python
"""Converts dyads into gyral coordinate system"""


def main():
    import argparse

    parser = argparse.ArgumentParser("Converts the dyads from the cardinal to the gyral coordinate system")
    parser.add_argument('-d', '--dyads', required=True, help='NIFTI file with the estimated fibre orientations')
    parser.add_argument('-c', '--coord', required=True, nargs='+', help='NIFTI file with the gyral coordinates produced by gcoord_gen')
    parser.add_argument('-o', '--output', required=True, help='Output NIFTI files with the fibre orientations in gyral coordinates')

    args = parser.parse_args()
    from mcot.core.surface import utils
    utils.gcoord_mult(args)
