#!/usr/bin/env python
"""Extract radial, sulcal, and gyral orientations from gyral coordinate NIFTI file"""


def main():
    import argparse

    parser = argparse.ArgumentParser("Extract radial, sulcal, and gyral dyads from a coord NIFTI file")
    parser.add_argument('coord', help='name of the coord file')
    parser.add_argument('-b', '--base', help='Basename of the output files')
    parser.add_argument('-r', '--radial', help='Filename for the radial output (overrides the --base option)')
    parser.add_argument('-s', '--sulcal', help='Filename for the sulcal output (overrides the --base option)')
    parser.add_argument('-g', '--gyral', help='Filename for the gyral output (overrides the --base option)')

    args = parser.parse_args()
    from mcot.core.surface import utils
    utils.gcoord_split(args)
