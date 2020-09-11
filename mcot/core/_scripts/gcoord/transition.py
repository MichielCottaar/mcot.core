#!/usr/bin/env python
"""Extract fibre orientation transition boundary"""


def main():
    import argparse
    parser = argparse.ArgumentParser('Fits a sigmoidal model to the radial index across the white/gray matter boundary')
    parser.add_argument('coord', help='NIFTI file with radial & tangential orientations from gcood_gen')
    parser.add_argument('dyad', help='NIFTI file with DTI or other hemisphere')
    parser.add_argument('white', help='GIFTI file with white/gray matter boundary')
    parser.add_argument('output', help='CIFTI dscalar file with the output parameters')
    parser.add_argument('-m', '--mask', help='GIFTI file with non-zero values for the vertices to use')
    parser.add_argument('-w', '--weight', type=float, default=1., help='Weighting of the smoothing term (defaults: 1., which works well for HCP data)')
    parser.add_argument('-md', '--min-dist', type=float, default=-3, help='Minimal voxel distance from the white/gray matter boundary (default: -3)')
    parser.add_argument('--watson', action='store_true', help='Assumes Watson rather than Normal distribution for radial index')
    parser.add_argument('--idx-vertex', help='Optional NIFTI file with the index of the vertex that every voxel should be assigned to')
    parser.add_argument('--distance', help='Optional NIFTI file with the distance from the WM/GM boundary')
    parser.add_argument('-ri', '--radial-index', help='Optional NIFTI file output containing the radial index of the best-fit model')
    args = parser.parse_args()
    from mcot.core.surface import radial_transition
    radial_transition.run_from_args(args)
