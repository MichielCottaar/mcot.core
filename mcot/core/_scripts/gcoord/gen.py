#!/usr/bin/env python
"""Generates the gyral coordinates in a volume"""


def main():
    import argparse
    parser = argparse.ArgumentParser("Computes the radial and tangential hemisphere for every point in space", formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("algorithm", choices=['line', 'interp', 'closest'], help="""Select which algorithm to use to compute the radial/tangential orientations. There are three options:
1. 'line': strongly recommended with -pd -1: The surface normal (& sulcal depth gradient) are linearly interpolated along N randomly oriented lines and averaged
2. 'interp': best with -pd -3: All surface normals (& sulcal depth depth gradients) are averaged with the weighting depending on the distance from the voxel of interest
3. 'closest': Use the normal (& sulcal depth gradient) from the closest vertex""")
    parser.add_argument("output", help='Output NIFTI file where the gyral coordinate system will be stored')
    parser.add_argument("-m", "--mask", required=True, help="computes the gyral coordinates for any non-zero voxels")
    parser.add_argument("-w", "--white", required=True,
                        help="GIFTI file defining the white/gray matter boundary (ending with .surf.gii)")
    parser.add_argument("-p", "--pial", required=True,
                        help="GIFTI file defining the pial surface (ending with .surf.gii)")
    parser.add_argument("-sd", "--sulcal_depth",
                        help="GIFTI file defining the sulcal depth (ending with .shape.gii or .func.gii)")
    parser.add_argument("-N", "--norient", default=300, type=int,
                        help="Number of random line orientations to average over (only if algorithm='line'); " +
                             "default: 300")
    parser.add_argument("--outside_pial", action='store_true', help='Include voxels outside of the pial surface (does not work for the line algorithm)')
    parser.add_argument("-pd", "--power_dist", default=-1., type=float,
                        help="weights in averaging = d ** `power_dist`, where weights is the line length " +
                             "(if algorithm='line') or the distance between vertex and point of interest " +
                             "(if algorithm='interp'); default: -1")
    parser.add_argument("-t", "--thickness",
                        help="Optional NIFTI output file containing the length of the shortest line connecting " +
                             "the cortical surfaces through the voxel and the voxel location along this line " +
                             "(0 for edge, 0.5 for gyral centre). Only valid for the `line` algorithm")
    parser.add_argument("--flip-inpr", action='store_true', help=argparse.SUPPRESS)
    parser.add_argument("--zval", default=None, type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.outside_pial and args.algorithm == 'line':
        raise ValueError("Algorithm 'line' can not include voxels outside of the pial surface.")
    from mcot.core.surface import orientation
    orientation.run_from_args(args)
