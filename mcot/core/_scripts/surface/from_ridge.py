#!/usr/bin/env python
"""
Assigns each vertex the value from the closest ridge (BUGGY)
"""
from mcot.core.surface import CorticalMesh, track
from mcot.core import write_gifti
import nibabel as nib


def run(surface: CorticalMesh, sulcal_depth, to_extract, minimum=False):
    """
    Extracts values from the ridge
    """
    gradient = surface.gradient(sulcal_depth)
    if minimum:
        gradient *= -1
    return track.extract_ridge_values(surface, gradient.T, to_extract)


def run_from_args(args):
    """
    Runs the script based on a Namespace containing the command line arguments
    """
    surface = CorticalMesh.read(args.surface)
    sulcal_depth = nib.load(args.sulcal_depth).darrays[0].data
    to_extract = nib.load(args.extract).darrays[0].data
    values = run(
            surface,
            sulcal_depth,
            to_extract,
            args.minimum
    )
    write_gifti(args.out, [values], surface.anatomy)


def add_to_parser(parser=None):
    """
    Creates the parser of the command line arguments
    """
    parser.add_argument('sulcal_depth', help='GIFTI file with the sulcal depth')
    parser.add_argument('extract', help='GIFTI file with the values to extract from the ridge')
    parser.add_argument('out', help='GIFTI file with the extracted values')
    parser.add_argument('-m', '--minimum', action='store_true',
                        help='extracts values from the sulcal fundus rather than gyral crown')
