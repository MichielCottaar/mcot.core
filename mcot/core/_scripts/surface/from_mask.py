#!/usr/bin/env python
"""
Creates a surface covering the mask

Uses the scikit-image marching cubes algorithm

The resulting surface will follow the edge of the mask exactly and hence look very jagged.
You might want to run `mc_script surface smooth` afterwards to get a nicer looking surface.
"""
import argparse
from skimage import measure
from mcot.core.surface import Mesh2D
from numpy import linalg
import nibabel as nib
import numpy as np


def run(img):
    """
    Generates the mesh based on the tumour mask

    uses the marching cube algorithm from scikit-image

    :param img: mask that is non-zero in the lesion
    :return: new surface
    """
    if len(img.shape) == 4:
        mask = img.dataobj[:, :, :, 0] > 0
    else:
        mask = np.asarray(img.dataobj) > 0
    verts, faces, _, _ = measure.marching_cubes_lewiner(mask, 0.5, allow_degenerate=False)
    surf = Mesh2D(verts.T, faces.T)
    return surf.apply_affine(linalg.inv(img.affine))


def run_from_args(args):
    """
    Runs the script based on a Namespace containing the command line arguments
    """
    surf = run(
            nib.load(args.mask)
    )
    surf.write(args.surface)


def add_to_parser(parser=None):
    """
    Creates the parser of the command line arguments
    """
    if parser is None:
        parser = __doc__
    if isinstance(parser, str):
        parser = argparse.ArgumentParser(parser)
    parser.add_argument('mask', help='NIFTI file with positive values in the mask')
    parser.add_argument('surface', help='output surface covering the mask')
