import numpy as np
from numpy import linalg, isfinite
import nibabel
import os


def average_orientation(orientations, weights=None, return_val=False):
    """Averages the provided orientations using PCA

    :param orientations: (..., N, M) array of N orientations that will be averaged in M-dimensional space
    :param weights: (..., N) array with the weighting of the orientations
    :param return_val: if True returns the eigenvalues as well as the mean hemisphere
    :return: (..., M) array with the mean hemisphere
    """
    if weights is None:
        weights = np.ones(orientations.shape[-2])
    weighted_orient = weights[..., None] * orientations
    cov = np.sum(weighted_orient[..., None, :] * weighted_orient[..., :, None], -3)
    val, vec = linalg.eigh(cov)
    indices = (tuple(np.mgrid[tuple(slice(None, sz) for sz in orientations.shape[:-2])]) +
               (slice(None), np.argmax(val, -1)))

    if return_val:
        return vec[indices], val
    return vec[indices]


def affine_mult(affine, coordinates):
    """
    Convert the given coordinates with the affine transformation.

    :param affine: (4 x 4) array defining the affine transformation
    :param coordinates: (..., 3) array of locations.
    """
    return np.dot(coordinates, affine[:3, :3].T) + affine[:3, -1]


def gcoord_mult(args):
    """
    Runs the gcoord_mult script

    :param args: arguments from the command line
    """
    img = nibabel.load(args.dyads)
    cardinal = img.get_data()
    coord = nibabel.load(args.coord[0]).get_data()
    for filename in args.coord[1:]:
        new_coord = nibabel.load(filename).get_data()
        replace = (~isfinite(coord)).any((-1, -2)) | (coord == 0).all((-1, -2))
        coord[replace] = new_coord[replace]

    gyral = (cardinal[..., None] * coord).sum(-2)
    gyral[~isfinite(gyral)] = 0.
    nibabel.Nifti1Image(gyral, img.affine).to_filename(args.output)


def gcoord_split(args):
    """
    Runs the gcoord_split script

    :param args: arguments from the command line
    """
    if not os.path.isfile(args.coord):
        raise IOError('Input coordinate NIFTI file %s not found' % args.coord)
    if args.base is None and args.radial is None and args.sulcal is None and args.gyral is None:
        raise ValueError("No output files set")

    img = nibabel.load(args.coord)
    coord = img.get_data()

    for idx, name in enumerate(('radial', 'sulcal', 'gyral')):
        if getattr(args, name, None) is not None:
            filename = getattr(args, name)
        elif args.base is not None:
            filename = args.base + '_%s.nii.gz' % name
        else:
            continue
        nibabel.Nifti1Image(coord[..., idx], img.affine).to_filename(filename)


def signed_tetrahedral_volume(p1, p2, p3):
    """
    Computes the signed tetrahedral volume

    Tetrahedron is formed by (p1, p2, p3, and origin)

    :param p1: (..., 3) array with positions of first point
    :param p2: (..., 3) array with positions of second point
    :param p3: (..., 3) array with positions of third point
    :return: (..., ) array with the volumes (negative if normal points towards the origin)
    """
    v321 = p3[..., 0] * p2[..., 1] * p1[..., 2]
    v231 = p2[..., 0] * p3[..., 1] * p1[..., 2]
    v312 = p3[..., 0] * p1[..., 1] * p2[..., 2]
    v132 = p1[..., 0] * p3[..., 1] * p2[..., 2]
    v213 = p2[..., 0] * p1[..., 1] * p3[..., 2]
    v123 = p1[..., 0] * p2[..., 1] * p3[..., 2]
    return (-v321 + v231 + v312 - v132 - v213 + v123) / 6.
