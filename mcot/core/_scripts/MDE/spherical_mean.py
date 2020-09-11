#!/usr/bin/env python
"""Computes spherical mean signal"""
import nibabel as nib
import numpy as np
from scipy.special import sph_harm
from numpy import linalg
import os.path as op
from mcutils.utils.sidecar import AcquisitionParams
from mcutils.scripts.sidecar import index as sidecar_index
from fsl.data.image import removeExt
from numpy.random import shuffle
from loguru import logger
from mcutils.utils.spherical import cart2spherical


def read_bvecs(filename):
    if filename is None:
        return None
    bvecs = np.genfromtxt(filename).T
    if bvecs.shape[0] == 3 and bvecs.shape[1] != 3:
        return bvecs.T
    return bvecs


class SPHModel(object):
    """
    Fits the diffusion data with spherical harmonics
    """
    def __init__(self, data, bvecs, order=4):
        self.data = data
        n = np.concatenate([np.zeros(n * 2 + 1, dtype='i4') + n for n in range(0, order + 1, 2)])
        m = np.concatenate([np.arange(-n, n + 1, 1) for n in range(0, order + 1, 2)])
        _, phi, theta = cart2spherical(*bvecs.T)

        self.diff_mat = np.real(sph_harm(m[None, :], n[None, :], phi[:, None], theta[:, None])) * (2 * np.sqrt(np.pi))

    @property
    def ncomponents(self, ):
        return self.diff_mat.shape[1]

    @property
    def ngrad(self, ):
        return self.diff_mat.shape[0]

    def fit(self, select=None):
        if select is not None:
            data = self.data[..., select]
            diff_mat = self.diff_mat[select]
        else:
            data, diff_mat = self.data, self.diff_mat
        ngrad = diff_mat.shape[0]
        solution, _, _, _ = linalg.lstsq(diff_mat, data.reshape(-1, ngrad).T, rcond=None)
        return solution.T.reshape(data.shape[:-1] + (self.ncomponents, ))

    def leavekout(self, k):
        """
        Computes accuracy by leaving k out

        :param k: number of parameters to leave out
        :return: mean squared residual
        """
        select = np.arange(self.ngrad)
        shuffle(select)
        solution = self.fit(select[k:])
        predicted = np.tensordot(solution, self.diff_mat[select[:k], :], (-1, -1))
        res = np.mean((predicted - self.data[..., select[:k]]) ** 2, -1)
        return res


def fit_sph_harm(diffusion_data, bvecs, order=None):
    """
    Fits spherical harmonics to the diffusion data

    :param diffusion_data: (Nx, Ny, Nz, Ng) array of diffusion data
    :param bvecs: (Ng, 3) array of diffusion-encoding gradients
    :param order: maximum spherical harmonic order
    :return: (Nx, Ny, Nz, Nc) array with the components
    """
    if order is not None:
        logger.info('Computing spherical mean using spherical harmonics up to order {}'.format(order))
        return SPHModel(diffusion_data, bvecs, order).fit()

    ngrad = bvecs.shape[0]
    k_out = int(np.ceil(ngrad / 10))
    accuracy = np.inf
    for test_order in range(0, 12, 2):
        logger.debug('testing prediction accuracy of unseen data for spherical harmonics up to order %d', test_order)
        model = SPHModel(diffusion_data, bvecs, test_order)
        new_accuracy = np.mean([model.leavekout(k_out) for _ in range(4)])
        logger.debug('New accuracy: %f (compared with previous of %f', new_accuracy, accuracy)
        if new_accuracy + 1e-10 >= accuracy:
            logger.info('Maximum accuracy reached with order {}'.format(test_order - 2))
            return solution
        solution = model.fit()
        accuracy = new_accuracy
    if new_accuracy + 1e-10 >= accuracy:
        logger.warning('Maximum accuracy still not reached at order {}'.format(test_order - 2))


def run_single(diffusion_img: nib.Nifti1Image, method='sample', bvecs=None, order=None):
    """
    Computes the mean across a diffusion image

    :param diffusion_img: 4D image with diffusion data (shape=(Nx, Ny, Nz, Ng))
    :param method: method used to compute the spherical mean
    :param bvecs: (Ng, 3) array of gradient orientations
    :param order: maximum spherical harmonic order
    :return: image with the spherical mean
    """
    diffusion_data = np.asanyarray(diffusion_img.dataobj)
    if (diffusion_data.ndim == 4 and diffusion_data.shape[-1] == 1) or diffusion_data.ndim == 3:
        logger.info("Only single volume provided, this is returned as its own mean")
        mean_data = diffusion_data
        if mean_data.ndim == 4:
            mean_data = mean_data[..., 0]
    else:
        logger.info("Computing {} mean over {} volumes".format(method, diffusion_data.shape[-1]))
        mask = np.isfinite(diffusion_data).all(-1) & (diffusion_data != 0).any(-1)
        mean_data = np.zeros(diffusion_data.shape[:-1])
        if method == 'sample':
            mean_data[mask] = np.mean(diffusion_data[mask], -1)
        else:
            if bvecs is None:
                raise ValueError("Need to provide b-factor for method %s" % method)
            if method == 'shm':
                mean_data[mask] = fit_sph_harm(diffusion_data[mask], bvecs, order=order)[..., 0]
            else:
                raise ValueError(f"Unrecognized method: {method}")

    return nib.Nifti1Image(mean_data, affine=None, header=diffusion_img.header)


def run_multi(diffusion_img: nib.Nifti1Image, indices, method='sample', bvecs=None, order=None):
    """
    Computes the spherical mean for each b-value

    :param diffusion_img: image with the diffusion data
    :param indices: integer array with shell ids (zero-based)
    :param method: method used to compute the spherical mean
    :param bvecs: (Ng, 3) array of gradient orientation
    :param order: maximum spherical harmonic order
    :return: image with spherical mean for each shell
    """
    res = np.zeros(diffusion_img.shape[:3] + (max(indices) + 1, ))
    for idx_shell in range(max(indices) + 1):
        res[..., idx_shell] = run_single(
                nib.Nifti1Image(diffusion_img.get_data()[..., indices == idx_shell],
                                diffusion_img.affine),
                method=method,
                bvecs=None if bvecs is None else bvecs[indices == idx_shell, :],
                order=order
        ).get_data()
    return nib.Nifti1Image(res, affine=None, header=diffusion_img.header)


def run_from_args(args):
    if args.xps is not None:
        if args.bvals is not None or args.bvecs is not None:
            logger.warning("Sidecar file has been provided, so bvals and bvecs will be ignored")
        xps = AcquisitionParams.read(args.xps)
    elif args.bvals is not None:
        xps = AcquisitionParams(b=np.loadtxt(args.bvals).flatten())
    bvecs = read_bvecs(args.bvecs)
    if args.xps is None and args.bvals is None:
        run_single(
                diffusion_img=nib.load(args.data),
                method=args.method,
                bvecs=bvecs,
                order=args.shm_order,
        ).to_filename(args.out)
    else:
        indices = sidecar_index.get_indices(xps, args)
        mean_img = run_multi(
                diffusion_img=nib.load(args.data),
                method=args.method,
                bvecs=bvecs,
                order=args.shm_order,
                indices=indices,
        )
        mean_img.to_filename(args.out)
        directory, filename = op.split(args.out)
        if args.xps is None:
            shell_filename = op.join(directory, removeExt(filename) + '.bvals')
            np.savetxt(shell_filename, xps.groupby(indices)['b'], fmt='%d')
        else:
            xps_filename = op.join(directory, removeExt(filename) + '.json')
            xps.groupby(indices, drop=('theta', 'phi', 'psi')).write(xps_filename)


def add_to_parser(parser):
    parser.add_argument("data", help='aligned 4D diffusion data file (post-eddy)')
    parser.add_argument("out", help='output filename')
    parser.add_argument("method", choices=('sample', 'shm'),
                        help='method used to calculate the spherical mean:\n'
                             '1. sample: computes the sample mean\n'
                             '2. shm: returns the first component of a fit of spherical harmonics')
    parser.add_argument("-r", "--bvecs", help='text file with the gradient orientations')
    parser.add_argument("-b", "--bvals", help='text file with the diffusion weighting. '
                                              'If provided computes the spherical mean for each b-shell.')
    parser.add_argument("-x", "--xps", help='Acquisition parameter file (.json or .mat) containing the b-matrices')
    parser.add_argument('--shm_order', type=int, default=None,
                        help='maximum spherical harmonic order (should be multiple of 2). ' +
                             'Defaults to testing which order has the highest accuracy.')
    sidecar_index.add_index_params(parser)
