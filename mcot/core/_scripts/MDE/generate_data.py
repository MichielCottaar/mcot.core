#!/usr/bin/env python
"""
Module to generate fake mutiple diffusion encoding data under the Gaussian assumption
"""
from mcutils.utils.sidecar import AcquisitionParams
from . import fit_dispersion, micro_anisotropy
import numpy as np
from dataclasses import dataclass
from dipy.data import get_sphere
import nibabel as nib


@dataclass
class Tissue(object):
    volume_fraction: float
    d_axial: float
    d_radial: float

    @property
    def MD(self, ):
        return (self.d_axial + self.d_radial * 2) / 3

    @property
    def micro_anisotropy(self, ):
        return self.d_axial - self.d_radial

    def signal(self, bmat, bingham):
        bval = np.trace(bmat, axis1=1, axis2=2)
        anisotropy_signal = fit_dispersion.stick_dispersion(self.d_axial - self.d_radial, bmat, bingham)
        return self.volume_fraction * np.exp(-bval * self.d_radial) * anisotropy_signal


class Generator(object):
    def __init__(self, tissues, log_k1, log_k2):
        self.tissues = tissues
        self.log_k1 = log_k1
        self.log_k2 = log_k2

    @property
    def dispersion(self, ):
        return tuple(fit_dispersion.calc_dispersion(np.exp(lk)) for lk in (self.log_k1, self.log_k2))

    def bingham(self, ):
        return fit_dispersion.bingham_matrix(self.log_k1, self.log_k2, 0, 0, 0)

    def __call__(self, bmat, noise=0):
        bmat = np.asarray(bmat)
        assert np.sum([t.volume_fraction for t in self.tissues]) == 1
        S = np.zeros(bmat.shape[0])
        bingham = self.bingham()
        for t in self.tissues:
            S += t.signal(bmat, bingham)
        S += np.random.randn(bmat.shape[0]) * noise
        return S

    def micro_anisotropy(self, b_value, b_delta=0):
        total = np.sum([t.volume_fraction * np.exp(-t.MD * b_value) * micro_anisotropy.mean_signal(
            t.micro_anisotropy * b_value, b_delta) * t.micro_anisotropy for t in self.tissues])
        norm = np.sum([t.volume_fraction * np.exp(-t.MD * b_value) * micro_anisotropy.mean_signal(
            t.micro_anisotropy * b_value, b_delta) for t in self.tissues])
        return total / norm


def get_bmat(LTE=False, PTE=False, STE=0, ):
    """
    Which b-matrices to include in acquisition

    :param LTE: if True include 100 LTE scans
    :param PTE: if True include 100 PTE scans
    :param STE: Number of STE scans to include
    :return: concatenated b-matrices
    """
    sphere = get_sphere('repulsion100')
    bvecs = sphere.vertices
    LTE_mat = (bvecs[:, None, :] * bvecs[:, :, None])
    PTE_mat = (np.eye(3) - LTE_mat) / 2
    STE_mat = np.array([np.eye(3) / 3] * STE)
    matrices = [STE_mat]
    if PTE:
        matrices.insert(0, PTE_mat)
    if LTE:
        matrices.insert(0, LTE_mat)
    return np.concatenate(matrices, 0)


def run(multi_tissue: Generator, bval=np.arange(1, 9, 1), PTE=False):
    """
    Generates signal from the signal generator

    :param multi_tissue: super-position of multiple tissue types
    :param bval: which b-values to include
    :param PTE: set to True to include PTE data
    :return: nibabel and sidecar file describing the dataset
    """
    bmat_unscaled = get_bmat(LTE=True, PTE=PTE, STE=100)
    bmat = (bmat_unscaled[None, ...] * bval[:, None, None, None]).reshape((-1, 3, 3))
    noise = 5e-4 * bmat_unscaled.shape[0]
    signal = np.stack([multi_tissue(bmat, noise) for _ in range(300)], 0)
    as_nifti = nib.Nifti1Image(signal[:, None, None, :], np.eye(4))
    sidecar = AcquisitionParams(bmat=bmat)
    return as_nifti, sidecar


def run_from_args(args):
    """
    Runs the script based on a Namespace containing the command line arguments
    """
    nifti, sidecar = run(
        Generator([Tissue(*tissue_args) for tissue_args in args.tissue],
                  log_k1=np.log(args.k1), log_k2=np.log(args.k2))
    )
    nifti.to_filename(args.output + '.nii.gz')
    sidecar.to_json(args.output + '.json')


def add_to_parser(parser):
    """
    Creates the parser of the command line arguments
    """
    parser.add_argument('-t', '--tissue', type=float, nargs=3, action='append', required=True,
                        help='Adding a different tissue type defined by three numbers: ' +
                             '<volume fraction> <parallel diffusivity> <perpendicular diffusivity>')
    parser.add_argument('-k1', help='dispersion along minor axis (default: exp(2))', default=np.exp(2), type=float)
    parser.add_argument('-k2', help='dispersion along major axis (default: exp(3))', default=np.exp(3), type=float)
    parser.add_argument('-P', '--PTE', help='If set include 100 PTE volumes', action='store_true')
    parser.add_argument('-S', '--STE', type=int, help='number of STE scans to include (default: 100)', default=100)
    parser.add_argument('-o', '--output', required=True,
                        help='basename of the output. <output>.json and <output>.nii.gz will be produced')
