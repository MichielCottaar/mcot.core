#!/usr/bin/env python
"""Extracts micro-anisotropy from mean signal"""
import numpy as np
import nibabel as nib
from scipy import special, optimize
from mcutils.utils.sidecar import AcquisitionParams
from mcutils.scripts.sidecar import index as sidecar_index


def ratio_PTE_STE(anisotropy):
    """
    Computes the powder-averaged PTE / STE

    :param anisotropy: b * (d_parallel - d_perpendicular)
    :return: ratio of powder-averaged signal
    """
    return np.exp(-anisotropy / 6) * np.sqrt(np.pi / (2 * anisotropy)) * special.erfi(np.sqrt(anisotropy / 2))


def ratio_LTE_STE(anisotropy):
    """
    Computes the powder-averaged LTE / STE

    :param anisotropy: b * (d_parallel - d_perpendicular)
    :return: ratio of powder-averaged signal
    """
    return np.exp(anisotropy / 3) * np.sqrt(np.pi / (4 * anisotropy)) * special.erf(np.sqrt(anisotropy))


def ratio_LTE_PTE(anisotropy):
    """
    Computes the powder-averaged LTE / PTE

    :param anisotropy: b * (d_parallel - d_perpendicular)
    :return: ratio of powder-averaged signal
    """
    return np.exp(anisotropy / 2) * np.sqrt(0.5) * special.erf(np.sqrt(anisotropy)) / special.erfi(np.sqrt(anisotropy / 2))


def mean_signal(anisotropy, b_delta, Sbase=1):
    """
    Computes the powder-averaged signal

    :param anisotropy: diffusion anisotropy weighted by the b-value: b (d_parallel - d_perp)
    :param b_delta: anisotropy of the b-tensor (-0.5 for PTE, 0 for STE, and 1 for LTE)
    :param Sbase: signal attenuated with mean diffusivity: S_0 exp(-b MD)
    :return: predicted mean signal
    """
    b_delta = np.asarray(b_delta)
    anisotropy = np.asarray(anisotropy)

    weight = b_delta * anisotropy
    negative = weight < 0
    res_integral = np.zeros(weight.shape)
    res_integral[negative] = special.erfi(np.sqrt(-weight[negative]))
    res_integral[~negative] = special.erf(np.sqrt(weight[~negative]))
    res_integral[weight != 0] /= np.sqrt((4 * abs(weight[weight != 0])) / np.pi)
    res_integral[weight == 0] = 1.
    return Sbase * np.exp(anisotropy * b_delta / 3) * res_integral


def run_single_shell(signal, b_delta, bvals=None):
    """
    Computes the anisotropy from the mean signal

    :param signal: spherical mean signal (M, N) array
    :param b_delta: anisotropy of the b-tensor (M, ) array
    :param bvals: if provided will divide the anisotropy by the b-value to get the d_parallel - d_perpendicular (M, ) array
    :return: (2, N) array with anisotropy, Sbase
    """
    signal = np.asarray(signal)
    if signal.ndim == 1:
        signal = signal[:, None]
    b_delta = np.atleast_1d(b_delta)
    if signal.shape[0] != b_delta.size and b_delta.ndim == 1:
        print(b_delta.shape, signal.shape)
        raise ValueError("Signal and b_delta should share the same last dimension")

    result = np.zeros((2, signal.shape[1]))
    for idx, voxel in enumerate(signal.T):
        start_params = (1, np.min(voxel))
        if np.isfinite(voxel).all() and (voxel != 0).all():
            cost_func = lambda params: np.inf if params[0] < 0 else np.sum((mean_signal(params[0], b_delta)
                                                                            * params[1] - voxel) ** 2)
            best_fit = optimize.minimize(cost_func, start_params, method='powell')
            result[:, idx] = best_fit.x
    if bvals is not None:
        result[0] /= bvals
    return result


def run_from_args(args):
    img = nib.load(args.input)
    sidecar = AcquisitionParams.read(args.input_sidecar)
    if args.mask is None:
        mask = np.ones(img.shape[:3], dtype='bool')
    else:
        mask = nib.load(args.mask).get_data() > 0
        if mask.shape != img.shape[:3]:
            raise ValueError("Shape of the mask does not match shape of the input file")
    signal = img.get_data()[mask].T
    indices = sidecar_index.get_indices(sidecar, args)

    res = np.zeros((2, ) + mask.shape + (max(indices) + 1, ))
    for index in range(max(indices) + 1):
        res[:, mask, index] = run_single_shell(
                signal[index == indices],
                sidecar['b_delta'][index == indices],
                np.mean(sidecar['b'][index == indices])
        )

    nib.Nifti1Image(res[0], img.affine).to_filename(args.output)
    sidecar.groupby(indices, drop=('b_delta', 'b_eta')).write(args.output_sidecar)
    if args.Sbase is not None:
        nib.Nifti1Image(res[1], img.affine).to_filename(args.Sbase)


def add_to_parser(parser):
    parser.add_argument('input', help='input NIFTI image with the spherical mean')
    parser.add_argument('input_sidecar', help='input acuisition parameter file with the b-tensors (.mat or .json)')
    parser.add_argument('output', help='output anisotropy map')
    parser.add_argument('output_sidecar', help='output acquisition parameter file (.mat or .json)')
    parser.add_argument('-S', '--Sbase', help='Optional output of the S_0 exp(-b MD) map')
    parser.add_argument('-m', '--mask', help='Optional mask')
    sidecar_index.add_index_params(parser)
