#!/usr/bin/env python
"""Fit dispersing zeppelin model (single Bingham)"""
import os.path as op
from loguru import logger
import numpy as np
from scipy import optimize, special
from mcutils.utils.hypergeometry import bingham_normalization, der_bingham_normalization
from mcutils.utils.sidecar import AcquisitionParams
from mcutils.utils import spherical
from mcutils.utils.noise import log_non_central_chi
from mcutils.scripts.sidecar import index as sidecar_index
from fsl.utils.filetree import FileTree
from numpy.linalg import LinAlgError
import nibabel
import time
import pandas as pd
import string
from functools import lru_cache

cached_euler2mat = lru_cache(maxsize=4)(spherical.euler2mat)


def bingham_matrix(lk1, lk2, phi, theta, psi, derivative=False):
    """
    Returns the Bingham matrix after rotation

    :param lk1: logarithm of k1
    :param lk2: logarithm of k2
    :param phi: main fibre orientation in the x-y plane
    :param theta: polar angle (0 for main fibre orientation in z-direction, pi/2 for main fibre orientation in x-y plane)
    :param psi: rotation of the major dispersion axis around the main fibre orientation
    :param derivative: also return a (5, 3, 3) derivative matrix
    :return: (3, 3) Bingham matrix (and a (5, 3, 3) matrix with derivatives if requested)
    """
    arr = np.zeros((3, 3))
    arr[0, 0] = -np.exp(lk1)
    arr[1, 1] = -np.exp(lk2)
    rotation = cached_euler2mat(phi, theta, psi)
    res = np.dot(np.dot(rotation, arr), rotation.T)
    if not derivative:
        return res
    der = np.zeros((5, 3, 3))
    der[0] = arr[0, 0] * rotation[:, 0, None] * rotation[None, :, 0]
    der[1] = arr[1, 1] * rotation[:, 1, None] * rotation[None, :, 1]

    angles = [phi, theta, psi]
    for idx in range(3):
        angles[idx] += 1e-6
        upper_rotation = cached_euler2mat(*angles)
        res_upper = np.dot(np.dot(upper_rotation, arr), upper_rotation.T)
        der[idx + 2] = (res_upper - res) / 1e-6
        angles[idx] -= 1e-6
    return res, der


def stick_dispersion(dstick, bmat, bingham, derivative=False):
    """
    Returns the signal for dispersing sticks

    :param dstick: diffusion along the stick
    :param bmat: b-matrix defining the signal acquisition
    :param bingham: Bingham matrix describing the dispersion
    :param derivative: also return the derivatives adds two output with dS/dstick, dS/dbingham
    :return: signal attenuation (1 for unattenuated signal)
    """
    Q = bingham - dstick * bmat
    if not derivative:
        res = bingham_normalization(Q) / bingham_normalization(bingham)
        return res
    num, der_num = der_bingham_normalization(Q)
    denum, der_denum = der_bingham_normalization(bingham)

    dS_ddstick = -(der_num * bmat).sum((-1, -2)) / denum

    dS_dbingham = der_num / denum - (num / denum ** 2)[:, None, None] * der_denum
    return num / denum, dS_ddstick, dS_dbingham


def zepp_to_signal(angles, params, bmat, derivative=False):
    """
    Converts 4/5 parameters into a signal

    The 4 or 5 parameters correspond to
    - primary dispersion eigenvalue
    - secondary dispersion eigenvalue
    - diffusion anisotropy
    - MD - log(S_0) / b (or just MD if S0 is also provided)
    - optionally S0

    :param angles: (theta, phi, and psi)
    :param params: (log of k1, log of k2, dpara - dperp, maximum signal)
    :param bmat: (N, 3, 3) b-matrices
    :param derivative: compute the derivative if requested (adds two outputs with dS/dangles and dS/dparams)
    :return: signal attenuation
    """
    params = np.asarray(params)
    angles = np.asarray(angles)
    if len(params) == 4:
        lk1, lk2, anisotropy, MD = params
        S0 = 1
    elif len(params) == 5:
        lk1, lk2, anisotropy, MD, S0 = params
    else:
        raise ValueError('4 or 5 parameters expected')
    if len(angles) != 3:
        raise ValueError('3 angles expected')
    bh = bingham_matrix(lk1, lk2, angles[0], angles[1], angles[2], derivative=derivative)
    bval = np.trace(bmat, axis1=-1, axis2=-2)
    if not derivative:
        return S0 * np.exp(-bval * MD) * np.exp(bval * anisotropy / 3) * stick_dispersion(anisotropy, bmat, bh)
    bh, dbh_dparams = bh
    stick_res, dstick_dd, dstick_dbh = stick_dispersion(anisotropy, bmat, bh, derivative=True)
    pre_fix = np.exp(-bval * MD) * np.exp(bval * anisotropy / 3)

    dS_dparams = np.zeros((len(params), ) + bmat.shape[:-2])

    dS_dangle = S0 * pre_fix * (dbh_dparams[2:, None, :, :] * dstick_dbh).sum((-1, -2))
    dS_dparams[:2] = S0 * pre_fix * (dbh_dparams[:2, None, :, :] * dstick_dbh).sum((-1, -2))
    dS_dparams[2] = S0 * pre_fix * (dstick_dd + bval / 3 * stick_res)
    dS_dparams[3] = -bval * S0 * pre_fix * stick_res
    if len(params) == 5:
        dS_dparams[4] = pre_fix * stick_res
    return S0 * pre_fix * stick_res, dS_dangle, dS_dparams


def params_to_signal(params, bmat_list, include_S0=False, derivative=False):
    """
    Converts parameters for one or more b-shells into a signal

    First three parameters are the three angles defining the orientation
    for each component there are two more parameters with the:
    1. log(k1)
    2. log(k2)
    3. d2-d1
    4. MD - log(S_0) / b (or MD if S0 is also provided)
    5. optionally S0

    :param params: (3 + 4/5 * N, ) array for N b-shells
    :param bmat_list: (N, ) list of lists containing the b-matrices
    :param include_S0: include a parameter representing S0 rather than letting it be submerged in MD
    :param derivative: also return the (N, ) tuple with (3 + 4/5 * N, M) array with the derivatives
    :return: (N, ) tuple of lists with the signal attenuations (or full signal if include_S0 is True)
    """
    nparams = len(params)
    nshells = len(bmat_list)
    nparams_per_shell = 5 if include_S0 else 4
    assert nparams == nshells * nparams_per_shell + 3
    res = []
    der_res = []
    for idx, bmat in enumerate(bmat_list):

        signal = zepp_to_signal(
                params[:3],
                params[3 + nparams_per_shell * idx: 3 + nparams_per_shell * (idx + 1)],
                bmat, derivative=derivative
        )
        if derivative:
            signal, dS_dangle, dS_dparams = signal
            der_res.append(np.zeros((nparams, signal.size)))
            der_res[-1][:3, :] = dS_dangle
            der_res[-1][3 + nparams_per_shell * idx: 3 + nparams_per_shell * (idx + 1), :] = dS_dparams
            res.append(signal)
        else:
            res.append(signal)
    if derivative:
        return tuple(res), tuple(der_res)
    return tuple(res)


def params_to_error(params, observed, bmat_list, include_S0=False, derivative=False):
    """
    Computes the error between the predicted and observed signal attenuations

    :param params: (3 + 4/5 * N, ) array for N b-shells (see params_to_signal for more details)
    :param observed: (N, ) list of array with the observed signal attenuations in each b-shell
    :param bmat_list: (N, ) list of lists containing the b-matrices
    :param include_S0: include a parameter representing S0 rather than letting it be submerged in MD
    :param derivative: also return the (3 + 4/5 * N, ) array with the error derivative
    :return: float with the Euclidean error
    """
    try:
        signal_full = params_to_signal(params, bmat_list, include_S0=include_S0, derivative=derivative)
    except LinAlgError:
        if derivative:
            return np.inf, np.zeros(len(params))
        else:
            return np.inf

    if not derivative:
        total = 0.
        for sig, obs in zip(signal_full, observed):
            total += np.sum((sig - obs) ** 2)
        return total
    signal, der_signal = signal_full
    total = 0.
    total_der = np.zeros(len(params))
    for sig, der_sig, obs in zip(signal, der_signal, observed):
        total += np.sum((sig - obs) ** 2)
        total_der += 2 * np.sum((sig - obs) * der_sig, -1)
    return total, total_der


def params_to_non_central_logp(params, observed, bmat_list, include_S0=False, derivative=False, ncoils=1,
                               fixed_noise_var=None):
    """
    Computes the log(p) of the Non-central-chi fit to the data

    :param params: (4 + 4/5 * N, ) array for N b-shells (last parameter is the log of the noise variance, for others see params_to_signal)
    :param observed: (N, ) list of array with the observed signal attenuations in each b-shell
    :param bmat_list: (N, ) list of lists containing the b-matrices
    :param include_S0: include a parameter representing S0 rather than letting it be submerged in MD
    :param derivative: also return the (3 + 4/5 * N, ) array with the error derivative
    :param ncoils: number of coils (set to 1 for the Rician noise distribution)
    :param fixed_noise_var: noise variance (if not set, last parameters is log(noise variance)
    :return: float with the log(p)
    """
    p_signal = params[:-1] if fixed_noise_var is None else params
    try:
        signal_full = params_to_signal(p_signal, bmat_list, include_S0=include_S0, derivative=derivative)
    except LinAlgError:
        if derivative:
            return -np.inf, np.zeros(len(params))
        else:
            return -np.inf
    noise_var = np.exp(params[-1]) if fixed_noise_var is None else fixed_noise_var
    total = 0.
    if not derivative:
        for sig, obs in zip(signal_full, observed):
            total += np.sum(log_non_central_chi(obs, sig, noise_var, ncoils))
        return total
    signal, der_signal = signal_full
    total_der = np.zeros(len(params))
    for sig, der_sig, obs in zip(signal, der_signal, observed):
        res = log_non_central_chi(obs, sig, noise_var, ncoils,
                                  derivative="both" if fixed_noise_var is None else "signal")
        total += np.sum(res[0])
        signal_slice = slice(None)
        if fixed_noise_var is None:
            signal_slice = slice(None, -1)
            total_der[-1] += np.sum(res[2]) * noise_var
        total_der[signal_slice] += np.sum(
                der_sig * res[1], axis=-1
        )
    return total, total_der


def angles_from_dti(V1, V2, V3):
    """
    Returns the Euler angles based on DTI data

    :param V1: (N, 3) array of primary eigenvector
    :param V2: (N, 3) array of secondary eigenvector
    :param V3: (N, 3) array of tertiary eigenvector
    :return: (N, 3) array of Euler angles (phi, theta, and psi)
    """
    return np.stack(spherical.mat2euler(
            np.stack([V3, V2, V1], -1)
    ), -1)


class Fitter(object):
    def __init__(self, data_list, bmat_list, include_S0=False, ncoils=0, noise_var=None):
        """
        Provides a fit for a single voxel

        :param data_list: (K,) length list with (M_k, ) array for K shells of N voxels with M_k acquisitions
        :param bmat_list: (K,) length list with (M_k, 3, 3) array of b-matrices
        :param include_S0: if True include a parameter representing the S0 for every shell
        :param ncoils: If set to non-zero: assumes a non-central model for the noise with 2 ncoils degrees of freedom rather than Gaussian
        :param noise_var: Set to fix the noise variance rather than allow it to be a free parameters (only used if ncoils is non-zero)
        :return: best-fit parameters (same length as init_params)
        """
        self.data_list = data_list
        self.bmat_list = bmat_list
        self.include_S0 = include_S0
        self.ncoils = ncoils
        self.noise_var = noise_var

    @property
    def nparams_per_shell(self, ):
        return 5 if self.include_S0 else 4

    @property
    def nparams(self, ):
        return self.nparams_per_shell * self.nshells + (4 if self.ncoils and self.noise_var is None else 3)

    @property
    def nshells(self, ):
        return len(self.data_list)

    def single(self, init_params, ignore_angle=False):
        """
        Provides a single fit to the full dataset at once

        :param init_params: initial parameter set
        :param ignore_angle: if True do not fit the angle parameters
        :return: best-fit parameters (same length as init_params)
        """
        if init_params.size != self.nparams:
            raise ValueError(f"Initial parameters has incorrect size of {init_params.size} instead of {self.nparams}")
        bounds = np.zeros((self.nparams, 2))
        bounds[()] = (-np.inf, np.inf)
        bounds[3:-1:self.nparams_per_shell] = (-5, 5)
        bounds[4:-1:self.nparams_per_shell] = (-5, 5)
        bounds[5::self.nparams_per_shell] = (0., np.inf)
        if self.include_S0:
            bounds[6::self.nparams_per_shell] = (0, np.inf)
            bounds[7::self.nparams_per_shell] = (0, np.inf)

        if ignore_angle:
            full_params = np.zeros(self.nparams)
            full_params[:3] = init_params[:3]
            if self.ncoils:
                if self.noise_var is None:
                    full_params[-1] = init_params[-1]
                use_slice = slice(3, -1) if self.noise_var is None else slice(3, None)
                def tofit(params):
                    full_params[use_slice] = params
                    f, g = params_to_non_central_logp(full_params, self.data_list, self.bmat_list,
                                                      self.include_S0, True, ncoils=self.ncoils,
                                                      fixed_noise_var=self.noise_var)
                    return -f, -g[use_slice]
                actual_init = init_params[use_slice]
                bounds = bounds[use_slice]
            else:
                def tofit(params):
                    full_params[3:] = params
                    f, g = params_to_error(full_params, self.data_list, self.bmat_list, self.include_S0, True)
                    return f, g[3:]
                actual_init = init_params[3:]
                bounds = bounds[3:]
        else:
            actual_init = init_params
            if self.ncoils:
                def tofit(params):
                    f, g = params_to_non_central_logp(params, self.data_list, self.bmat_list,
                                                      self.include_S0, True, ncoils=self.ncoils,
                                                      fixed_noise_var=self.noise_var)
                    return -f, -g
            else:
                def tofit(params):
                    return params_to_error(params, self.data_list, self.bmat_list, self.include_S0, True)

        voxel_fit = optimize.minimize(tofit, actual_init, method='l-bfgs-b', bounds=bounds, jac=True,
                                      options={'eps': 1e-5, 'maxiter': 500})

        if ignore_angle:
            if self.ncoils and self.noise_var is None:
                full_params[3:-1] = voxel_fit.x
            else:
                full_params[3:] = voxel_fit.x
            return full_params
        else:
            return voxel_fit.x

    def per_shell(self, init_params):
        """
        Fits the parameters for every shell separately
        """
        res = np.zeros(init_params.shape)
        res[:3] = init_params[:3]
        if self.ncoils:
            res[-1] = init_params[-1]
        for idx in range(self.nshells):
            sub_fitter = Fitter(
                    data_list=[self.data_list[idx]],
                    bmat_list=[self.bmat_list[idx]],
                    include_S0=self.include_S0,
                    ncoils=self.ncoils,
                    noise_var=self.noise_var
            )
            sub_params = np.concatenate(
                    (init_params[:3],
                     init_params[self.nparams_per_shell * idx + 3:self.nparams_per_shell * (idx + 1) + 3]),
            )
            if self.ncoils and self.noise_var is None:
                sub_params = np.append(sub_params, init_params[-1])
            shell_fit = sub_fitter.single(sub_params, ignore_angle=True)
            if self.ncoils and self.noise_var is None:
                shell_fit = shell_fit[:-1]
            res[self.nparams_per_shell * idx + 3:self.nparams_per_shell * (idx + 1) + 3] = shell_fit[3:]
        return res

    def only_angle(self, init_params):
        """
        Fits only the angular parameters
        """
        full_params = np.zeros(self.nparams)
        full_params[3:] = init_params[3:]
        if self.ncoils:
            if self.noise_var is None:
                actual_init = np.append(init_params[:3], init_params[-1])
            else:
                actual_init = init_params[:3]

            def tofit(params, data_list, bmat_list, include_S0, derivative=True):
                full_params[:3] = params[:3]
                if self.noise_var is None:
                    full_params[-1] = params[-1]
                f, g = params_to_non_central_logp(full_params, data_list, bmat_list,
                                                  include_S0, derivative, ncoils=self.ncoils,
                                                  fixed_noise_var=self.noise_var)
                grad = g[:3]
                if self.noise_var is None:
                    grad = np.append(grad, g[-1])
                return -f, -grad
        else:
            actual_init = init_params[:3]

            def tofit(params, data_list, bmat_list, include_S0, derivative=True):
                full_params[:3] = params
                f, g = params_to_error(full_params, data_list, bmat_list, include_S0, derivative)
                return f, g[:3]
        voxel_fit = optimize.minimize(tofit, actual_init, method='l-bfgs-b',
                                      args=(self.data_list, self.bmat_list,
                                            self.include_S0, True), jac=True,
                                      options={'eps': 1e-5, 'maxiter': 500})
        full_params[:3] = voxel_fit.x[:3]
        if self.ncoils and self.noise_var is None:
            full_params[-1] = voxel_fit.x[3]
        return full_params

    def iterate(self, init_params):
        """
        Iterates between fitting the individual shells and the angles
        """
        if self.ncoils:
            gauss_init_params = init_params[:-1] if self.noise_var is None else init_params
            gauss_params = Fitter(self.data_list, self.bmat_list, self.include_S0, ncoils=0).iterate(gauss_init_params)
            if self.noise_var is None:
                gauss_signal = np.concatenate(params_to_signal(gauss_params, self.bmat_list, self.include_S0), 0)
                init_params = np.append(gauss_params, np.log(np.var(gauss_signal - np.concatenate(self.data_list, 0))))
            else:
                init_params = gauss_params

        res = self.per_shell(init_params)
        for _ in range(2 if self.ncoils else 3):
            angle = self.only_angle(res)
            for idx in range(3, 3 + self.nparams_per_shell):
                use_slc = slice(idx, -1, self.nparams_per_shell) if self.ncoils and self.noise_var is None else slice(idx, None, self.nparams_per_shell)
                angle[use_slc] = np.median(angle[use_slc])
            res = self.per_shell(angle)
        return res


def run(data, sidecar: AcquisitionParams, indices=None, dti_vectors=None, include_S0=False,
        ncoils=False, noise_var=None, return_fitter=False, init_params=None):
    """
    Fits the dispersing zeppelin model to the observed signal attenuation

    :param data: (N, M) array for N voxels and M gradient orientations
    :param sidecar: describes the acquired data
    :param indices: index of which shell each observation will be grouped into (default: all in the same shell)
    :param dti_vectors: tuple with (V1, V2, and V3), i.e., 3 (N, 3) arrays
    :param include_S0: include a parameter representing S0 rather than letting it be submerged in MDprime
    :param ncoils: Assumes a non-central chi model with 2 * ncoils dof for the noise rather than Gaussian
    :param noise_var: noise variance (treated as variable if not set; not recommended)
    :param return_fitter: debugging option to return the fitter
    :param init_params: initial values for the parameters  log(k1), log(k2), d2-d1, MD - log(S_0) / b (or MD if include_S0 is true), S0 if include_S0)
    :return: Tuple with:

        - Acquisition parameters per shell
        - best-fit parameters per shell in a pandas dataframe
    """
    if sidecar.n != data.shape[-1]:
        raise ValueError(f'XPS file describes {sidecar.n} volumes, but data contains {data.shape[-1]} volumes')
    if indices is None:
        indices = np.zeros(data.shape[-1], dtype='i4')

    nshells = max(indices) + 1
    logger.info(f'Total number of shells: {nshells}')
    bmat_list = [sidecar['btensor'][indices == idx] / 1e3 for idx in range(nshells)]
    data_list = [data[:, indices == idx] for idx in range(nshells)]
    bvals = [np.median(np.trace(bmat, axis1=-1, axis2=-2)) for bmat in bmat_list]
    logger.info(f'Median b-value per shell: {bvals}')

    nparams_per_shell = 5 if include_S0 else 4
    nparams = nshells * nparams_per_shell + (4 if ncoils and noise_var is None else 3)
    res = np.zeros((data_list[0].shape[0], nparams)) + 0.1
    res[:, 3::nparams_per_shell] = 1.5
    res[:, 4::nparams_per_shell] = 1.
    res[:, 5::nparams_per_shell] = 1.
    if ncoils and noise_var is None:
        res[:, -1] = 0

    # set initial dispersion/diffusion parameters
    if include_S0:
        res[:, 6::nparams_per_shell] = np.array([np.amax(d, -1) for d in data_list]).T
        # mean signal relative to max
        res[:, 7::nparams_per_shell] = np.array([-np.log(np.mean(d, -1) / np.amax(d, -1)) / b
                                                 for b, d in zip(bvals, data_list)]).T
    else:
        # mean signal relative to 1
        res[:, 6::nparams_per_shell] = np.array([-np.log(np.mean(d, -1)) / b for b, d in zip(bvals, data_list)]).T
    if init_params is not None:
        for idx, value in enumerate(init_params):
            res[:, (idx + 3)::nparams_per_shell] = value

    if dti_vectors is not None:
        # the primary eigenvector should point in the z-direction after inverse rotation
        res[:, :3] = angles_from_dti(*dti_vectors)

    if return_fitter:
        return [(Fitter([d[idx] for d in data_list], bmat_list, include_S0=include_S0,
                        ncoils=ncoils, noise_var=noise_var), res[idx])
                for idx in range(res.shape[0])]

    start = time.time()
    for idx in range(res.shape[0]):
        fitter = Fitter([d[idx] for d in data_list], bmat_list, include_S0=include_S0, ncoils=ncoils,
                        noise_var=noise_var)
        try:
            res[idx] = fitter.iterate(res[idx])
        except Exception as e:
            logger.warning(f'Failed for voxel {idx} with message:')
            logger.warning(str(e))
            res[idx] = np.nan
        if idx == 9 or idx % max(int(res.shape[0] / 20), 1) == 0:
            logger.info('Processed {} voxels with an average processing time of {} ms'.format(
                    idx + 1, int(1000 * (time.time() - start) / (idx + 1))))
    return (
        sidecar.groupby(indices, drop=('theta', 'phi', 'psi', 'b_delta')),
        clean(res, include_S0=include_S0, non_central=ncoils and noise_var is None)
    )


def clean(parameters, include_S0, non_central):
    """
    Cleans the parameters

    :param parameters: (M, (3 or 4) + (4 or 5) * N) array of best-fit parameters for N shells and M voxels
    :param include_S0: Whether S0 was included as a separate parameter in the fitting
    :param non_central: Whether the noise variance in the non-central chi noise was estimated
    :return: (M, ) pandas dataframe with the labeled best-fit parameters
    """
    nparams_per_shell = 5 if include_S0 else 4

    # clean the angles
    parameters = swap_dispersion(parameters, nparams_per_shell)
    angles_arr = spherical.clean_euler(parameters[:, 0], parameters[:, 1], parameters[:, 2])
    res = [
        ('phi', angles_arr[0]),
        ('theta', angles_arr[1]),
        ('psi', angles_arr[2]),
    ]

    rotate_mat = spherical.euler2mat(parameters[:, 0], parameters[:, 1], parameters[:, 2])
    for idx2, name in enumerate(('sec_disp', 'main_disp', 'dyad')):
        for idx1, dim in enumerate('xyz'):
            res.append((f'{name}_{dim}', rotate_mat[:, idx1, idx2]))

    # clean the other parameters
    for idx_param, label in zip(
            range(3, parameters.shape[-1] - 1, nparams_per_shell),
            ('_' + letter for letter in string.ascii_uppercase),
    ):
        if parameters.shape[-1] == (4 if non_central else 3) + nparams_per_shell:
            label = ''
        lk1, lk2, anis, MD = parameters[:, idx_param: idx_param + 4].T
        as_dict = [
            ('log_k1', lk1),
            ('log_k2', lk2),
            ('k1', np.exp(lk1)),
            ('k2', np.exp(lk2)),
            ('disp1', calc_dispersion(np.exp(lk1))),
            ('disp2', calc_dispersion(np.exp(lk2))),
            ('anisotropy', anis / 1e3),  # unit conversion to mm^2/s
        ]
        if include_S0:
            as_dict.append(('MD', MD / 1e3))
            as_dict.append(('S0', parameters[:, idx_param + 4]))
        else:
            as_dict.append(('MDprime', MD / 1e3))
        res.extend((key + label, value) for key, value in as_dict)
    if non_central:
        res.append(('global_noise', np.exp(parameters[:, -1])))
    return pd.DataFrame(dict(res))


def swap_dispersion(res_fit, nparams_per_shell=4):
    """
    Ensures that the dispersion along axis 2 is greater than across axis 1

    :param res_fit: input parameters
    :return: parameters with the same signal
    """
    disp1_degrees = calc_dispersion(np.exp(res_fit[..., 3:-2:nparams_per_shell].mean(-1)))
    disp2_degrees = calc_dispersion(np.exp(res_fit[..., 4:-2:nparams_per_shell].mean(-1)))
    swap = disp1_degrees > disp2_degrees
    logger.info('swapping dispersion for %d out of %d voxels', swap.sum(), swap.size)
    res = res_fit.copy()
    res[swap, 2] = (res[swap, 2] + np.pi / 2) % np.pi
    res[swap, 3:-2:nparams_per_shell] = res_fit[swap, 4:-2:nparams_per_shell]
    res[swap, 4:-2:nparams_per_shell] = res_fit[swap, 3:-2:nparams_per_shell]
    return res


@np.vectorize
def calc_dispersion(k):
    """
    Computes the angular dispersion from the kappa parameter

    Finds the angle in degrees that contains 50% of the fibres
    """
    zero_point = -special.erfi(np.sqrt(k) * np.cos(0))
    end_point = -special.erfi(np.sqrt(k) * np.cos(np.pi/2.))
    half_point = (zero_point + end_point) / 2.
    res = optimize.minimize_scalar(lambda x: (half_point + special.erfi(np.sqrt(k) * np.cos(x))) ** 2,
                                   bounds=(0, np.pi/2.), method='Golden')
    return res.x / np.pi * 180.


@np.vectorize
def calc_k(angle):
    """
    Computes the kappa parameter from the angular dispersion

    :param angle: angle containing 50% of the fibres
    :return: kappa parameter
    """
    return optimize.minimize_scalar(
        lambda x: (special.erfi(np.sqrt(x) * np.cos(angle * np.pi / 180.)) / special.erfi(np.sqrt(x)) - 0.5) ** 2
    ).x


def run_from_args(args):
    """
    Runs the script based on a Namespace containing the command line arguments
    """
    img = nibabel.load(args.mask)
    mask = img.get_data() > 0
    sidecar = AcquisitionParams.read(args.sidecar)
    signal = nibabel.load(args.data).get_data()[mask]

    dti_vectors = None
    if args.DTI is not None:
        dti_vectors = tuple(nibabel.load('{}_V{}.nii.gz'.format(args.DTI, idx)).get_data()[mask] for idx in range(1, 4))
    indices = sidecar_index.get_indices(sidecar, args)

    use = np.zeros(indices.shape, dtype='bool')
    for idx in range(max(indices) + 1):
        if max(sidecar['b'][indices ==idx]) > 300:
            use[indices == idx] = True
    signal = signal[:, use]
    sidecar = sidecar[use]
    indices = sidecar_index.get_indices(sidecar, args)

    sidecar_shell, res = run(
            data=signal,
            sidecar=sidecar,
            dti_vectors=dti_vectors,
            indices=indices,
            include_S0=args.include_S0,
            ncoils=args.ncoils,
            noise_var=args.noise_var
    )
    direc, filename = op.split(args.output)

    def to_full(arr):
        full_arr = np.zeros(mask.shape + arr.shape[1:])
        full_arr[mask] = arr
        return nibabel.Nifti1Image(full_arr, img.affine)

    def to_full_dyad(name):
        as_arr = np.stack([res[f'{name}_{dim}'] for dim in 'xyz'], -1)
        return to_full(as_arr)

    def to_full_shell(name):
        as_arr = np.stack([res[f'{name}_{idx_shell}'] for idx_shell in string.ascii_uppercase[:sidecar_shell.n]], -1)
        return to_full(as_arr)

    full_res = np.zeros(mask.shape + (res.shape[1], ))
    full_res[mask] = res

    tree = FileTree.read('fit_dispersion', directory=direc, basename=filename)
    to_full_dyad('dyad').to_filename(tree.get('dyad', make_dir=True))
    to_full_dyad('main_disp').to_filename(tree.get('dyad_disp'))

    to_full_shell('disp1').to_filename(tree.get('dispA'))
    to_full_shell('disp2').to_filename(tree.get('dispB'))
    to_full_shell('anisotropy').to_filename(tree.get('microA'))
    if args.include_S0:
        to_full_shell("MD").to_filename(tree.get('MD'))
        to_full_shell("S0").to_filename(tree.get('S0'))
    else:
        to_full_shell("MDprime").to_filename(tree.get('MDprime'))
    if args.ncoils and args.noise_var is not None:
        to_full(res['global_noise']).to_filename(tree.get('global_noise'))

    sidecar_shell.write(tree.get('sidecar'))


def add_to_parser(parser):
    """
    Creates the parser of the command line arguments
    """
    parser.add_argument('data', help='NIFTI file with diffusion data')
    parser.add_argument('sidecar', help='.json or .mat file with the sequence information (see sidecar_merge)')
    parser.add_argument('output', help='output basename')
    parser.add_argument('-m', '--mask', help='mask which voxels should be used')
    parser.add_argument('--DTI', help='basename of a DTI fit ot the LTE data (used to initialize the fit)')
    parser.add_argument('-S0', '--include_S0', action='store_true',
                        help='Include S0 as a separate parameter in the fitting rather than including it in the MDprime')
    parser.add_argument('-nc', '--ncoils', type=int,
                        help='Model the noise as Non-central distribution with ' +
                             'ncoils * 2 degrees of freedom rather than Gaussian')
    parser.add_argument('-var', '--noise_var', type=float,
                        help='Variance of the noise as estimated outside of the brain or in the CSF '
                             '(only used if ncoils is set; default: treat as variable)')
    sidecar_index.add_index_params(parser, exclude=('b_delta',))
