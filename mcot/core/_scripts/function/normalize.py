#!/usr/bin/env python
"""
Normalizes the functional MRI data based on the noise level and concatenates results

Based on the variance_normalize.m script from Saad Jbabdi
"""
from mcot.core import scripts
from scipy import linalg
import numpy as np
from nibabel import cifti2


def run(data, demean=True):
    """
    Normalizes the functional MRI data

    :param data: (..., N) array of fMRI data with N timepoints
    :param demean: if True demean the data as well as normalizing
    :return: (..., N) array of normalized and potentially demeaned data

    Example matlab code from variance_normalize.m by Saad Jbabdi
    function yn = variance_normalise(y)
    % yn = variance_normalise(y)
    % y is TxN

    addpath ~steve/matlab/groupPCA/
    yn=y;
    [uu,ss,vv]=ss_svds(y,30);
    vv(abs(vv)<2.3*std(vv(:)))=0;
    stddevs=max(std(yn-uu*ss*vv'),0.001);
    yn=yn./repmat(stddevs,size(yn,1),1);
    """
    res = data.copy()
    if demean:
        res -= res.mean(-1)[..., None]
    u, s, v = linalg.svd(res.reshape(-1, res.shape[-1]), full_matrices=False)
    # threshold spatial maps
    u[abs(u) < 2.3 * np.std(u, 0)] = 0
    sdiag = np.diag(s, 0)
    residuals = res - u.dot(sdiag.dot(v))
    res /= np.std(residuals)
    return res.reshape(data.shape)


def run_from_args(args):
    """
    Runs the script based on a Namespace containing the command line arguments
    """
    res = []
    ref_axes = None
    for arr, axes in args.input:
        res.append(run(arr.T).T)
        if ref_axes is None:
            ref_axes = axes
        else:
            assert ref_axes == axes
    full_arr = np.concatenate(res, 0)
    series = ref_axes[0]
    new_series = cifti2.SeriesAxis(series.start, series.step, full_arr.shape[0], series.unit)
    args.output((full_arr, (new_series, ref_axes[1])))


def add_to_parser(parser):
    """
    Creates the parser of the command line arguments
    """
    parser.add_argument('output', type=scripts.output, help='normalized output dataset')
    parser.add_argument('input', type=scripts.greyordinate_in, nargs='+', help='one or more functional MRI dataset')
