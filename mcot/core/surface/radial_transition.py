"""Script that fits a sigmoid to the radial index across the white/gray matter boundary
"""

import numpy as np
try:
    import pymc3
    from theano import tensor, sparse
except ImportError:
    raise ImportError("Fitting the radial transition model requires pymc3")
from scipy import sparse as sp_sparse
from scipy import optimize
import nibabel
from . import grid, CorticalMesh
import cifti


def model(radial_index, white, affine=None, surf_mask=None, min_dist=-2., smooth_weight=0.7,
          watson=False, idx_vertex=None, dist=None):
    """
    Creates a probabilistic model of the transition across the white/gray matter bounary

    :param radial_index: (Nx, Ny, Nz) array with the radial index
    :param white: white/gray matter boundary
    :param affine: (4, 4) array with the transformation from voxel to mm space
    :param surf_mask: (Nvertex, ) boolean array, which is True on vertices to be included in the fit
    :param min_dist: only include voxels within this distance from WM/GM boundary
    :param smooth_weight: weighting for the smoothing parameter
    :param watson: assume a Watson distribution for the radial index rather than a normal distribution
    :param idx_vertex: (Nx, Ny, Nz) array with index of closest vertex
    :param dist: distance from the WM/GM boundary
    :return: pymc3 model which fits the sigmoidal transition across the surface
    """
    if affine is None:
        affine = np.eye(4)

    if surf_mask is None:
        surf_mask = np.ones(white.nvertex, dtype='bool')

    vol_mask = np.isfinite(radial_index) & (radial_index != 0.)
    if dist is None:
        dist = grid.signed_distance(white, radial_index.shape, affine)
    if idx_vertex is None:
        idx_vertex_raw = grid.closest_surface(white, vol_mask, affine)
    else:
        idx_vertex_raw = idx_vertex

    nvertex = surf_mask.sum()
    idx_real_vertex = -np.ones(surf_mask.size, dtype='i4')
    idx_real_vertex[surf_mask] = np.arange(nvertex)
    voxel_vertex = idx_real_vertex[idx_vertex_raw]
    voxel_vertex[idx_vertex_raw == -1] = -1
    voxel_use = (voxel_vertex != -1) & (dist > min_dist) & vol_mask & (dist != 0)

    gpp = white.graph_point_point()[surf_mask, :][:, surf_mask]
    smooth = sp_sparse.diags(np.array(1 / gpp.sum(-1)).ravel()).dot(gpp)

    assert abs(smooth.dot(np.ones(smooth.shape[0])) - 1).max() < 1e-13

    # radial index model
    with pymc3.Model() as model:
        d0 = pymc3.Flat('d0', testval=0.5, shape=nvertex)
        log_sigma = pymc3.Flat('log_sigma', testval=-0.5, shape=nvertex)
        sigma = tensor.exp(log_sigma)
        # radial index is zero in WM, 1 in GM
        model_ri = 1 / (1 + tensor.exp(-(dist[voxel_use] - d0[voxel_vertex[voxel_use]]) / sigma[voxel_vertex[voxel_use]]))
        if watson:
            pymc3.Potential('alignment', model_ri * abs(radial_index[voxel_use]) +
                            tensor.sqrt(1 - model_ri ** 2) * np.sqrt(1 - radial_index[voxel_use] ** 2))
        else:
            pymc3.Potential('alignment', -(model_ri - abs(radial_index[voxel_use])) ** 2)
        d_neigh = sparse.structured_dot(smooth, tensor.stack([d0], -1))[:, 0]
        pymc3.Potential('d0_smooth', -smooth_weight * (d_neigh - d0) ** 2)
        ls_neigh = sparse.structured_dot(smooth, tensor.stack([log_sigma], -1))[:, 0]
        pymc3.Potential('ls_smooth', -smooth_weight * (ls_neigh - log_sigma) ** 2)

        # additional output to check
        pymc3.Deterministic('model_ri_1d', model_ri)
        for name, arr_1d in [('model_ri', model_ri), ('observed_ri', abs(radial_index[voxel_use])),
                             ('dist', dist[voxel_use]), ('d0', d0[voxel_vertex[voxel_use]]),
                             ('sigma', sigma[voxel_vertex[voxel_use]])]:
            vol_nan = tensor.fill(radial_index, np.nan)
            vol_filled = tensor.set_subtensor(vol_nan[voxel_use.nonzero()], arr_1d)
            pymc3.Deterministic('%s_3d' % name, vol_filled)
    return model


def run_from_args(args):
    img = nibabel.load(args.coord)
    coord = img.get_data()
    white = CorticalMesh.read(args.white)
    dti = nibabel.load(args.dyad).get_data()
    gdti = (coord * dti[..., None]).sum(-2)
    if args.mask is None:
        surf_mask = np.ones(white.nvertices, dtype='bool')
    else:
        surf_mask = nibabel.load(args.mask).darrays[0].data != 0
    idx_vertex = None if args.idx_vertex is None else nibabel.load(args.idx_vertex).get_data()
    if idx_vertex.ndim == 4:
        idx_vertex = idx_vertex[..., 0]
    distance = None if args.distance is None else nibabel.load(args.distance).get_data()
    tofit = model(abs(gdti[..., 0]), white, img.affine, surf_mask, min_dist=args.min_dist, smooth_weight=args.weight,
                  watson=args.watson, idx_vertex=idx_vertex, dist=distance)
    res = pymc3.find_MAP(model=tofit, fmin=optimize.fmin_cg)

    if args.radial_index is not None:
        model_ri = tofit.fastfn(tofit.model_ri_3d)(res)
        obs_ri = tofit.fastfn(tofit.observed_ri_3d)(res)
        nibabel.Nifti1Image(np.stack([model_ri, obs_ri], -1), img.affine).to_filename(args.radial_index)

    bm = cifti.BrainModel.from_surface(np.where(surf_mask)[0], surf_mask.size, white.anatomy.cifti)
    mat = np.stack([res['d0'], res['log_sigma'], np.exp(res['log_sigma'])])
    cifti.write(args.output, mat, (cifti.Scalar.from_names(['offset', 'log_width', 'width']), bm))
