"""Computes the gyral coordinate system

For each point in space defines the radial, sulcal, and gyral orientations
"""
from numpy import linalg
import numpy as np
import nibabel
from nibabel import gifti
from . import grid, utils
from .cortical_mesh import CorticalMesh
from fsl.wrappers import gps, LOAD


class WeightedOrientation(object):
    """Can compute an averaged radial/tangential fiber orientations for every voxel
    """
    _flip_inpr = False

    def __init__(self, white, pial, sulcal_depth, resolution_grid, target_affine):
        """Prepares computation of the radial/tangential fiber orientations

        :param white: white/gray matter boundary
        :type white: CorticalMesh
        :param pial: pial surface
        :type pial: CorticalMesh
        :param sulcal_depth: sulcal depth
        :param resolution_grid: voxel size of the accelerator grid in mm
        :param target_affine: (4x4) array giving the voxel -> mm conversion for the target grid
        """
        self.white = white
        self.pial = pial
        self.sulcal_depth = sulcal_depth
        self.resolution_grid = resolution_grid
        self.target_affine = target_affine

        shape, affine = grid.bounding(pial, resolution_grid)
        self.affine = affine

        self.white_hit = grid.intersect(white, shape, affine)
        self.pial_hit = grid.intersect(pial, shape, affine)

        self.white_vox = self.white.apply_affine(target_affine)
        self.pial_vox = self.pial.apply_affine(target_affine)

        self.white_grad = self.white_vox.gradient(sulcal_depth)
        self.pial_grad = self.pial_vox.gradient(sulcal_depth)
        self.white_normal = self.white_vox.normal()
        self.pial_normal = self.pial_vox.normal()

        self.white_grad_point = self.white_vox.gradient(sulcal_depth, atpoint=True)
        self.pial_grad_point = self.pial_vox.gradient(sulcal_depth, atpoint=True)
        self.white_normal_point = self.white_vox.normal(atpoint=True)
        self.pial_normal_point= self.pial_vox.normal(atpoint=True)

        self.smooth_orient = np.zeros((0, 3))

    def average_line_vox(self, mm_index, norient=1000, power_dist=-1.):
        """Computes the radial/tangential hemisphere at the given point

        This uses the main FOTACS algorithm:

        1. Draw straight lines through the point of interest connecting the cortical surfaces at both sides
        2. Linearly interpolate the normal/sulcal depth gradient along this line

        Repeat these steps for `norient` random orientations.
        Average these orientations with the weighting set by the line length ** `power_dist`.

        :param mm_index: (3, ) vector of position in mm
        :param norient: number of random orientations to try
        :param power_dist: power-law used to downweight longer faces (`weight = dist ** power_dist`)
        :return: Tuple with 4 elements:

            1. interpolated normal
            2. interpolated sulcal depth gradient
            3. length of shortest line hitting surface on both sides
            4. number between 0 and 0.5 indicating location along shortest line (0 if at edge, 0.5 if in middle of gyrus)
        """
        if self.smooth_orient.shape[0] != norient:
            rand_orient = np.randn(3, norient)
            rand_orient /= np.sqrt(np.sum(rand_orient ** 2, 0))

            self.smooth_orient = gps(LOAD, ndir=3000)

        orientations = np.concatenate((self.smooth_orient, -self.smooth_orient), 0)
        w_ix, w_pos = self.white_hit.ray_intersect(mm_index, orientations)
        normal_inpr = np.sum(self.white.normal()[:, w_ix] * orientations.T, 0)
        segment = 1 if normal_inpr[w_ix != -1].sum() < 0 else 2
        if segment == 1:
            o_ix, o_pos = self.pial_hit.ray_intersect(mm_index, -orientations, pos_inpr=1)
            use = (w_ix != -1) & (o_ix != -1) & (normal_inpr <= 0)
        else:
            o_ix, o_pos = w_ix[norient:], w_pos[norient:]
            w_ix, w_pos = w_ix[:norient], w_pos[:norient]
            use = (w_ix != -1) & (o_ix != -1) & (normal_inpr[:norient] >= 0) & (normal_inpr[norient:] >= 0)

        other_grad = self.white_grad if segment == 2 else self.pial_grad
        other_normal = -self.white_normal if segment == 2 else self.pial_normal

        # linear interpolation of the orientations
        dist_white = np.sqrt(np.sum((w_pos[use, :] - mm_index) ** 2, -1))[:, None]
        dist_other = np.sqrt(np.sum((o_pos[use, :] - mm_index) ** 2, -1))[:, None]
        dist = dist_white + dist_other
        weight = dist ** power_dist

        res = []
        for other_inp, white_inp in [(other_normal, self.white_normal),
                                     (other_grad, self.white_grad)]:
            linear_interp = (dist_white * other_inp.T[o_ix[use], :] + dist_other * white_inp.T[w_ix[use], :]) / dist
            linear_interp *= weight / np.sqrt(np.sum(linear_interp ** 2, -1))[:, None]
            linear_interp[~np.isfinite(linear_interp)] = 0.
            cov = np.dot(linear_interp.T, linear_interp)
            val, vec = linalg.eigh(cov)
            res.append(vec[:, np.argmax(val)])
        res.append(np.inf if dist.size == 0 else dist.min())
        ratio_length = np.nan
        if dist.size != 0:
            idx = np.argmin(dist)
            ratio_length = min((dist_white[idx], dist_other[idx])) / dist[idx]
        res.append(ratio_length)
        return tuple(res)

    def average_line_grid(self, shape, zval=None, norient=1000, power_dist=-1.):
        """Computes the radial/tangential orientations on a grid.

        This uses the main FOTACS algorithm:
        1. Draw straight lines through the point of interest connecting the cortical surfaces at both sides
        2. Linearly interpolate the normal/sulcal depth gradient along this line
        Repeat these steps for `norient` random orientations.
        Average these orientations with the weighting set by the line length ** `power_dist`.

        :param shape: (nx, ny, nz); defines the shape of the output volume
        :param zval: only evaluates a single horizontal slice if set
        :param norient: number of random orientations to try
        :param power_dist: power-law used to downweight longer faces (`weight = dist ** power_dist`)
        """
        if len(shape) != 3:
            use = shape != 0
            shape = use.shape
        else:
            use = np.ones(shape, dtype='bool')
        res = np.zeros(shape + (3, 2), dtype='f4')
        thickness = np.zeros(shape + (2, ), dtype='f4')
        thickness[()] = np.nan
        use = use & (grid.signed_distance(self.pial, shape, affine=self.target_affine) < 0)
        if zval is not None:
            use = use & (np.arange(shape[-1])[None, None, :] == zval)
        res[~use, :] = np.nan
        for ixvox in zip(*np.where(use)):
            av_line = self.average_line_vox(utils.affine_mult(self.target_affine, ixvox),
                                            norient=norient, power_dist=power_dist)
            thickness[ixvox] = av_line[2:]
            res[ixvox] = np.array(av_line[:2]).T
        return res, thickness

    @staticmethod
    def _weights(surface_vertices, target, normal, pos_inpr=0, power_dist=-1.):
        """Computes the weight of every surface element on every volume_element.

        :param surface_vertices: (N, 3) array with vertex locations along the surface
        :param target: (3, ) array with position of interest
        :param normal: (N, 3) array with the surface normals at the vertices
        :param pos_inpr: ignore vertex element with normals pointing to the point of interest (if pos_inpr < 0) or away from the point of interest (if pos_inpr < 0); default: no filtering
        :param power_dist: resulting weights = distance ** power_dist
        :return: (N, ) array of vertex weights (set to 0 for ignored vertices; see `pos_inpr` parameter)
        """
        offset = surface_vertices - np.array(target)[:, None]
        weight = np.sum(offset ** 2, 0) ** (power_dist / 2)
        if pos_inpr != 0:
            inpr = np.sum(offset * normal, 0)
            if pos_inpr > 0:
                weight[inpr > 0] = 0
            else:
                weight[inpr < 0] = 0
        return weight

    def average_point(self, voxel, power_dist=-1.):
        """Computes the radial and tangential hemisphere at given voxel.

        This algorithm averages the normal/sulcal depth gradient of every vertex weighted by its distance from the point of interest

        :param mm: (3, ) array with point of interest in voxel coordinates
        :param power_dist: power-law used to downweight longer faces (`weight = dist ** power_dist`)
        """
        res = np.zeros((3, 2))
        cov_radial = np.zeros((3, 3))
        cov_tangential = np.zeros((3, 3))
        for surf, norm, grad, pos_inpr in [(self.white_vox, self.white_normal_point, self.white_grad_point, 0),
                         (self.pial_vox, self.pial_normal_point, self.pial_grad_point, 1 if self._flip_inpr else -1)]:
            weight = self._weights(surf.vertices, voxel, norm, power_dist=power_dist, pos_inpr=pos_inpr)
            weighted_norm = weight[None, :] * norm
            cov_radial += np.dot(weighted_norm, weighted_norm.T)

            weighted_grad = weight[None, :] * grad
            cov_tangential += np.dot(weighted_grad, weighted_grad.T)
        val, vec = linalg.eigh(cov_radial)
        res[:, 0] = vec[:, np.argmax(val)]
        val, vec = linalg.eigh(cov_tangential)
        res[:, 1] = vec[:, np.argmax(val)]
        return res

    def average_point_grid(self, shape, zval=None, power_dist=-1., outside_pial=False):
        """Computes the primary orientations at every point within the pial surface.

        This algorithm averages the normal/sulcal depth gradient of every vertex weighted by its distance from the point of interest

        :param shape: shape of the resulting array
        :param zval: only process a single horizontal slice
        :param power_dist: power-law used to downweight longer faces (`weight = dist ** power_dist`)
        :param outside_pial: if True also run for voxels outside of the pial surface
        """
        if len(shape) != 3:
            use = shape != 0
            shape = use.shape
        else:
            use = np.ones(shape, dtype='bool')
        if not outside_pial:
            use = use & (grid.signed_distance(self.pial, shape, self.target_affine) < 0)
        if zval is not None:
            use = use & (np.arange(shape[-1])[None, None, :] == zval)
        res = np.zeros(use.shape + (3, 2))
        for voxel in zip(*np.where(use)):
            res[voxel] = self.average_point(voxel, power_dist=power_dist)
        return res

    def closest_vertex_grid(self, shape, zval=None, outside_pial=False):
        """Computes the primary orientations at every point within the pial surface.

        This algorithm selects the normal/sulcal depth gradient from the closest point.

        :param shape: shape of the resulting array
        :param zval: only process a single horizontal slice
        :param outside_pial: if True also run for voxels outside of the pial surface
        """
        if len(shape) != 3:
            use = shape != 0
            shape = use.shape
        else:
            use = np.ones(shape, dtype='bool')
        dist_pial = grid.signed_distance(self.pial, shape, self.target_affine)
        dist_white = grid.signed_distance(self.white, shape, self.target_affine)

        if not outside_pial:
            use = use & (dist_pial < 0)
        if zval is not None:
            use = use & (np.arange(shape[-1])[None, None, :] == zval)
        res = np.zeros(use.shape + (3, 2))

        idx_get_pial = grid.closest_surface(self.pial, use & (abs(dist_pial) < abs(dist_white)), self.target_affine)
        res[idx_get_pial != -1, :, 0] = self.pial_normal_point.T[idx_get_pial[idx_get_pial != -1]]
        res[idx_get_pial != -1, :, 1] = self.pial_grad_point.T[idx_get_pial[idx_get_pial != -1]]
        idx_get_white = grid.closest_surface(self.white, use & (abs(dist_white) <= abs(dist_pial)), self.target_affine)
        res[idx_get_white != -1, :, 0] = self.white_normal_point.T[idx_get_white[idx_get_white != -1]]
        res[idx_get_white != -1, :, 1] = self.white_grad_point.T[idx_get_white[idx_get_white != -1]]
        return res / np.sqrt(np.sum(res ** 2, -2)[..., None, :])


def align_vector_field(from_field, to_field):
    """Returns a new vector field parallel to `from_field`, which is aligned with `to_field`

    :param from_field: (Nx, Ny, Nz, 3, Nf) original vector field
    :param to_field: (Nx, Ny, Nz, 3, Nf) reference vector field with the correct rough hemisphere
    :return: same as `from_field`, but with every vector flipped that has a negative inner product with `to_field`
    """
    res = from_field.copy()
    _, flip = np.broadcast_arrays(res, (np.sum(from_field * to_field, -2) < 0)[:, :, :, None, :])
    res[flip] *= -1
    return res


def make_perpendicular(radial, tangential):
    """Projects the tangential field, so that it is perpendicular to the radial field.

    :param radial: (Nx, Ny, Nz, 3) array with the radial orientations
    :param tangential: (Nx, Ny, Nz, 3) array with the tangential orientations
    :return: (Nx, Ny, Nz, 3, 3) array with the radial orientations [..., 0],
        the projected tangential orientations [..., 1] and the orientations perpendicular to both [..., 2].
        All output orientations will be normalized.
    """
    proj_tang = tangential - np.sum(radial * tangential, -1)[..., None] * radial
    new_tang = np.cross(radial, proj_tang)
    return np.stack([arr / np.sqrt(np.sum(arr ** 2, -1))[..., None] for arr in [radial, proj_tang, new_tang]], -1)


def run_from_args(args):
    if args.thickness is not None and args.algorithm != 'line':
        raise ValueError("Optional thickness output only available for the 'line algorithm, not for %s" % args.algorithm)
    white = CorticalMesh.read(args.white)
    pial = CorticalMesh.read(args.pial)
    if args.sulcal_depth is None:
        sd = np.zeros(white.nvertices)
    else:
        sd = gifti.read(args.sulcal_depth).darrays[0].data
    img_mask = nibabel.load(args.mask)
    mask = img_mask.get_data()
    if mask.ndim != 3:
        raise ValueError("Input mask should be 3-dimenasional")

    wo = WeightedOrientation(white, pial, sd, 1., img_mask.affine)
    wo._flip_inpr = args.flip_inpr
    rough = wo.closest_vertex_grid(mask, zval=args.zval, outside_pial=args.outside_pial)
    if args.algorithm == 'closest':
        field = rough
    elif args.algorithm == 'line':
        field, thickness = wo.average_line_grid(mask, norient=args.norient, power_dist=args.power_dist, zval=args.zval)
        # line algorithm might fail for voxels on the edge
        replace = ~np.isfinite(thickness[..., 0])
        field[replace] = rough[replace]
        if args.thickness is not None:
            nibabel.Nifti1Image(thickness, affine=img_mask.affine).to_filename(args.thickness)
    elif args.algorithm == 'interp':
        field = wo.average_point_grid(mask, power_dist=args.power_dist, zval=args.zval, outside_pial=args.outside_pial)
    flipped = align_vector_field(field, rough)
    coords = make_perpendicular(flipped[..., 0], flipped[..., 1])

    nibabel.Nifti1Image(coords, affine=img_mask.affine).to_filename(args.output)
