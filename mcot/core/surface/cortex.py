"""Defines cortical layers/cortices defined by 2 or more cortical meshes.

`CorticalLayer` always consists of exactly two meshes (giving the full extent of the layer), while `Cortex` can have 2 or more describing the full cortex
"""
from .cortical_mesh import CorticalMesh
import numpy as np
import glob
import os
import tempfile
import subprocess
import nibabel as nib
from . import grid
import numba


class Cortex(object):
    """
    A cortex defined by 2 or more cortical meshes (sorted from inner to outer).

    The first one will generally be the white/gray matter boundary and the last one the pial surface with arbitrarily many cortical meshes defined in between.

    A ValueError will be raised if the cortical meshes describe different structures
    """
    def __init__(self, surfaces):
        """
        Creates a new cortex from the provided cortical meshes

        :param surfaces: 2 or more cortical meshes describing the same brain structure (in the same hemisphere if appropriate)
        :type surfaces: Iterable[CorticalMesh]
        :raises: ValueError
        """
        self.surfaces = tuple(surfaces)
        for surf in self.surfaces[1:]:
            if isinstance(self.surfaces[0], CorticalMesh):
                if surf.anatomy.primary != self.primary and self.primary is not None:
                    raise ValueError("Not all cortical meshes provided describe the same primary structure")
                if surf.anatomy.hemisphere != self.hemisphere and self.hemisphere is not None:
                    raise ValueError("Not all cortical meshes provided describe the same hemisphere")
            if surf.nvertices != self.nvertices:
                raise ValueError("The cortical meshes have different number of vertices")
            if (surf.faces != self.surfaces[0].faces).any():
                raise ValueError("The cortical meshes have different faces between the vertices")
        self.layers = tuple(CorticalLayer(lower, upper) for lower, upper in zip(self.surfaces[:-1], self.surfaces[1:]))

    @property
    def primary(self, ):
        """
        Name of the structure described by the cortical meshes
        """
        return self.surfaces[0].anatomy.primary

    @property
    def hemisphere(self, ):
        """
        Hemisphere containing the structure described by the cortical meshes
        """
        return self.surfaces[0].anatomy.hemisphere

    @property
    def nvertices(self, ):
        """
        Number of vertices in the cortical meshes
        """
        return self.surfaces[0].nvertices

    @property
    def nfaces(self, ):
        """
        Number of faces in the cortical meshes
        """
        return self.surfaces[0].nfaces

    def wedge_volume(self, use_wb=False, atpoint=False):
        """
        Computes the total wedge volume associated with every mesh triangle.

        Uses the algorithm from http://dx.doi.org/10.1101/074666 (Winkler et al, 2016)

        :param use_wb: use workbench to compute the wedge volume if True (about 1.5 times larger than my implementation)
        :param atpoint: if True assigns the wedge volume to the vertices rather than the faces
        :return: (Nfaces, )-array with the wedge volume for every surface element
        """
        return np.sum([layer.wedge_volume(use_wb=use_wb, atpoint=atpoint) for layer in self.layers], 0)

    def volume(self, use_wb=False):
        """
        Computes the total volume of the cortex

        :param use_wb: use workbench to compute the wedge volume if True (about 1.5 times larger than my implementation)
        :return: float with the total volume
        """
        return np.sum(self.wedge_volume())

    def segment(self, shape, affine=None):
        """
        Segments a volume based on the cortical meshes.

        :param shape: (Nx, Ny, Nz) tuple giving the shape of the resulting array
        :param affine: (4 x 4) array giving the transformation from voxel to mm coordinates
        :return: (Nx x Ny x Nz) array, that is 0 outside the brain, and increases by 1 for every surface crossed into the brain (so WM will be len(self.surfaces)).
        """
        final = np.zeros(shape, dtype='i4')
        for surf in self.surfaces:
            dist = grid.signed_distance(surf, shape, affine)
            final[dist < 0] += 1
        return final

    def gyral_bias(self, curvature, use_cortex=None, zero_curv_threshold=0.05):
        """
        Computes the theoretical gyral bias based on a script originally derived by David van Essen

        Converted from a matlab script from Stamatios Sotiropoulos.

        :param curvature: (N, ) array describing the curvature of the surface at every point
        :param use_cortex: which part of the cortex to consider (e.g. remove the medial wall), default: not masked
        :param zero_curv_threshold: threshold below which the surface is considered to be flat
        :return: (Nfaces, )-array with the relative fiber density expected to cross each surface element
        """
        wedge_volume = self.wedge_volume()
        vertex_area = self.surfaces[0].size_vertices()
        ratio = wedge_volume / vertex_area
        if use_cortex is None:
            use_cortex = np.ones(ratio.size, dtype='bool')
        isflat = abs(curvature) < zero_curv_threshold
        mean_flat = np.mean(ratio[use_cortex & isflat])
        norm_ratio = ratio / mean_flat
        norm_ratio[~use_cortex] = np.nan
        return norm_ratio

    def distance(self, mask, affine=None):
        """
        Computes for each non-zero voxel in `mask` the closest vertex

        Each vertex is represented by the line elements connecting that surface between the different cortical layers.
        The distance to each voxel center is calculated based on the distance to these line element.

        :param mask: (Nx, Ny, Nz) array, which is non-zero for all the voxels that will be processed
        :param affine: transformation from the voxel space of `mask` to the mm-space in which the cortex has been defined
        :return: labeled array with 'distance', 'closest', 'valid':

            - `valid`: boolean, which is True wherever the input `mask` is non-zero
            - `distance`: contains the distance from the closest vertex (np.inf wherever the input `mask` is zero)
            - `closest`: contains the index of the closest vertex (-1 wherever the input `mask` is zero)
        """
        if affine is None:
            affine = np.eye(4)
        start = np.concatenate([layer.lower.vertices.T for layer in self.layers], 0)
        end = np.concatenate([layer.upper.vertices.T for layer in self.layers], 0)

        res = np.zeros(mask.shape, dtype=[('valid', 'bool'), ('closest', 'i4'), ('distance', 'f8')])
        res['valid'] = mask != 0
        res['closest'] = -1
        res['distance'] = np.inf
        pos_vox = np.stack(np.where(res['valid']), -1)
        pos_mm = grid.affine_mult(affine, pos_vox)
        for idxpos, pos in enumerate(pos_mm):
            dist, closest = _closest_line(pos, start, end)
            index = tuple(pos_vox[idxpos])
            res[index]['distance'] = dist
            res[index]['closest'] = closest % self.nvertices
        return res


class CorticalLayer(object):
    """
    Represents a cortical layer bounded by a `lower` and `upper` mesh
    """
    def __init__(self, lower, upper):
        """
        Defines a cortical layer between a lower and upper surface.

        :param lower: lower boundary of the cortical layer
        :type lower: CorticalMesh
        :param upper: upper boundary of the cortical layer
        :type upper: CorticalMesh
        """
        self.lower = lower
        self.upper = upper
        if isinstance(self.upper, CorticalMesh):
            if self.upper.anatomy.primary != self.lower.anatomy.primary and self.lower.anatomy.primary is not None:
                raise ValueError("Not all cortical meshes provided describe the same primary structure")
            if self.upper.anatomy.hemisphere != self.lower.anatomy.hemisphere and self.lower.anatomy.hemisphere is not None:
                raise ValueError("Not all cortical meshes provided describe the same hemisphere")
        if self.upper.nvertices != self.lower.nvertices:
            raise ValueError("The cortical meshes have different number of vertices")
        if (self.upper.faces != self.lower.faces).any():
            raise ValueError("The cortical meshes have different faces between the vertices")

    @property
    def structure(self, ):
        """
        Name of the structure described by the cortical meshes.
        """
        return self.lower.anatomy.primary

    @property
    def hemisphere(self, ):
        """
        Hemisphere containing the structure described by the cortical meshes
        """
        return self.lower.anatomy.hemisphere

    @property
    def nfaces(self, ):
        return self.lower.nfaces

    @property
    def nvertices(self, ):
        return self.lower.nvertices

    def wedge_volume(self, use_wb=False, atpoint=False):
        """Computes the wedge volume covered by every triangle between the two surfaces.

        Uses the algorithm from http://dx.doi.org/10.1101/074666 (Winkler et al, 2016)

        :param use_wb: use workbench to compute the volume (about 1.5 times larger than my implementation).
        :param atpoint: if True assigns the wedge volume to the vertices rather than the faces
        :return: (Nfaces, )-array with the wedge volume for every triangle in the mesh
        """
        if (self.lower.faces != self.upper.faces).any():
            raise ValueError("lower and upper surface of CorticalLayer should have the same mesh")
        if use_wb:
            if not atpoint:
                raise ValueError("Workbench can only compute wedge volumes at the surfaces")
            surf1 = self.lower.as_temp_file()
            surf2 = self.upper.as_temp_file()
            file = tempfile.NamedTemporaryFile(suffix='.shape.gii', delete=False)
            file.close()
            try:
                subprocess.call(('wb_command -surface-wedge-volume %s %s %s' % (surf1, surf2, file.name)).split())
                res = nib.load(file.name).darrays[0].data
            finally:
                os.remove(surf1)
                os.remove(surf2)
                os.remove(file.name)
            return res
        ref = self.upper.vertices[:, self.upper.faces[0, :]]
        def getp(low, index):
            surf = (self.lower if low else self.upper)
            return surf.vertices[:, surf.faces[index, :]] - ref

        tetra1 = np.array([getp(True, 0), getp(True, 1), getp(True, 2)])
        tetra2 = np.array([getp(False, 1), getp(False, 2), getp(True, 1)])
        tetra3 = np.array([getp(False, 2), getp(True, 1), getp(True, 2)])

        volume = np.zeros(self.nfaces)
        for tetra in (tetra1, tetra2, tetra3):
            volume += abs(np.sum(tetra[0] * np.cross(tetra[1], tetra[2], axis=0), 0)) / 6
        if not atpoint:
            return volume
        return np.bincount(self.lower.faces.flatten(), np.array([volume] * 3).flatten()) / 3

    def volume(self, ):
        """Computes the total volume between the two surfaces.
        """
        return np.sum(self.wedge_volume())


def read_HCP(fsaverage_directory):
    """Reads the cortical meshes from an `fsaverage_directory`.

    :param fsaverage_directory: directory containing the GIFTI .surf.gii files describing the freesurfer surfaces.
    :return: tuple of (right Cortex, left Cortex). Each cortex will contain the white/gray matter boudnary, the midthickness (if available), and the pial surface
    """
    if not os.path.isdir(fsaverage_directory):
        raise IOError('Directory %s not found' % fsaverage_directory)
    cortices = []
    for hemisphere in ['R', 'L']:
        surfaces = []
        for surf_name in ['white', 'midthickness', 'pial']:
            filenames = glob.glob(os.path.join(fsaverage_directory, '*.%s.%s.*.surf.gii' % (hemisphere, surf_name)))
            if len(filenames) == 0:
                if surf_name == 'midthickness':
                    continue
                else:
                    raise IOError("%s surface not found for %s hemisphere in %s" % (
                        surf_name, hemisphere, fsaverage_directory))
            surfaces.append(CorticalMesh.read(filenames[0]))
        cortices.append(Cortex(surfaces))
    return cortices


@numba.jit(nopython=True)
def _closest_line(pos, start, end):
    """Finds the closest line element to the point

    This is a helper function used by Cortex.distance

    :param pos: (L, N) array giving the position of the point in N-dimensional space.
    :param start: (M, N) array giving the start of the M line elements.
    :param end: (M, N) array giving the end of the M line elements.
    """
    dist_min = np.inf
    idx_min = -1
    for idx_line in range(start.shape[0]):
        ndim = pos.size
        nominator = 0
        denominator = 0
        for ixdim in range(ndim):
            nominator += (end[idx_line, ixdim] - start[idx_line, ixdim]) * (pos[ixdim] - start[idx_line, ixdim])
            denominator += (end[idx_line, ixdim] - start[idx_line, ixdim] + 1e-10) ** 2
        on_line = nominator / denominator
        elem_dist = 0
        for ixdim in range(ndim):
            if on_line < 0:
                ref_pos = start[idx_line, ixdim]
            elif on_line > 1:
                ref_pos = end[idx_line, ixdim]
            else:
                ref_pos = start[idx_line, ixdim] * (1 - on_line) + end[idx_line, ixdim] * on_line
            elem_dist += (pos[ixdim] - ref_pos) ** 2
        if elem_dist < dist_min:
            dist_min = elem_dist
            idx_min = idx_line
    return np.sqrt(dist_min), idx_min
