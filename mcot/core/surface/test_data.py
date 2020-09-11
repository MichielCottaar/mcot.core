from mcot.core.surface import CorticalMesh, Mesh2D, BrainStructure, Cortex, read_HCP
import numpy as np
from scipy import spatial
import os
from operator import xor
from fsl.wrappers import gps, LOAD

fsaverage_directory = os.path.join(os.path.split(__file__)[0], 'fsaverage')

def read_fsaverage_surface():
    """
    Reads the left pial surface from the fsaverage directory
    """
    filename = os.path.join(fsaverage_directory, '100000.L.pial.32k_fs_LR.surf.gii')
    return CorticalMesh.read(filename)


def read_fsaverage_cortex():
    """
    Reads all surfaces in the fsaverage directory
    """
    return read_HCP(fsaverage_directory)


def mesh_to_cortex(mesh: Mesh2D, cortex='left') -> CorticalMesh:
    """
    Converts mesh to cortical surface by defining it as left or right cortex

    :param mesh: surface mesh
    :param cortex: 'left' or 'right', defining which cortex this should represent
    :return: same mesh but as a CorticalMesh
    """
    return CorticalMesh(mesh.vertices, mesh.faces, mesh.flip_normal,
                        anatomy=BrainStructure('cortex', hemisphere=cortex))


def triangle_mesh(ndim=3) -> Mesh2D:
    """
    Draws triangle between (0, 0), (1, 0), (0, 1)

    :param ndim: number of dimensions of the space (triangle will be in the first two)
    :return: mesh
    """
    vertices = np.zeros((ndim, 3))
    vertices[0, 1] = 1
    vertices[1, 2] = 1
    mesh = Mesh2D(
        vertices=vertices,
        faces=[[0], [1], [2]],
    )
    assert mesh.nvertices == 3
    assert mesh.ndim == ndim
    assert mesh.nfaces == 1
    return mesh


def sphere(norient=300) -> Mesh2D:
    """
    Generates a mesh of a unit sphere with `norient` vertices per hemisphere.

    :param norient: number of vertices
    :return: mesh of sphere with radius 1
    """
    points = gps(LOAD, norient, optws=True)

    ch = spatial.ConvexHull(points)
    mesh = Mesh2D(ch.points.T, ch.simplices.T)

    meanp = mesh.vertices[:, mesh.faces].mean(1)
    flip = (meanp * mesh.normal()).sum(0) < 0
    mesh.faces[:2, flip] = mesh.faces[1::-1, flip]
    mesh = Mesh2D(mesh.vertices, mesh.faces)
    assert mesh.nvertices == norient
    assert mesh.ndim == 3
    return mesh


def create_spherical_cortex(norient=300) -> Cortex:
    """
    Creates a cortex with a WM/GM boundary (at radius of 1) and a pial surface (at a radius of 2)

    :param norient: number of vertices on the sphere
    :return: spherical cortex
    """
    white = sphere(norient)
    pial = Mesh2D(white.vertices * 2, white.faces)
    return Cortex([mesh_to_cortex(white), mesh_to_cortex(pial)])


def rectangle(size=(1., 1., 1.), inward_normals=True):
    """generate a rectangle as triangular mesh

    :param size: rectangle size (in arbitrary units)
    :param inward_normals: ensure the normals are pointing inwards (otherwise they point in random directions.
    :return: triangular mesh covering the surface of a rectangle
    :rtype: Mesh2D
    """
    points = np.array(np.broadcast_arrays(np.arange(2)[:, None, None],
                                          np.arange(2)[None, :, None],
                                          np.arange(2)[None, None, :])).reshape((3, 8))
    vertices = np.zeros((3, 12), dtype='i4')
    for ixdim in range(3):
        ixdim_other = list(range(3))
        ixdim_other.remove(ixdim)
        for ixface in range(2):
            use_points = points[ixdim, :] == ixface
            assert np.sum(use_points) == 4
            ixpoint1 = np.where(use_points)[0][:1]
            ixneigh = np.where(use_points & (np.sum(abs(points - points[:, ixpoint1]), 0) == 1))[0]
            ixpoint2 = np.where(use_points & (np.sum(abs(points - points[:, ixpoint1]), 0) == 2))[0]
            if inward_normals:
                offset = points[:, ixneigh] - points[:, ixpoint1]
            assert (np.cross(offset[:, 0], offset[:, 1])[ixdim_other] == 0).all()
            assert (abs(np.cross(offset[:, 0], offset[:, 1])[ixdim]) == 1).all()
            if xor(np.cross(offset[:, 0], offset[:, 1])[ixdim] < 0, ixface == 1):
                ixneigh = ixneigh[::-1]
            vertices[:, ixdim + ixface * 3] = np.append(ixpoint1, ixneigh)
            vertices[:, ixdim + ixface * 3 + 6] = np.append(ixpoint2, ixneigh[::-1])
    mesh = Mesh2D(points * np.array(size)[:, None], vertices)
    assert mesh.nvertices == 8
    assert mesh.nfaces == 12
    assert mesh.ndim == 3
    return mesh
