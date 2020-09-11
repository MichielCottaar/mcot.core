#!/usr/bin/env python
from mcot.core.surface.test_data import rectangle, fsaverage_directory
import numpy as np
import tempfile
import subprocess
import os
from mcot.core.surface import mesh
import shutil
import pytest
from numpy.testing import assert_allclose, assert_array_equal
import nibabel as nib


def test_normals():
    size = [2., 2.52, 1.3321]
    mesh = rectangle(size)
    norm = mesh.normal()
    assert norm.shape == (3, mesh.nfaces)
    assert np.sum(norm) == 0
    assert (np.sum(norm ** 2, 0) == 1).all()
    for ixdim in range(3):
        for ixface in range(2):
            use_vertex = np.all(mesh.vertices[ixdim, :][mesh.faces] == ixface * size[ixdim], 0)
            assert np.sum(use_vertex) == 2
            if ixface == 0:
                assert (norm[ixdim, use_vertex] == 1).all()
            else:
                assert (norm[ixdim, use_vertex] == -1).all()
    assert mesh.normal(atpoint=True).shape == (3, mesh.nvertices)
    assert np.amax(abs(np.sum(mesh.normal(atpoint=True) ** 2, 0) - 1)) < 1e-5


def test_point_conn_graph():
    mesh = rectangle(np.ones(3))
    conn_pos = (mesh.graph_connection_point().T * mesh.vertices.T).T
    assert (np.unique(conn_pos) == np.arange(4)).all()
    assert (np.sum((conn_pos == 0) | (conn_pos == 3), 0) == 1).all()

    assert mesh.graph_connection_point('i4').dtype == 'i4'
    assert mesh.graph_connection_point('f4').dtype == 'f4'
    assert mesh.graph_connection_point('bool').dtype == 'bool'

    assert (mesh.graph_connection_point().sum(0) == 3).all()


def test_affine():
    mesh = rectangle(np.ones(3))
    new_mesh = mesh.apply_affine(np.eye(4) * 2)
    assert (new_mesh.vertices == rectangle(np.ones(3) / 2).vertices).all()
    new_mesh = mesh.apply_affine(np.eye(4) / 3)
    assert (new_mesh.vertices == rectangle(np.ones(3) * 3).vertices).all()


def test_volume():
    mesh = rectangle([2., 3., 5.])
    assert mesh.volume() == -30.

    mesh.faces = mesh.faces[::-1, :]
    assert mesh.volume() == 30.

    assert_allclose(mesh.inflate(volume=10).volume(), 40.)
    assert mesh.inflate(shift=1).volume() > 30.
    assert mesh.inflate(shift=-0.1).volume() < 30.


@pytest.mark.skipif(shutil.which('wb_command') is None,
                    reason='Requires wb_command installed')
def test_mesh_size():
    file, metric_filename = tempfile.mkstemp('.shape.gii')
    surf_filename = os.path.join(fsaverage_directory, '100000.L.pial.32k_fs_LR.surf.gii')
    subprocess.call('wb_command -surface-vertex-areas %s %s' % (surf_filename, metric_filename), shell=True)
    wb_data = nib.load(metric_filename).darrays[0].data
    py_data = mesh.Mesh2D.read(surf_filename).size_vertices()
    assert np.amax(abs(py_data - wb_data)) < 1e-3


def test_single_size():
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]]).T + 1.3
    faces = np.array([[0, 1, 2]]).T
    single_surf = mesh.Mesh2D(vertices, faces)
    assert single_surf.ndim == 3
    assert single_surf.nvertices == 3
    assert single_surf.nfaces == 1
    assert_allclose(single_surf.size_faces(), 0.5)
    assert_allclose(single_surf.size_vertices(), 0.5 / 3.)

    single_surf = mesh.Mesh2D(vertices[:2, :], faces)
    assert single_surf.ndim == 2
    assert single_surf.nvertices == 3
    assert single_surf.nfaces == 1
    assert_allclose(single_surf.size_faces(), 0.5)
    assert_allclose(single_surf.size_vertices(), 0.5 / 3.)


def test_getitem():
    pial = mesh.Mesh2D.read(os.path.join(fsaverage_directory, '100000.L.pial.32k_fs_LR.surf.gii'))

    new_pial = pial[np.ones(pial.nvertices, dtype='bool')]
    assert_array_equal(new_pial.vertices, pial.vertices)
    assert_array_equal(new_pial.faces, pial.faces)

    empty = pial[np.zeros(pial.nvertices, dtype='bool')]
    assert empty.nvertices == 0
    assert empty.nfaces == 0

    new_pial = pial[np.arange(pial.nvertices, dtype=int)]
    assert_array_equal(new_pial.vertices, pial.vertices)
    assert_array_equal(new_pial.faces, pial.faces)

    flipped = pial[np.arange(pial.nvertices, dtype=int)[::-1]]
    assert_array_equal(flipped.vertices[:, ::-1], pial.vertices)
    assert_array_equal(pial.nvertices - flipped.faces - 1, pial.faces)
