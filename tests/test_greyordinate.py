from mcot.core import greyordinate
import numpy as np
import nibabel as nib
from fsl.utils.tempdir import tempdir
from mcot.core import write_gifti
from numpy import testing
from pytest import raises
from filecmp import cmp
import os


def surface_greyordinate(arr, surface_mask=None):
    """
    Creates a surface greyordinate file

    :param arr: (..., Nvertices) array
    :param surface_mask: bool mask of the surface mesh (default Nvertices True array)
    """
    if surface_mask is None:
        surface_mask = np.ones(arr.shape[-1], dtype='bool')
    else:
        surface_mask = np.asarray(surface_mask) != 0
    assert arr.shape[-1] == surface_mask.sum()
    return greyordinate.GreyOrdinates(
        arr, greyordinate.BrainModelAxis.from_mask(surface_mask, name='cortex_left')
    )


def nifti_image(shape=()):
    arr = np.zeros((3, 4, 5) + shape)
    arr[1, 1, 1] = 10.
    affine = np.eye(4) * 2
    affine[-1, -1] = 1
    return nib.Nifti1Image(arr, affine=affine)


def gifti_arr(shape=()):
    arr = np.zeros(shape + (30, ))
    arr[..., 1] = 10.
    return arr


def test_nifti():
    with tempdir():
        for shape in (), (2, ), (1, 2):
            img = nifti_image(shape)
            go = greyordinate.GreyOrdinates.from_filename(img)
            assert len(go.brain_model_axis) == 1
            assert go.data.shape == shape + (1, )
            testing.assert_equal(go.volume().dataobj[1, 1, 1], 10.)
            assert np.isfinite(go.volume().get_fdata()).sum() == np.prod(shape)
            testing.assert_equal(go.volume().shape, img.shape)
            testing.assert_equal(go.volume().affine, img.affine)
            with raises(ValueError):
                go.surface('CortexLeft')

            img.to_filename('test.nii.gz')
            go = greyordinate.GreyOrdinates.from_filename('test.nii.gz')
            assert len(go.brain_model_axis) == 1
            assert go.data.shape == shape + (1, )
            testing.assert_equal(go.volume().dataobj[1, 1, 1], 10.)
            assert np.isfinite(go.volume().get_fdata()).sum() == np.prod(shape)
            testing.assert_equal(go.volume().shape, img.shape)
            testing.assert_equal(go.volume().affine, img.affine)
            with raises(ValueError):
                go.surface('CortexLeft')

            go.to_filename('test2.nii.gz')
            cmp('test.nii.gz', 'test2.nii.gz', shallow=False)


def test_gifti():
    with tempdir():
        for shape in (), (2, ), (1, 2):
            arr = gifti_arr(shape)
            write_gifti('test.shape.gii', [arr], brain_structure='CortexLeft')
            go = greyordinate.GreyOrdinates.from_filename('test.shape.gii')

            assert len(go.brain_model_axis) == 1
            assert go.data.shape == shape + (1, )
            new_arr = go.surface('left_cortex')

            testing.assert_equal(new_arr.shape, arr.shape)
            testing.assert_equal(new_arr[..., 1], arr[..., 1])
            assert np.isfinite(new_arr).sum() == np.prod(shape)

            with raises(ValueError):
                go.surface('CortexRight')
            with raises(ValueError):
                go.volume()

            go.to_filename('test2.shape.gii')
            cmp('test.shape.gii', 'test2.shape.gii', shallow=False)


def test_writable():
    with tempdir():
        nifti_image().to_filename('test.nii.gz')
        arr = gifti_arr()
        write_gifti('test.shape.gii', [arr], brain_structure='CortexLeft')

        go = greyordinate.parse_greyordinate('test.nii.gz@test.shape.gii')
        for ext in ('.dscalar.nii', '.hdf5', '.zarr'):
            go.to_filename('data' + ext)
            img = go.from_filename('data' + ext, writable=True)
            img.data[..., 0] = np.pi * 3
            del img
            assert (go.from_filename('data' + ext).data[..., 0] == np.pi * 3).all()


def test_cifti():
    with tempdir():
        nifti_image().to_filename('test.nii.gz')
        arr = gifti_arr()
        write_gifti('test.shape.gii', [arr], brain_structure='CortexLeft')

        go = greyordinate.parse_greyordinate('test.nii.gz@test.shape.gii')
        assert len(go.brain_model_axis) == 2
        go.to_filename('full.dscalar.nii')
        go2 = go.from_filename('full.dscalar.nii')
        testing.assert_equal(go.data[None, :], go2.data)

        assert go.brain_model_axis == go2.brain_model_axis
        assert go2.data.shape == (1, 2)

        with greyordinate.GreyOrdinates.empty_cifti(
                'empty.dconn.nii', (go.brain_model_axis, go.brain_model_axis)) as go:
            arr = np.random.randn(2, 2)
            go.data[:] = arr

        go2 = greyordinate.parse_greyordinate('empty.dconn.nii')
        assert (np.array(go2.data) == arr).all()
        assert go2.brain_model_axis == go.brain_model_axis
        assert go2.other_axes == (go.brain_model_axis, )


def test_hdf5():
    for ext in ('.hdf5', '.zarr'):
        with tempdir():
            nifti_image().to_filename('test.nii.gz')
            arr = gifti_arr()
            write_gifti('test.shape.gii', [arr], brain_structure='CortexLeft')

            go = greyordinate.parse_greyordinate('test.nii.gz@test.shape.gii')
            assert len(go.brain_model_axis) == 2
            go.to_filename('full' + ext)
            go2 = go.from_filename('full' + ext)
            testing.assert_equal(go.data, go2.data)

            assert go.brain_model_axis == go2.brain_model_axis
            assert go2.data.shape == (2, )

            func = greyordinate.GreyOrdinates.empty_zarr if ext == '.zarr' else greyordinate.GreyOrdinates.empty_hdf5
            with func('empty' + ext, (go.brain_model_axis, go.brain_model_axis)) as go:
                arr = np.random.randn(2, 2)
                go.data[:] = arr

            go2 = greyordinate.parse_greyordinate('empty' + ext)
            assert (np.array(go2.data) == arr).all()
            assert go2.brain_model_axis == go.brain_model_axis
            assert go2.other_axes == (go.brain_model_axis, )
            print(ext)
            print(go2.data.chunks)
            if ext == '.zarr':
                print(go2.data.compressor)


def test_concatenate():
    data = np.random.randn(7, 5)
    first_batch = np.array([True, True, False, False, False], dtype='bool')
    part1 = surface_greyordinate(data[:, first_batch], first_batch)
    part2 = surface_greyordinate(data[:, ~first_batch], ~first_batch)
    concat = greyordinate.concatenate([part1, part2], axis=-1)
    assert (concat.data == data).all()
    assert concat.brain_model_axis == surface_greyordinate(data).brain_model_axis


def test_no_write_on_error():
    """
    Tests that no file is writing when an error is produced during the writing
    """
    for ext in ('.hdf5', '.dconn.nii', '.zarr'):
        with tempdir():
            axis = nib.cifti2.BrainModelAxis.from_mask(np.ones((3, 3, 2)), affine=np.eye(4))
            fn = 'empty' + ext
            try:
                with greyordinate.GreyOrdinates.empty(fn, (axis, axis)) as go:
                    go.data[()] = 1.
                    1/0
            except ZeroDivisionError:
                pass
            else:
                assert False  # ZeroDivisionError was incorrectly caught
            assert not os.path.exists(fn)

            try:
                with greyordinate.GreyOrdinates.empty(fn, (axis, axis)) as go:
                    1/0
            except ZeroDivisionError:
                pass
            else:
                assert False  # ZeroDivisionError was incorrectly caught
            assert not os.path.exists(fn)

            with greyordinate.GreyOrdinates.empty(fn, (axis, axis)) as go:
                go.data[()] = 2.
            assert os.path.exists(fn)


