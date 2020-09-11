from mcot.core import cifti
from nibabel import cifti2
import numpy as np
import h5py
import io
import zarr


def get_axes():
    bms = [
        cifti2.BrainModelAxis.from_mask(np.ones(5, dtype='bool'), name='cortex_left'),
        cifti2.BrainModelAxis.from_mask(np.ones((2, 3, 2), dtype='bool'), affine=np.eye(4)),
    ]

    return [
               cifti2.ParcelsAxis.from_brain_models([('surf', bms[0]), ('vol', bms[1]), ('comb', bms[0] + bms[1])]),
               cifti2.ScalarAxis(['a', 'b', 'c']),
               cifti2.SeriesAxis(start=0.3, step=0.1, size=11, unit='HERTZ'),
           ] + bms


def test_axis_hdf5():
    for ax in get_axes():
        bytes_io = io.BytesIO()
        with h5py.File(bytes_io, 'w') as f:
            cifti.axis_to_hdf5(f, ax)
        with h5py.File(bytes_io, 'r') as f:
            new_ax = cifti.axis_from_hdf5(f)
        assert new_ax == ax

        group = zarr.group()
        cifti.axis_to_hdf5(group, ax)
        new_ax = cifti.axis_from_hdf5(group)
        assert new_ax == ax


def test_arr_hdf5():
    for ax1 in get_axes():
        for ax2 in get_axes():
            arr = np.random.randn(len(ax1), len(ax2))
            bytes_io = io.BytesIO()
            with h5py.File(bytes_io, 'w') as f:
                cifti.to_hdf5(f, arr, (ax1, ax2))
            with h5py.File(bytes_io, 'r') as f:
                new_arr, (new_ax1, new_ax2) = cifti.from_hdf5(f)
                data = np.array(new_arr)
            assert (arr == data).all()
            assert ax1 == new_ax1
            assert ax2 == new_ax2

            group = zarr.group()
            cifti.to_hdf5(group, arr, (ax1, ax2))
            new_arr, (new_ax1, new_ax2) = cifti.from_hdf5(group)
            assert (arr == data).all()
            assert ax1 == new_ax1
            assert ax2 == new_ax2
