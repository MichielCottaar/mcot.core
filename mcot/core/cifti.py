from nibabel import cifti2
from typing import Tuple, Sequence
from fsl.utils import path
import numpy as np


file_types = [
    (3001, ['.dconn.nii'], (cifti2.BrainModelAxis, cifti2.BrainModelAxis)),
    (3002, ['.dtseries.nii'], (cifti2.SeriesAxis, cifti2.BrainModelAxis)),
    (3003, ['.pconn.nii'], (cifti2.ParcelsAxis, cifti2.ParcelsAxis)),
    (3004, ['.ptseries.nii'], (cifti2.SeriesAxis, cifti2.ParcelsAxis)),
    (3006, ['.dscalar.nii', '.dfan.nii'], (cifti2.ScalarAxis, cifti2.BrainModelAxis)),
    (3007, ['.dlabel.nii'], (cifti2.LabelAxis, cifti2.BrainModelAxis)),
    (3008, ['.pscalar.nii'], (cifti2.ScalarAxis, cifti2.ParcelsAxis)),
    (3009, ['.pdconn.nii'], (cifti2.BrainModelAxis, cifti2.ParcelsAxis)),
    (3010, ['.dpconn.nii'], (cifti2.ParcelsAxis, cifti2.BrainModelAxis)),
    (3011, ['.pconnseries.nii'], (cifti2.ParcelsAxis, cifti2.ParcelsAxis, cifti2.SeriesAxis)),
    (3012, ['.pconnscalar.nii'], (cifti2.ParcelsAxis, cifti2.ParcelsAxis, cifti2.ScalarAxis)),
]


def guess_extension(axes: Tuple[cifti2.Axis]) -> Sequence[str]:
    """
    Guesses the extension based on the CIFTI axes

    :param axes: CIFTI axes describing the rows/columns of a CIFTI file
    :return: tuple of possible file extensions
    """
    for _, extensions, axes_types in file_types:
        if len(axes_types) == len(axes) and all(isinstance(a, at) for a, at in zip(axes, axes_types)):
            return tuple(extensions)
    return ()


def write(filename: str, arr: np.ndarray, axes: Tuple[cifti2.Axis]):
    """
    Writes a CIFTI file guessing the extension of the filename

    :param filename: full filename of basename
    :param arr: array to be stored
    :param axes: CIFTI axes describing the rows/columns of a CIFTI file
    """
    extensions = guess_extension(axes)
    if len(extensions) == 0:
        raise ValueError("No valid extensions found for axes of type {}".format(type(a) for a in axes))
    new_filename = path.addExt(filename, allowedExts=extensions, mustExist=False, defaultExt=extensions[0])
    cifti2.write(new_filename, arr, axes)


def _greyordinate_index(brain_model: cifti2.BrainModelAxis):
    assert (brain_model.name[0] == brain_model.name).all()
    idx = np.zeros(brain_model.size, dtype='i4')
    if not brain_model.surface_mask.all():
        voxel = brain_model.voxel[~brain_model.surface_mask]
        idx[brain_model.volume_mask] = brain_model.volume_shape[0] * (
            brain_model.volume_shape[1] * voxel[:, 2] +
            voxel[:, 1]
        ) + voxel[:, 0]
    idx[brain_model.surface_mask] = -brain_model.vertex[brain_model.surface_mask]
    return idx


def _find_overlap(bm1: cifti2.BrainModelAxis, bm2: cifti2.BrainModelAxis):
    full_idx1 = []
    full_idx2 = []
    as_dict1 = {n: (i, bm) for n, i, bm in bm1.iter_structures()}
    for name, idx2, bm2_part in bm2.iter_structures():
        if name in as_dict1:
            idx1, bm1_part = as_dict1[name]
            _, sub_idx1, sub_idx2 = np.intersect1d(
                    _greyordinate_index(bm1_part),
                    _greyordinate_index(bm2_part),
                    return_indices=True
            )
            sorter = np.argsort(sub_idx1)
            full_idx1.append(idx1.start + sub_idx1[sorter])
            full_idx2.append(idx2.start + sub_idx2[sorter])
    full_idx1 = np.concatenate(full_idx1, 0)
    full_idx2 = np.concatenate(full_idx2, 0)
    assert (len(full_idx1) == 0 and len(full_idx2) == 0) or (bm1[full_idx1] == bm2[full_idx2])
    return full_idx1, full_idx2


def combine(brain_models: Sequence[cifti2.BrainModelAxis]):
    """
    Find the common space of multiple BrainModel axes

    :param brain_models: sequence of brain model axes
    :return: tuple of common brain model and sequence of indices with the common space
    """
    common_bm = brain_models[0]
    for bm in brain_models[1:]:
        common_bm = common_bm[_find_overlap(common_bm, bm)[0]]
    return common_bm, [_find_overlap(bm, common_bm)[0] for bm in brain_models]


def axis_from_hdf5(group: "h5py.Group"):
    """
    Stores the information from an axis in HDF5 group
    """
    name = group.attrs['name']
    if name == 'None':
        return None
    if name == 'Scalar':
        return cifti2.ScalarAxis(np.array(group['name']).astype('U'))
    if name == 'BrainModel':
        nvertices = {str(key): int(value) for key, value in
                     zip(group.attrs['nvertices_keys'], group.attrs['nvertices_values'])}
        return cifti2.BrainModelAxis(
            np.array(group['name']).astype('U'), np.array(group['voxel']),
            np.array(group['vertex']), np.array(group['affine']) if 'affine' in group else None,
            tuple(int(sz) for sz in group['volume_shape']) if 'volume_shape' in group else None,
            nvertices)
    if name == 'Parcels':
        nvertices = {str(key): int(value) for key, value in
                     zip(group.attrs['nvertices_keys'], group.attrs['nvertices_values'])}
        voxels = [np.array(group[f'voxels{idx}']) for idx in range(len(group['name']))]
        vertices = []
        for idx in range(len(group['name'])):
            res = {}
            for name in group.attrs['nvertices_keys']:
                fname = f'vertices{idx}_{name}'
                if fname in group:
                    res[name] = np.array(group[fname])
            vertices.append(res)
        return cifti2.ParcelsAxis(
            np.array(group['name']).astype('S'), voxels, vertices,
            np.array(group['affine']) if 'affine' in group else None,
            tuple(int(sz) for sz in group['volume_shape']) if 'volume_shape' in group else None,
            nvertices)
    if name == 'Series':
        return cifti2.SeriesAxis(
            group.attrs['start'], group.attrs['step'], group.attrs['size'], group.attrs['unit']
        )
    raise ValueError(f"Reading {name} from HDF5 is not currently supported")


def axis_to_hdf5(group: "h5py.Group", axis: cifti2.Axis):
    """
    Stores the information from an axis in HDF5 group
    """
    if axis is None:
        group.attrs['name'] = 'None'
    elif isinstance(axis, cifti2.ScalarAxis):
        group.attrs['name'] = 'Scalar'
        group['name'] = axis.name.astype('S')
    elif isinstance(axis, cifti2.BrainModelAxis):
        group.attrs['name'] = 'BrainModel'
        group['name'] = axis.name.astype('S')
        group['voxel'] = axis.voxel
        group['vertex'] = axis.vertex
        if axis.affine is not None:
            group['affine'] = axis.affine
        if axis.volume_shape is not None:
            group['volume_shape'] = axis.volume_shape
        group.attrs['nvertices_keys'] = list(axis.nvertices.keys())
        group.attrs['nvertices_values'] = list(axis.nvertices.values())
    elif isinstance(axis, cifti2.ParcelsAxis):
        group.attrs['name'] = 'Parcels'
        for idx in range(len(axis)):
            for key, value in axis.vertices[idx].items():
                if key not in axis.nvertices:
                    raise KeyError(f"Defining vertices for undefined surface {key}")
                group[f'vertices{idx}_{key}'] = value
            group[f'voxels{idx}'] = axis.voxels[idx]
        group['affine'] = axis.affine
        group['name'] = axis.name.astype('S')
        group.attrs['nvertices_keys'] = list(axis.nvertices.keys())
        group.attrs['nvertices_values'] = list(axis.nvertices.values())
        group['volume_shape'] = axis.volume_shape
    elif isinstance(axis, cifti2.SeriesAxis):
        group.attrs['name'] = 'Series'
        group.attrs['start'] = axis.start
        group.attrs['size'] = axis.size
        group.attrs['step'] = axis.step
        group.attrs['unit'] = axis.unit
    else:
        raise ValueError(f"storing {axis.__class__.__name__} in HDF5 is not currently supported")


def from_hdf5(group: "h5py.Group") -> Tuple["h5py.Dataset", Sequence[cifti2.Axis]]:
    """
    Reads a CIFTI array from the HDF5 format

    :param group: HDF5 group the data was stored in
    :return: tuple with data array (still on disk) and sequence of axes
    """
    data = group['data']
    axes = [axis_from_hdf5(group[f'axis{idx}']) for idx in range(data.ndim)]
    return data, axes


def to_hdf5(group: "h5py.Group", arr, axes: Sequence[cifti2.Axis], compression='gzip'):
    """
    Store the CIFTI array in an HDF5 format

    :param group: HDF5 group to store the data in (can be top-level HDF5 file)
    :param arr: data array
    :param axes: sequence of axes (optionally None)
    :param compression: which compression to use on the main data array (None, 'gzip', or 'lzf')
    """
    assert len(axes) == arr.ndim
    assert all(ax is None or len(ax) == sz for sz, ax in zip(arr.shape, axes))
    group.create_dataset('data', data=arr, compression=compression)
    for idx, axis in enumerate(axes):
        axis_to_hdf5(group.create_group(f'axis{idx}'), axis)


def empty_hdf5(group: "h5py.Group", axes: Sequence[cifti2.Axis], dtype=float, compression='gzip'):
    """
    Creates a new HDF5 group with an empty dataset

    :param group: HDF5 group
    :param axes: sequence of axes (all have to be defined)
    :param dtype: data type
    :param compression: which compression to use on each chunk
    :return: new array to be filled
    """
    group.create_dataset('data', shape=(len(ax) for ax in axes), dtype=dtype, compression=compression)
    for idx, axis in enumerate(axes):
        axis_to_hdf5(group.create_group(f'axis{idx}'), axis)
    return group['data']


def empty_zarr(group: "zarr.Group", axes: Sequence[cifti2.Axis], dtype=float, compressor='default'):
    """
    Creates a new zarr group with an empty dataset

    :param group: zarr group
    :param axes: sequence of axes (all have to be defined)
    :param dtype: data type
    :param compressor: which compressor to use on each chunk
    :return: new array to be filled
    """
    group.create_dataset('data', shape=(len(ax) for ax in axes), dtype=dtype, compressor=compressor)
    for idx, axis in enumerate(axes):
        axis_to_hdf5(group.create_group(f'axis{idx}'), axis)
    return group['data']
