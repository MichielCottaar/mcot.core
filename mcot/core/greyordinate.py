from nibabel.cifti2.cifti2_axes import BrainModelAxis, Axis, ScalarAxis
from nibabel.cifti2 import Cifti2Image, Cifti2Header
from typing import Sequence
import numpy as np
import nibabel as nib
from nibabel.gifti import GiftiImage
from mcot.core.surface import cortical_mesh
import argparse
from fsl.utils.path import hasExt
from ._write_gifti import write_gifti
from . import cifti
from numpy.lib.stride_tricks import as_strided
from contextlib import contextmanager
import dask.array
import os
import shutil


class GreyOrdinates(object):
    """
    Represents data on voxels or vertices
    """

    def __init__(self, data, brain_model_axis: BrainModelAxis, other_axes: Sequence[Axis]=None, parent_file=None):
        """
        Defines a new dataset in greyordinate space

        :param data: (..., N) array for N greyordinates
        :param brain_model_axis: CIFTI axis describing the greyordinate space
        :param other_axes: sequence of CIFTI axes describing the other dimensions
        :param parent_file: file in which the dataset has been stored
        """
        self.data = data
        if data.shape[-1] != len(brain_model_axis):
            raise ValueError("Last axis of data does not match number of greyordinates")
        self.brain_model_axis = brain_model_axis
        if other_axes is not None:
            if len(other_axes) != self.data.ndim - 1:
                raise ValueError("Number of axis does not match dimensionality of the data")
            if tuple(len(ax) for ax in other_axes) != self.data.shape[:-1]:
                raise ValueError("Size of other axes does not match data size")
        self.other_axes = None if other_axes is None else tuple(other_axes)
        self.parent_file = parent_file

    def volume(self, ):
        """
        Get the volumetric data as a Nifti1Image
        """
        if self.brain_model_axis.volume_mask.sum() == 0:
            raise ValueError(f"Can not create volume without voxels in {self}")
        data = np.full(self.brain_model_axis.volume_shape + self.data.shape[:-1], np.nan,
                       dtype=self.data.dtype)
        voxels = self.brain_model_axis.voxel[self.brain_model_axis.volume_mask]
        data[tuple(voxels.T)] = np.transpose(self.data, (-1, ) + tuple(range(self.data.ndim - 1)))[self.brain_model_axis.volume_mask]
        return nib.Nifti1Image(data, affine=self.brain_model_axis.affine)

    def surface(self, anatomy, fill=np.nan, partial=False):
        """
        Gets a specific surface

        :param anatomy: BrainStructure or string like 'CortexLeft' or 'CortexRight'
        :param fill: which value to fill the array with if not all vertices are defined
        :param partial: only return the part of the surface defined in the greyordinate file (ignores `fill` if set)
        :return:
            - if not partial: (..., n_vertices) array
            - if partial: (N, ) int array with indices on the surface included in (..., N) array
        """
        if isinstance(anatomy, str):
            anatomy = cortical_mesh.BrainStructure.from_string(anatomy, issurface=True)
        if anatomy.cifti not in self.brain_model_axis.name:
            raise ValueError(f"No surface data for {anatomy.cifti} found")
        slc, bm = None, None
        arr = np.full(self.data.shape[:-1] + (self.brain_model_axis.nvertices[anatomy.cifti],), fill,
                      dtype=self.data.dtype)
        for name, slc_try, bm_try in self.brain_model_axis.iter_structures():
            if name == anatomy.cifti:
                if partial:
                    if bm is not None:
                        raise ValueError(f"Surface {anatomy} does not form a contiguous block")
                    slc, bm = slc_try, bm_try
                else:
                    arr[..., bm_try.vertex] = self.data[..., slc_try]
        if not partial:
            return arr
        else:
            return bm.vertex, self.data[..., slc]

    def to_hdf5(self, group, compression='gzip'):
        """
        Stores the image in the HDF5 group
        """
        other_axes = (None, ) * (self.data.ndim - 1) if self.other_axes is None else self.other_axes
        cifti.to_hdf5(group, self.data, other_axes + (self.brain_model_axis, ), compression=compression)

    @classmethod
    def from_hdf5(cls, group):
        """
        Retrieves data from HDF5 group
        """
        data, axes = cifti.from_hdf5(group)
        return cls(data, axes[-1], axes[:-1])

    @classmethod
    @contextmanager
    def empty_hdf5(cls, filename, axes, dtype=float):
        """
        Creates an empty greyordinate object based on the axes

        Data will be stored on disk in an HDF5 file

        :param filename: filename to store the HDF5 file in or HDF5 group
        :param axes: cifti2 axes
        :param dtype: data type of array
        :return: Greyordinate object where the data is stored in the new HDF5 file
        """
        import h5py
        if isinstance(filename, str):
            group = h5py.File(filename, 'w')
            to_close = True
        else:
            group = filename
            to_close = False
        data = cifti.empty_hdf5(group, axes, dtype)
        try:
            yield cls(data, axes[-1], axes[:-1], parent_file=group)
        except:
            if to_close:
                os.remove(filename)
            raise
        if to_close:
            group.close()

    @classmethod
    @contextmanager
    def empty_zarr(cls, filename, axes, dtype=float):
        """
        Creates an empty greyordinate object based on the axes

        Data will be stored on disk in an HDF5 file

        :param filename: filename to store the HDF5 file in or HDF5 group
        :param axes: cifti2 axes
        :param dtype: data type of array
        :return: Greyordinate object where the data is stored in the new HDF5 file
        """
        import zarr
        if isinstance(filename, str):
            group = zarr.group(filename, 'w')
        else:
            group = filename
        data = cifti.empty_zarr(group, axes, dtype)
        try:
            yield cls(data, axes[-1], axes[:-1], parent_file=group)
        except:
            to_delete = filename if group is not filename else group.path
            if os.path.isdir(to_delete):
                shutil.rmtree(to_delete)
            raise

    def to_cifti(self, other_axes=None):
        """
        Create a CIFTI image from the data

        :param other_axes: defines the axes besides the greyordinate one
        :return: nibabel CIFTI image
        """
        if other_axes is None:
            other_axes = self.other_axes
            if other_axes is None:
                if self.data.ndim != 1:
                    raise ValueError("Can not store to CIFTI without defining what is stored along the other dimensions")
                other_axes = []
        if other_axes is not None:
            if len(other_axes) != self.data.ndim - 1:
                raise ValueError("Number of axis does not match dimensionality of the data")
            if tuple(len(ax) for ax in other_axes) != self.data.shape[:-1]:
                raise ValueError("Size of other axes does not match data size")

        data = self.data
        if data.ndim == 1:
            data = data[None, :]
            other_axes = [ScalarAxis(['default'])]

        return Cifti2Image(
                data,
                header=Cifti2Header.from_axes(list(other_axes) + [self.brain_model_axis])
        )

    @classmethod
    def from_cifti(cls, filename, writable=False):
        """
        Creates new greyordinate object from dense CIFTI file

        :param filename: CIFTI filename
        :param writable: if True, opens data array in writable mode
        """
        if isinstance(filename, str):
            img: Cifti2Image = nib.load(filename)
        else:
            img = filename
        if writable:
            data = np.memmap(filename, img.dataobj.dtype, mode='r+',
                             offset=img.dataobj.offset, shape=img.shape, order='F')
        else:
            data = np.asanyarray(img.dataobj)
        axes = [img.header.get_axis(idx) for idx in range(data.ndim)]
        if not isinstance(axes[-1], BrainModelAxis):
            raise ValueError("Last axis of dense CIFTI file should be a BrainModelAxis")
        return GreyOrdinates(data, axes[-1], axes[:-1])

    @classmethod
    @contextmanager
    def empty_cifti(cls, filename, axes, dtype=float):
        """
        Creates an empty greyordinate object based on the axes

        Data will be stored on disk in CIFTI format

        :param filename: filename to store the CIFTI file in
        :param axes: cifti2 axes
        :param dtype: data type of array
        :return: Greyordinate object where the data is stored in the new HDF5 file
        """
        hdr = cifti.cifti2.Cifti2Header.from_axes(axes)
        data = np.zeros(1, dtype=dtype)
        shape = tuple(len(ax) for ax in axes)
        data_shaped = as_strided(data, shape=shape, strides=tuple(0 for _ in axes),
                                 writeable=False)
        cifti.cifti2.Cifti2Image(data_shaped, header=hdr).to_filename(filename)
        go = cls.from_cifti(filename, writable=True)
        try:
            yield go
        except:
            os.remove(filename)
            raise
        go.data.flush()
        del go

    @classmethod
    def from_gifti(cls, filename, mask_values=(0, np.nan)):
        """
        Creates a new greyordinate object from a GIFTI file

        :param filename: GIFTI filename
        :param mask_values: values to mask out
        :return: greyordinate object representing the unmasked vertices
        """
        if isinstance(filename, str):
            img = nib.load(filename)
        else:
            img = filename
        datasets = [darr.data for darr in img.darrays]
        if len(datasets) == 1:
            data = datasets[0]
        else:
            data = np.concatenate(
                    [np.atleast_2d(d) for d in datasets], axis=0
            )
        mask = np.ones(data.shape, dtype='bool')
        for value in mask_values:
            if value is np.nan:
                mask &= ~np.isnan(data)
            else:
                mask &= ~(data == value)
        while mask.ndim > 1:
            mask = mask.any(0)

        anatomy = cortical_mesh.get_brain_structure(img)

        bm_axes = BrainModelAxis.from_mask(mask, name=anatomy.cifti)
        return GreyOrdinates(data[..., mask], bm_axes)

    @classmethod
    def from_nifti(cls, filename, mask_values=(np.nan, 0)):
        """
        Creates a new greyordinate object from a NIFTI file

        :param filename: NIFTI filename
        :param mask_values: which values to mask out
        :return: greyordinate object representing the unmasked voxels
        """
        if isinstance(filename, str):
            img = nib.load(filename)
        else:
            img = filename
        data = img.get_fdata()

        mask = np.ones(data.shape, dtype='bool')
        for value in mask_values:
            if value is np.nan:
                mask &= ~np.isnan(data)
            else:
                mask &= ~(data == value)
        while mask.ndim > 3:
            mask = mask.any(-1)

        inverted_data = np.transpose(data[mask], tuple(range(1, data.ndim - 2)) + (0, ))
        bm_axes = BrainModelAxis.from_mask(mask, affine=img.affine)
        return GreyOrdinates(inverted_data, bm_axes)

    def __add__(self, other):
        """
        Adds the overlapping part of the arrays

        Only voxels/vertices in common between the greyordinate spaces are added
        """
        if not isinstance(other, GreyOrdinates):
            return NotImplemented

        if self.other_axes is not None:
            if other.other_axes is not None:
                if self.other_axes != other.other_axes:
                    raise ValueError("can not concatenate greyordinates when non-brain-model axes do not match")
            other_axes = self.other_axes
        else:
            other_axes = other.other_axes

        new_bm, slices = cifti.combine([self.brain_model_axis, other.brain_model_axis])

        return GreyOrdinates(
                data=self.data[..., slices[0]] + other.data[..., slices[1]],
                brain_model_axis=new_bm,
                other_axes=other_axes,
        )

    @classmethod
    @contextmanager
    def empty(cls, filename, axes, dtype=float):
        """
        Creates an empty file to store the greyordinates with the type determined by the extension:

        - .nii: CIFTI file
        - .h5/hdf5/he2/he5: HDF5 file representing CIFTI data
        - .zarr: zarr file representing CIFTI data

        :param filename: target filename
        :param axes: cifti2 axes
        :param dtype: data type of array
        :return: Greyordinate object where CIFTI data can be stored
        """
        formats = {
            'zarr': ('.zarr',),
            'hdf5': ('.hdf5', '.h5', '.he2', '.he5'),
            'cifti': ('.nii',),
        }
        for format, exts in formats.items():
            if hasExt(filename, exts):
                break
        else:
            raise ValueError(f"Extension of {filename} not recognised as CIFTI, HDF5 or zarr file")
        with getattr(cls, f'empty_{format}')(filename, axes, dtype) as go:
            yield go

    @classmethod
    def from_filename(cls, filename, mask_values=(0, np.nan), writable=False):
        """
        Reads greyordinate data from the given file

        File can be:

        - NIFTI mask
        - GIFTI mask
        - CIFTI file
        - HDF5 file representing CIFTI data
        - zarr file representing CIFTI data

        :param filename: input filename
        :param mask_values: which values are outside of the mask for NIFTI or GIFTI input
        :param writable: allow to write to disk
        :return: greyordinates object
        """
        try:
            import h5py
        except ImportError:
            h5py = False
        try:
            import zarr
        except ImportError:
            zarr = False
        if isinstance(filename, str):
            try:
                if not h5py:
                    raise OSError()
                img = h5py.File(filename, 'r+' if writable else 'r')
            except OSError:
                try:
                    if not zarr:
                        raise ValueError()
                    img = zarr.open(filename, mode='r+' if writable else 'r')
                except ValueError:
                    img = nib.load(filename)
        else:
            img = filename

        if h5py and isinstance(img, h5py.Group):
            return cls.from_hdf5(img)
        if zarr and isinstance(img, zarr.Group):
            return cls.from_hdf5(img)
        if isinstance(img, nib.Nifti1Image):
            if writable:
                raise ValueError("Can not open NIFTI file in writable mode")
            return cls.from_nifti(filename, mask_values)
        if isinstance(img, Cifti2Image):
            return cls.from_cifti(filename, writable=writable)
        if isinstance(img, GiftiImage):
            if writable:
                raise ValueError("Can not open GIFTI file in writable mode")
            return cls.from_gifti(filename, mask_values)
        raise ValueError(f"I do not know how to convert {type(img)} into greyordinates (from {filename})")

    def to_filename(self, filename):
        """
        Stores the greyordinate data to the given filename.

        Type of storage is determined by the extension of the filename:

        - .dscalar/dconn/dlabel.nii: CIFTI file
        - .h5/hdf5/he2/he5: HDF5 file representing CIFTI data
        - .zarr: zarr file representing CIFTI data
        - .gii: GIFTI file (only stores surface data;
            raises error if more that one surface is represented in the greyordinates)
        - .nii: NIFTI file (only stores the volumetric data)

        :param filename: target filename
        """
        if hasExt(filename, ('.dscalar.nii', '.dconn.nii', '.dlabel.nii')):
            self.to_cifti().to_filename(filename)
        elif hasExt(filename, ('.h5', '.hdf5', '.he2', 'he5')):
            import h5py
            with h5py.File(filename, 'w') as f:
                self.to_hdf5(f)
        elif hasExt(filename, ('.zarr', )):
            import zarr
            f = zarr.group(filename)
            self.to_hdf5(f)
        elif hasExt(filename, ('.gii', )):
            surfaces = np.unique(self.brain_model_axis.name[self.brain_model_axis.surface_mask])
            if len(surfaces) > 1:
                raise ValueError(f"Can not write to GIFTI file as more than one surface has been defined: {surfaces}")
            if len(surfaces) == 0:
                raise ValueError("Can not write to GIFTI file as no surface has been provided")
            write_gifti(filename, [self.surface(surfaces[0])], surfaces[0])
        elif hasExt(filename, ('.nii.gz', '.nii')):
            self.volume().to_filename(filename)
        else:
            raise IOError(f"Extension of {filename} not recognized for NIFTI, GIFTI, or CIFTI file")

    def as_dask(self, chunks='auto', name='greyordinates'):
        """
        Returns the greyordinates as a dask array

        :param chunks: shape of the chunks (defaults to chunks of the dataset)
        :param name: name of the dask array
        :return: dask array
        """
        if chunks == 'auto':
            chunks = getattr(self.data, 'chunks', 'auto')
            if chunks != 'auto':
                size_chunks = np.prod(chunks)
                if size_chunks < 2e7:
                    mult_factor = int((2e7 / size_chunks) ** (1. / self.data.ndim))
                    chunks = tuple(c * mult_factor for c in chunks)
        return dask.array.from_array(self.data, chunks, name)

    def transpose(self, ):
        """
        Transposes a dense connectome
        """
        if self.data.ndim != 2:
            raise ValueError("Can only transpose 2D datasets")
        return GreyOrdinates(np.transpose(self.data), brain_model_axis=self.other_axes[0],
                             other_axes=(self.brain_model_axis, ), parent_file=self.parent_file)


def stack(greyordinates, axis=0):
    """
    Stacks a sequene of greyordinates along the given axis

    Resulting GreyOrdinate will only contain voxels/vertices in all GreyOrdinate arrays

    :param greyordinates: individual greyordinates to be merged
    :return: merged greyordinate object
    """
    new_bm, slices = cifti.combine([go.brain_model_axis for go in greyordinates])
    new_arr = np.stack([go.data[..., slc] for go, slc in zip(greyordinates, slices)], axis=axis)
    ref_axes = set([go.other_axes for go in greyordinates if go.other_axes is not None])
    if len(ref_axes) == 0:
        other_axes = None
    elif len(ref_axes) == 1:
        other_axes = list(ref_axes[0])
        other_axes.insert(axis, ScalarAxis([f'stacked_{idx + 1}' for idx in range(len(greyordinates))]))
    else:
        raise ValueError("Failed to merge greyordinates as their other axes did not match")
    return GreyOrdinates(new_arr, new_bm, other_axes)


def concatenate(greyordinates, axis=0):
    """
    Stacks a sequene of greyordinates along the given axis

    Resulting GreyOrdinate will only contain voxels/vertices in all GreyOrdinate arrays

    :param greyordinates: individual greyordinates to be merged
    :return: merged greyordinate object
    """
    if len(greyordinates) == 1:
        return greyordinates[0]
    ref_axes = [go.other_axes for go in greyordinates if go.other_axes is not None]
    if len(ref_axes) == 0:
        other_axes = None
    elif any(ra != ref_axes[0] for ra in ref_axes[1:]):
        raise ValueError("Failed to merge greyordinates as their other axes did not match")
    else:
        other_axes = ref_axes[0]

    if axis == -1 or axis == greyordinates[0].ndim - 1:
        new_bm = greyordinates[0].brain_model_axis
        for go in greyordinates[1:]:
            new_bm = new_bm + go.brain_model_axis
        new_arr = np.concatenate([go.data for go in greyordinates], axis=axis)
        return GreyOrdinates(new_arr, new_bm, other_axes)
    else:
        new_bm, slices = cifti.combine([go.brain_model_axis for go in greyordinates])
        new_arr = np.concatenate([go.data[..., slc] for go, slc in zip(greyordinates, slices)], axis=axis)
        return GreyOrdinates(new_arr, new_bm, other_axes)


def parse_greyordinate(filename):
    """
    Parses a set of filenames as a single greyordinate object

    :param filename: '@'-symbol separated files (NIFTI, GIFTI, and/or CIFTI)
    :return: single Greyordinate object representing the full dataset
    """
    try:
        parts = [GreyOrdinates.from_filename(fn) for fn in filename.split('@')]
    except (ValueError, IOError) as e:
        raise argparse.ArgumentTypeError(*e.args)

    if len(parts) == 0:
        raise argparse.ArgumentParser("Can not parse empty string")

    return concatenate(parts, axis=-1)
