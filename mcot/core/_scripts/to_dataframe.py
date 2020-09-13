#!/usr/bin/env python
"""Converts a NIFTI/GIFTI/CIFTI file into a pandas dataframe

Each non-masked greyordinate (i.e., voxel or vertex) becomes one line in the resulting dataframe.
Each file becomes a column.

The output dataframe will be stored in the feather format, which is a language agnostic format
for storing tables (https://blog.rstudio.com/2016/03/29/feather/)
"""
from typing import Union, Optional
import nibabel as nib
from nibabel import gifti
import pandas as pd
import numpy as np
from mcot.core.surface import BrainStructure
from nibabel import cifti2
from nibabel.cifti2 import Cifti2Image
from nibabel.filebasedimages import ImageFileError


def from_nifti(img: Union[str, nib.Nifti1Image, np.ndarray],
               mask: Union[None, str, nib.Nifti1Image, np.ndarray] = None,
               name='nifti') -> pd.DataFrame:
    """
    Converts a NIFTI image into a pandas dataframe

    :param img: NIFTI image
    :param mask: masks which voxels to actually include (default: any non-zero voxels)
    :param name: column name
    :return: pandas dataframe with unmasked voxels
    """
    if isinstance(img, str):
        data = nib.load(img).get_data()
    elif isinstance(img, nib.Nifti1Image):
        data = img.get_data()
    data = np.atleast_3d(data)
    if mask is None:
        mask = data != 0
        while mask.ndim > 3:
            mask = mask.any(-1)
    elif isinstance(mask, str):
        mask = nib.load(mask).get_data() > 0
    elif isinstance(mask, nib.Nifti1Image):
        mask = mask.get_data() > 0
    mask = np.atleast_3d(mask)

    indices = np.indices(data.shape[:3])[:, mask]
    masked = data[mask]

    as_dict = {
        ('voxel', 'i'): indices[0],
        ('voxel', 'j'): indices[1],
        ('voxel', 'k'): indices[2],
    }
    if masked.ndim == 1:
        as_dict[(name, '')] = masked
    else:
        for idx in range(masked.shape[1]):
            as_dict[(name, str(idx))] = masked[:, idx]
    df = pd.DataFrame.from_dict(as_dict)
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    return df


def from_gifti(img: Union[str, gifti.GiftiImage],
               mask: Union[None, str, nib.Nifti1Image, np.ndarray] = None,
               hemisphere: Optional[str]=None, basename=''):
    """
    Converts a GIFTI image into a pandas dataframe

    :param img: input GIFTI image
    :param mask: input mask (default: include all vertices)
    :param hemisphere: hemisphere of the data
    :param basename: basename of the columns
    :return: pandas dataframe with unmasked vertices
    """
    if isinstance(img, str):
        img = nib.load(img)
    if hemisphere is None:
        hemisphere = BrainStructure.from_string(img.meta.metadata['AnatomicalStructurePrimary']).hemisphere
    as_dict = {
        ('vertex', ''): np.arange(img.darrays[0].data.shape[0]),
        ('structure', 'hemisphere'): hemisphere
    }
    if mask is None:
        part_mask = np.ones(img.darrays[0].data.shape[0], dtype='bool')
    for darray in img.darrays:
        data = darray.data
        name = darray.meta.metadata.get('Name', '')
        if len(img.darrays) == 1:
            name = ''
        if data.ndim == 1:
            as_dict[(basename, name)] = data
            if mask is None:
                part_mask &= data != 0
        else:
            for idx in range(data.shape[1]):
                as_dict[(basename, f'{name}_{idx:02i}')] = data[:, idx]
                if mask is None:
                    part_mask &= data[:, idx] != 0
    if mask is None:
        mask = part_mask
    else:
        mask = mask.darrays[0].get_data() > 0

    masked_dict = {key: value if isinstance(value, str) else value[mask]
                   for key, value in as_dict.items()}
    df = pd.DataFrame.from_dict(masked_dict)
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    return df


def from_cifti(filename: str, basename=''):
    """
    Converts a cifti file into a pandas dataframe

    :param filename: input dense cifti file
    :param basename: basename of the dataframe columns
    :return: pandas datraframe with any voxels/vertices in the dense input
    """
    img = cifti2.load(filename)
    arr = np.asarray(img.dataobj)
    axis = img.header.get_axis(0)
    bm = img.header.get_axis(1)

    if not isinstance(bm, cifti2.BrainModelAxis):
        raise ValueError(f'Input CIFTI file {filename} is not dense')

    as_dict = {
        ('voxel', 'i'): bm.voxel[:, 0],
        ('voxel', 'j'): bm.voxel[:, 1],
        ('voxel', 'k'): bm.voxel[:, 2],
        ('vertex', ''): bm.vertex,
        ('structure', 'hemisphere'): [BrainStructure.from_string(name).hemisphere for name in bm.name],
        ('structure', 'region'): [BrainStructure.from_string(name).primary for name in bm.name],
        ('structure', 'cifti_label'): bm.name
    }
    if isinstance(axis, cifti2.ScalarAxis):
        for sub_arr, name in zip(arr, axis.name):
            if len(axis) == 1:
                name = ''
            as_dict[(basename, name)] = sub_arr
    elif isinstance(axis, cifti2.SeriesAxis):
        for sub_arr, name in zip(arr, np.arange(len(axis))):
            as_dict[(basename, name)] = sub_arr
    elif isinstance(axis, cifti2.LabelAxis):
        for sub_arr, name, mapping in zip(arr, axis.name, axis.label):
            if len(axis) == 1:
                name = ''
            label_names = {key: value[0] for key, value in mapping.items()}
            as_dict[(basename, name)] = label_names[sub_arr]
    elif isinstance(axis, cifti2.ParcelsAxis):
        for sub_arr, name in zip(arr, axis.name):
            as_dict[(basename, name)] = sub_arr
    df = pd.DataFrame.from_dict(as_dict)
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    return df


def convert_filenames(filenames, vol_mask=None, surf_mask=None) -> pd.DataFrame:
    """
    Converts the list of filenames into a single pandas dataframe

    :param filenames: sequence of (basename, NIFTI, GIFTI, or CIFTI filename/object)
    :param vol_mask: volumetric NIFTI mask
    :param surf_mask: surface GIFTI mask
    :return: dataframe with all the data
    """
    df = None
    for basename, filename in filenames:
        if isinstance(filename, str):
            try:
                img = nib.cifti2.load(filename)
            except ImageFileError:
                img = nib.load(filename)
        else:
            img = filename
        if isinstance(img, gifti.GiftiImage):
            df_new = from_gifti(img, surf_mask, basename=basename)
        elif isinstance(img, Cifti2Image):
            df_new = from_cifti(filename, basename=basename)
        else:
            df_new = from_nifti(filename, vol_mask, name=basename)
        if df is None:
            df = df_new
        else:
            sort_by = [name for name in (
                ('structure', 'cifti_label'), ('structure', 'hemisphere'), ('structure', 'region'),
                ('vertex', ''), ('voxel', 'i'), ('voxel', 'j'), ('voxel', 'k'))
                       if name in df and name in df_new]
            df = df.merge(df_new, on=sort_by)
    return df


def dataframe_to_array(dataframe, names):
    """
    Writes a dataframe back to disc

    :param dataframe: pandas dataframe with the data
    :param names: columns to export
    :return: (N, M) array for N rows and M columns
    """
    if isinstance(names, str):
        names = [names]
    arrays = [dataframe[name].values for name in names]
    arrays_2d = [arr[:, None] if arr.ndim == 1 else arr for arr in arrays]

    return np.concatenate(arrays_2d, -1)


def to_nifti(dataframe, names, reference):
    """
    Writes the data from a pandas dataframe to a NIFTI file on disc

    :param filename: NIFTI output filename
    :param dataframe: Pandas dataframe listing the voxels and resulting data
    :param names: columns to export
    :param reference: reference image to set size and affine
    :return: Nibabel NIFTI image
    """
    img = nib.load(reference)
    arr = dataframe_to_array(dataframe, names)
    full_arr = np.zeros(img.shape[:3] + (arr.shape[-1], ))
    voxel_indices = dataframe['voxel'].values
    use = (voxel_indices > 0).all(-1)

    full_arr[tuple(voxel_indices[use].T)] = arr[use]
    return nib.Nifti1Image(full_arr, affine=None, header=img.header)


def run_from_args(args):
    """
    Runs the script based on a Namespace containing the command line arguments
    """
    df = convert_filenames(
            filenames=args.filenames,
            vol_mask=args.vol_mask,
            surf_mask=args.surf_mask
    )
    df.to_feather(args.output, complib='blosc', mode='w')


def add_to_parser(parser):
    """
    Creates the parser of the command line arguments
    """
    parser.add_argument('output', help='feather file to store the pandas dataframe')
    parser.add_argument('filenames', nargs='+', help='NIFTI, GIFTI, and/or CIFTI files which need to be converted')
    parser.add_argument('-v', '--vol_mask', help='volumetric mask applied to NIFTI files')
    parser.add_argument('-s', '--surf_mask', help='surface mask applied to GIFTI files')
