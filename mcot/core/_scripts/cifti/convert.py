"""Converts between CIFTI formats (NIFTI, HDF5, and zarr)"""
from loguru import logger
from mcot.core.greyordinate import GreyOrdinates
import zarr
import dask.array as da
import numpy as np


def copy(source: GreyOrdinates, target: GreyOrdinates):
    """
    Copies information from source to target.
    """
    if source.data.shape != target.data.shape:
        raise ValueError("Source and target shape do not match")

    if isinstance(target.data, zarr.Array):
        target.data[:] = source.data
    else:
        chunks = getattr(
            target.data, 'chunks', getattr(
                source.data, 'chunks', 'auto'
            )
        )
        for dataset in target.data, source.data:
            if not hasattr(dataset, 'chunks'):
                if chunks == 'auto':
                    chunks = [1, 1]
                else:
                    chunks = list(chunks)
                chunks[np.argmin(dataset.strides)] = None
        logger.info(f"Adopted chunk size: {tuple(chunks)} for CIFTI with shape {tuple(source.data.shape)}")

        data = source.as_dask(tuple(chunks))
        da.store(data, target.data)


def run_from_args(args):
    """
    Runs script from command line arguments

    :param args: command line arguments from :func:`add_to_parser`.
    """
    go_source = GreyOrdinates.from_filename(args.input)
    dtype = go_source.data.dtype if args.dtype is None else args.dtype
    logger.info("Creating empty output file")
    with GreyOrdinates.empty(
            args.output, go_source.other_axes + (go_source.brain_model_axis, ), dtype=dtype
    ) as go_target:
        copy(go_source, go_target)


def add_to_parser(parser):
    """
    Creates the parser of the command line arguments

    After parsing the script can be run using :func:`run_from_args`.

    :param parser: parser to add arguments to
    """
    parser.add_argument('input', help='source NIFTI, GIFTI, CIFTI, HDF5, or zarr file with CIFTI data')
    parser.add_argument('output', help='target CIFTI, HDF5, or zarr filename')
    parser.add_argument('-t', '--dtype', help='sets the output datatype (default: keep same as input)')
