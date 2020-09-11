#!/usr/bin/env python
"""Converts probtrackX to dconn"""
from nibabel import cifti2
import os.path as op
import nibabel as nib
from nibabel import gifti
from nibabel.filebasedimages import ImageFileError
import numpy as np
import pandas as pd
from scipy import sparse
from mcot.core.surface.cortical_mesh import get_brain_structure, CorticalMesh
from mcot.core.greyordinate import GreyOrdinates
from loguru import logger


def dot2matrix(dot_file):
    """
    Converts a fdt_matrix3.dot file into a sparse matrix

    :param dot_file: dot-file
    :return: (N, N) matrix
    """
    logging.debug(f'loading dot-matrix from {dot_file}')
    indices = pd.read_csv(dot_file, delim_whitespace=True, dtype=int).values.T
    shape = indices[:-1, -1]
    indices = indices[:, :-1]
    return sparse.coo_matrix((indices[2], (indices[0], indices[1])), shape=shape)


def run(output_file, dot_file, brain_model, other_brain_model=None) -> GreyOrdinates:
    """
    Creates a greyordinate object with the matrix data stored on disk

    :param output_file: output filename
    :param dot_file: dot-file containing the matrix
    :param brain_model: Defines greyordinates along second axis (i.e., first column in dot file)
    :param other_brain_model: Defines greyordinates along first axis
        (i.e., second column in dot file; defaults to brain_model)
    """

    with GreyOrdinates.empty(output_file, (other_brain_model, brain_model), dtype=int) as go:
        nrows = min(len(brain_model), int(2e8) // (len(other_brain_model) * 4))
        if hasattr(go.data, 'chunks') and go.data.chunks[1] < nrows:
            nrows = (nrows // go.data.chunks[1]) * go.data.chunks[1]
        logger.info(f"Storing {nrows} rows into memory before writing to disk")
        tmp_arr = np.zeros((nrows, len(other_brain_model)), dtype=int)

        current_idx = 0

        for idx, df in enumerate(pd.read_csv(dot_file, delim_whitespace=True, dtype=int,
                                             chunksize=int(1e6), header=None)):
            row, col, data = df.values.T
            if data[-1] == 0:
                row, col, data = row[:-1], col[:-1], data[:-1]
                print(row[-1], col[-1], data[-1])
                logger.info("Reached final row")
            assert (row[:-1] <= row[1:]).all()
            upper = np.searchsorted(row, current_idx + 1, 'right')
            tmp_arr[current_idx % nrows, col[:upper] - 1] = data[:upper]
            assert upper == 0 or row[0] == row[upper - 1]
            while row[-1] > current_idx + 1:
                current_idx += 1
                if current_idx % nrows == 0:
                    logger.debug(f'storing rows up to {current_idx}')
                    go.data[:, current_idx - nrows: current_idx] = tmp_arr.T
                    tmp_arr[()] = 0

                lower = upper
                upper = np.searchsorted(row[lower:], current_idx + 1, 'right') + lower
                assert (lower == upper) or (row[lower] == row[upper - 1])
                tmp_arr[current_idx % nrows, col[lower:upper] - 1] = data[lower:upper]

        nused = (current_idx % nrows) + 1
        go.data[:, current_idx - nused + 1:current_idx + 1] = tmp_arr[:nused].T

        logger.debug(f"Final row stored from {current_idx - nused + 1} till {current_idx + 1}")


def get_brain_model(filename: str, assumed_hemis=None) -> cifti2.BrainModelAxis:
    """
    Creates a CIFTI BrainModel axis based on the provided file

    :param filename: can be one of the following:

        - dense CIFTI file, whose BrainModel axis should be adopted
        - single volume (NIFTI) or surface (GIFTI/ASCII) used as a mask
        - list of volumes and surfaces

    :param assumed_hemis: editable list of hemispheres each surface is assumed to be
    :return: BrainModelAxis describing the dataset
    """
    try:
        if not isinstance(filename, str):
            img = filename
        else:
            img = nib.load(filename)
    except ImageFileError:
        with open(filename, 'r') as f:
            first_line = f.read()
        if first_line.startswith('#!ascii from CsvMesh'):
            return read_ascii(filename, None if assumed_hemis is None else assumed_hemis.pop(0))
        with open(filename, 'r') as f:
            bm_parts = []
            for line in f.readlines():
                single_filename = line.strip()
                if len(single_filename) > 0:
                    if not op.isfile(single_filename):
                        raise IOError(f"Mask filename {single_filename} not found")
                    bm_parts.append(get_brain_model(single_filename))
        if len(bm_parts) == 0:
            raise ValueError(f"No masks found in {filename}")
        bm = bm_parts[0]
        for part in bm_parts[1:]:
            bm = bm + part
        return bm

    # filename was loaded as a nibabel image
    if isinstance(img, cifti2.Cifti2Image):
        return img.header.get_axis(img.ndim - 1)
    elif isinstance(img, gifti.GiftiImage):
        mask = img.darrays[-1].data
        if mask.ndim == 2:
            mask = np.ones(CorticalMesh.read(img).nvertices, dtype='bool')
        return cifti2.BrainModelAxis.from_mask(mask, name=get_brain_structure(img).cifti)
    else:
        transposed_mask = np.transpose(img.get_fdata() > 0, (2, 1, 0))
        bm = cifti2.BrainModelAxis.from_mask(transposed_mask, affine=img.affine)
        bm.voxel = bm.voxel[:, ::-1]
        return bm


def read_ascii(filename, assumed_hemi=None):
    """
    Reads a surface stores as an ASCII file

    :param filename: ASCII file to read
    :param assumed_hemi: L or R defining the assumed hemisphere (default: guess from filename)
    :return: list of filenames in the ASCII file (or ASCII filename itself)
    """
    if assumed_hemi not in (None, 'L', 'R'):
        raise ValueError(f"Assumed hemisphere should be set to 'L' or 'R', not {assumed_hemi}")

    with open(filename, 'r') as f:
        first_line = f.readline()
        assert first_line.startswith('#!ascii from CsvMesh')
        nvertices, nfaces = [int(part) for part in f.readline().strip().split()]
        mask = np.zeros(nvertices, dtype='int')
        for idx in range(nvertices):
            mask[idx] = int(f.readline().strip().split()[-1])

    if assumed_hemi is not None:
        structure = 'CortexLeft' if assumed_hemi == 'L' else 'CortexRight'
    else:
        final_part = op.split(filename)[-1]
        if 'L' in final_part and 'R' in final_part:
            raise ValueError("Does not know which hemisphere %s belongs to" % filename)
        if 'L' in final_part:
            structure = 'CortexLeft'
        elif 'R' in final_part:
            structure = 'CortexRight'
        else:
            raise ValueError("Does not know which hemisphere %s belongs to" % filename)
    logger.info(f'{filename} interpreted as {structure}')

    return cifti2.BrainModelAxis.from_mask(mask, name=structure)


def run_from_args(args):
    logger.info('starting %s', op.basename(__file__))
    assumed_hemis = None if args.hemi is None else list(args.hemi)
    brain_model = get_brain_model(args.targets, assumed_hemis)
    other_brain_model = brain_model if args.otargets is None else get_brain_model(args.otargets, assumed_hemis)

    run(
        output_file=args.output,
        dot_file=args.dot_file,
        brain_model=brain_model,
        other_brain_model=other_brain_model
    )
    logger.info('ending %s', op.basename(__file__))


def add_to_parser(parser):
    parser.add_argument('dot_file', help='fdt_matrix.dot output from probtrackx')
    parser.add_argument('targets', help='CIFTI with brain model axis to use or '
                                        'ASCII file with target images (or single volume/surface file). '
                                        'Seed mask in case of matrix2')
    parser.add_argument('output', help='output .dconn.nii dense connectome file')
    parser.add_argument('--otargets', help='Target of matrix2 or --otargets for matrix3 (defaults to targets)')
    parser.add_argument('--hemi', nargs='+',
                        help='hemispheres to assume for the surface ASCII files (sequence of L or R). '
                             'First define for `targets` than `otargets`.')
