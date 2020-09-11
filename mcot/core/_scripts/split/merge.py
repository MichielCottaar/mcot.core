#!/usr/bin/env python
"""Merge result from individual runs"""
from .run import get_markers
from loguru import logger
import os.path as op
import nibabel as nib
import numpy as np
from subprocess import run
import shutil
import os
import glob
import filecmp


def merge_files(paths, out_path, clean=True):
    """
    Merges the given files into a single file

    :param paths: NIFTI filenames
    :param out_path: output NIFTI file
    :param clean: whether to clean the input files after merging
    """
    if all(filecmp.cmp(paths[0], p) for p in paths[1:]):
        if clean:
            shutil.move(paths[0], out_path)
            for path in paths[1:]:
                os.remove(path)
        else:
            shutil.copy(paths[0], out_path)
        return

    imgs = [nib.load(path) for path in paths]
    outside_mask = imgs[-1].dataobj[(0, ) * len(imgs[-1].shape)]

    res = np.full(imgs[0].shape, outside_mask)

    if not np.isfinite(outside_mask):
        test_run = np.isfinite
    else:
        test_run = lambda arr: arr != outside_mask

    for img in imgs:
        data = img.get_data()
        use = test_run(data)
        res[use] = data[use]

    nib.Nifti1Image(res, affine=None, header=imgs[0].header).to_filename(out_path)
    if clean:
        for path in paths:
            os.remove(path)


def merge_basenames(basenames, out_basename, clean=True):
    """
    Merges all files starting with given basenames into files starting with out_basename

    :param basenames: input basenames
    :param out_basename: output basename
    :param clean: whether to clean the input files after merging
    """
    poss_fn = glob.glob(basenames[0] + '*')
    for first_fn in poss_fn:
        all_fn = [first_fn.replace(basenames[0], name) for name in basenames]
        out_fn = first_fn.replace(basenames[0], out_basename)
        if all(op.isfile(fn) for fn in all_fn):
            merge_files(all_fn, out_fn, clean=clean)
        elif all(op.isdir(fn) for fn in all_fn):
            merge_directories(all_fn, out_fn, clean=clean)
        else:
            logger.warning(f'No merging strategy found for {out_fn}')


def merge_directories(directories, out_directory, clean=True):
    """
    Merges all files in the input directories into out_directory

    :param directories: input directories
    :param out_directory: output directory
    :param clean: whether to clean the input files after merging
    """
    merge_basenames(
            [op.join(direc, '') for direc in directories],
            op.join(out_directory, ''),
            clean=clean,
    )


def run(names, njobs, clean=True):
    """
    Access to the script for other python programs.
    """
    for name in names:
        if 'JOBID' not in name:
            raise ValueError("Expects JOBID to be present in every name")
        paths = [name.replace('JOBID', marker) for marker in get_markers(njobs)]
        if all(op.isdir(path) for path in paths):
            merge_directories(paths, name.replace('JOBID', ''), clean=clean)
        elif all(op.isfile(path) for path in paths):
            merge_files(paths, name.replace('JOBID', ''), clean=clean)
        elif all(not op.exists(path) for path in paths):
            merge_basenames(paths, name.replace('JOBID', ''), clean=clean)
        else:
            raise IOError(f"Could not consistently identify {name} as a directory, filename, or basename")
    pass


def run_from_args(args):
    """
    Runs the script based on a Namespace containing the command line arguments
    """
    run(
            args.names,
            args.njobs,
            clean=not args.no_clean
    )


def add_to_parser(parser):
    """
    Creates the parser of the command line arguments
    """
    parser.add_argument('njobs', type=int, help='number of jobs to merge')
    parser.add_argument('names', nargs='+', help='names of files/basenames/directories to merge')
    parser.add_argument('-nc', '--no_clean', action='store_true', help='skip cleaning the data prior to merging')
