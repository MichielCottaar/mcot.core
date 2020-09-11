#!/usr/bin/env python
"""Run part of the voxel-wise job"""
from loguru import logger
import nibabel as nib
from fsl.data.image import addExt
from subprocess import run as srun
import numpy as np
from typing import Sequence, Tuple
import string
import itertools
import tempfile
import sys


def get_markers(njobs: int) -> Tuple[str, ...]:
    """
    number of jobs to submit

    :param njobs: number of jobs
    :return: sequence of job identifiers
    """
    njobs = int(njobs)
    nletters = int(np.floor(np.log(njobs) / np.log(len(string.ascii_uppercase)))) + 1
    res = tuple(''.join(letters) + str(njobs) for letters in
                itertools.combinations_with_replacement(string.ascii_uppercase, nletters))
    assert len(res) > njobs
    return res[:njobs]


def run(job_id: int, njobs: int, mask_fn: str, command: Sequence[str]):
    """
    Runs part of the script

    :param job_id: job ID
    :param njobs: number of jobs
    :param mask_fn: mask filename
    :param command: script to run
    """
    logger.debug(f'loading original mask from {mask_fn}')
    mask_img = nib.load(addExt(mask_fn, mustExist=True, unambiguous=True))
    mask = mask_img.get_data() > 0
    voxels = np.where(mask)
    nvox = voxels[0].size
    boundaries = np.round(np.linspace(0, nvox, njobs + 1)).astype('int')

    use = tuple(vox[boundaries[job_id]: boundaries[job_id + 1]] for vox in voxels)
    logger.debug(f'creating new mask covering voxels {boundaries[job_id]} to ' +
                 f'{boundaries[job_id + 1]} out of {nvox} voxels')
    mask[()] = False
    mask[use] = True
    marker = get_markers(njobs)[job_id]
    with tempfile.NamedTemporaryFile(prefix='mask' + marker, suffix='.nii.gz') as temp_mask:
        logger.debug(f'Storing new mask under {temp_mask}')
        nib.Nifti1Image(mask.astype('i4'), affine=None, header=mask_img.header).to_filename(temp_mask.name)

        if not any('MASK' in part for part in command):
            raise ValueError('MASK not found')
        new_cmd = [part.replace('MASK', temp_mask.name).replace('JOBID', marker) for part in command]
        logger.info(f'Running {new_cmd}')
        srun(new_cmd, check=True)


def parse_args(argv=None):
    """
    Creates the parser of the command line arguments
    """
    if argv is None:
        argv = sys.argv[1:]
    if len(argv) < 3:
        print(usage())
        sys.exit(1)
    try:
        id_job = int(argv[0]) - 1
        njobs = int(argv[1])
        if id_job < 0 or id_job >= njobs:
            raise ValueError(f"Job id ({id_job + 1} should be between 1 and number of jobs {njobs}")
        img = argv[2]
        cmd = argv[3:]
    except Exception:
        print(usage())
        raise
    return id_job, njobs, img, cmd


def usage():
    return """
mc_script split_run [id_job] [njobs] [mask] command

in command replace the name of the mask with MASK
any output-file should have JOBID appended to it
e.g., the first out of four jobs for dtifit becomes
mc_script split.run 1 4 nodif_brain_mask.nii.gz dtifit -m MASK -b bvals -k data -r bvecs -o dtiJOBID
"""


def main():
    """
    Runs the script from the command line
    """
    args = parse_args()
    run(*args)
