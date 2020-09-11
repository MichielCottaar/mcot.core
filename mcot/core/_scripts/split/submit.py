#!/usr/bin/env python
"""Submits a job using a mask in multiple parts

In command replace the name of the mask with MASK
Any output-file should have JOBID appended to it
e.g., to submit dtifit in 4 individual jobs run:
mc_script split.submit 4 nodif_brain_mask.nii.gz -q short.q "dtifit -m MASK -b bvals -k data -r bvecs -o dtiJOBID"
"""
from loguru import logger
from fsl.utils.fslsub import SubmitParams
import string


def get_names(cmd):
    """
    Given a command iterates through all the mask-dependent outputs

    The mask-dependent outputs should have 'JOBID' in their name

    :param cmd: command to be split up for different sub-masks
    :yield: string with directory/basename/filename dependent on the mask
    """
    for part in cmd.split():
        if 'JOBID' in part:
            if part[0] == '-':
                if part.count('=') != 1:
                    raise ValueError(f"expected signle '='-sign in option: {part}")
                yield part.split('=')[1]
            else:
                yield part


def get_job_name(cmd):
    """
    Gets the name of the submitted job to set when submitting

    :param cmd: string or sequence with the command
    :return: descriptive name
    """
    if isinstance(cmd, str):
        cmd = cmd.split()
    if cmd[0] == 'python':
        for part in cmd[1:]:
            if part[0] != '-':
                return part
        return 'python'
    elif cmd[0] == 'mc_script':
        if cmd[1] in ('gcoord', 'MDE', 'plot', 'split'):
            return cmd[1] + '.' + cmd[2]
        return cmd[1]
    return cmd[0].split('/')[-1]


def run(njobs, mask_fn, cmd, submit_params: SubmitParams):
    """
    Submits the cmd multiple times for different parts of the mask

    :param njobs: number of jobs to submit
    :param mask_fn: mask filename (has to exist at the time the run scripts start
    :param cmd: command line string
    :param submit_params: submission parameters
    :return: string with the final job id
    """
    if isinstance(submit_params, dict):
        submit_params = SubmitParams(**submit_params)
    jobs = []
    if submit_params.job_name is None:
        submit_params.job_name = get_job_name(cmd)
    for job_id in range(1, njobs + 1):
        jobs.append(submit_params(
                f'mc_script split.run {job_id} {njobs} {mask_fn} {cmd}',
                job_name=submit_params.job_name + string.ascii_uppercase[job_id - 1]
        ))
    final_job = submit_params(
            ' '.join(('mc_script', 'split.merge', str(njobs)) + tuple(get_names(cmd))),
            wait_for=tuple(jobs), minutes=45, job_name=submit_params.job_name + '_merge'
    )
    logger.info(f'Submitted {njobs} jobs to run {cmd}')
    logger.debug(f'Final merge job id: {final_job}')
    return final_job


def run_from_args(args):
    """
    Runs the script based on a Namespace containing the command line arguments
    """
    print(run(
            args.njobs,
            args.mask_fn,
            args.cmd,
            submit_params=SubmitParams.from_args(args),
    ))


def add_to_parser(parser):
    """
    Creates the parser of the command line arguments
    """
    parser.add_argument('njobs', type=int, help='number of jobs to merge')
    parser.add_argument('mask_fn', help='mask filename')
    parser.add_argument('cmd', help='full command with MASK replacing the mask filename and JOBID as a placeholder')
    SubmitParams.add_to_parser(parser)
