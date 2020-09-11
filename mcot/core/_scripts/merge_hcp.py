#!/usr/bin/env python
"""
Creates new directory with links to HCP data

The full HCP directory structure is reproduced in the target directory with symlinks to the individual files.
The resulting directory will contain the structural, diffusion, and functional data.
"""
import argparse
import os.path as op
from pathlib import Path
import argparse
from warnings import warn
from mcot.core._scripts.iter_link import link_dir


def HCP_dir(modality='Diffusion', release='Q1200') -> str:
    """
    Gets the parent HCP directory
    """
    for path in (
            op.join(op.expanduser('~'), 'Work', 'fmrib', 'scratch', 'HCP', modality, release),
            op.join('/vols','Scratch', 'HCP', modality, release),
    ):
        if op.exists(path):
            return path
    raise IOError("HCP directory not found")


def link(src, modality='Diffusion', release='Q1200'):
    """
    Links to the diffusion data on the jalapeno server or the local copy

    :param src: path where the link will be placed
    :param modality: one of ('Diffusion', Structural', 'rfMRI', 'tfMRI')
    :param release: which release to link
    """
    src = Path(src).absolute()
    if not src.exists():
        target_path = Path(HCP_dir(modality, release))
        common_path = Path(op.commonpath([target_path, src]))
        rel_path = '/'.join(['..'] * tuple(src.parents).index(common_path)) / target_path.relative_to(common_path)

        src.symlink_to(rel_path)


def merge_hcp(target_dir, subject, release='Q1200'):
    """
    Create a new HCP-like directory in `target_dir`/`subject` with links to split HCP directories

    :param target_dir: new HCP directory
    :param subject: subject id
    """
    target_dir = Path(target_dir) / subject
    if not target_dir.is_dir():
        target_dir.mkdir(parents=True)
    for modality in ('Structural', 'Diffusion', 'tfMRI', 'rfMRI'):
        try:
            src_dir = Path(HCP_dir(modality, release)) / subject
        except IOError:
            try:
                src_dir = Path(HCP_dir(modality, f'subjects{release[1:]}')) / subject
            except IOError:
                warn(f"Modality {modality} not found for subject {subject}")
                continue

        link_dir(src_dir, target_dir)


def run_from_args(args):
    """
    Runs the script based on a Namespace containing the command line arguments
    """
    merge_hcp(
            args.target_dir,
            args.subject,
            args.release
    )


def add_to_parser(parser="Create a new HCP-like directory in `target_dir`/`subject` with links to the HCP data"):
    """
    Creates the parser of the command line arguments
    """
    if isinstance(parser, str):
        parser = argparse.ArgumentParser(parser)
    parser.add_argument("target_dir", help='new HCP directory')
    parser.add_argument("subject", help='subject id')
    parser.add_argument("release", default='Q1200', nargs='?', help='Release specifier (default: Q1200)')
