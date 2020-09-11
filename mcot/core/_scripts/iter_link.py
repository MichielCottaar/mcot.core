#!/usr/bin/env python
"""
Iterates through source directory linking each file to destination directory.

The result is that the directory structure in the destination directory will match the source directory,
with each file in the source directory linked in (as long as the file did not already exist).

This can be useful if the source directory is read-only and you want a local copy where you can add files
without actually copying all the data.
"""
import os.path as op
from pathlib import Path
from warnings import warn


def link_dir(source: Path, destination: Path):
    """
    Links each file in source to destination path iteratively

    :param source: source directory
    :param destination: destination directory
    """
    if not source.is_symlink() and source.is_dir():
        if not destination.is_dir():
            destination.mkdir()
        try:
            for child in source.iterdir():
                link_dir(child, destination / child.name)
        except PermissionError as e:
            print(str(e))
    else:
        if not source.exists():
            warn(f'Did not find {source}')
            return
        if not destination.exists():
            rel_path = op.relpath(source, destination.parent)
            destination.symlink_to(rel_path)


def run_from_args(args):
    """
    Runs the script based on a Namespace containing the command line arguments
    """
    link_dir(args.source, args.destination)


def add_to_parser(parser):
    """
    Creates the parser of the command line arguments
    """
    parser.add_argument("source", type=Path, help='source directory')
    parser.add_argument("destination", type=Path, help='destination directory')
