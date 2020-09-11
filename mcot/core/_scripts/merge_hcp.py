#!/usr/bin/env python
"""
Creates new directory with links to HCP data

The full HCP directory structure is reproduced in the target directory with symlinks to the individual files.
The resulting directory will contain the structural, diffusion, and functional data.
"""
import argparse
from mcutils import hcp_dir


def run_from_args(args):
    """
    Runs the script based on a Namespace containing the command line arguments
    """
    hcp_dir.merge_hcp(
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
