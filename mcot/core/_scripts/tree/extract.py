#!/usr/bin/env python
"""
Extract filenames from a FileTree

All matching, existing files and directories are returned.
The -I,--ignore flag can be used to find for which runs the pipeline crashed, for example:
`mc_script tree extract T1w -I T1w_bet`
will give you all the files for which the T1w file exists, but the corresponding T1w_bet file does not.
"""
import argparse
import os.path as op
from fsl.utils.filetree import FileTree, tree_directories


def run_from_args(args):
    """
    Runs the script based on a Namespace containing the command line arguments
    """
    tree_dir = op.split(args.tree)[0]
    tree_directories.append(tree_dir)

    vars = {}
    for full_var in args.vars:
        key, value = full_var.split('=')
        vars[key] = value

    tree = FileTree.read(args.tree, directory=args.directory).update(**vars)

    all_fns = []
    for name in args.short_names:
        for sub_tree in tree.get_all_trees(name, glob_vars='all'):
            if getattr(args, 'ignore', None) is not None and sub_tree.on_disk([args.ignore]):
                continue
            all_fns.append(sub_tree.get(name))
    print(' '.join(all_fns))


def add_to_parser(parser=None):
    """
    Creates the parser of the command line arguments
    """
    if parser is None:
        parser = __doc__
    if isinstance(parser, str):
        parser = argparse.ArgumentParser(parser)
    parser.add_argument("tree", help='FileTree object')
    parser.add_argument("short_names", nargs='+', help='filenames to print')
    parser.add_argument("-d", "--directory", default='.', help='base directory')
    parser.add_argument("-I", "--ignore", help='ignore sets of variables for which this short_name exists on disk')
    parser.add_argument("--vars", nargs='*', default=(), help='<key>=<value> pairs')
