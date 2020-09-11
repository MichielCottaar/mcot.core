#!/usr/bin/env python
"""Converts related files loaded from a tree into a pandas dataframe.

Each short name in the table will become a different column.
Additional columns will be added to store the voxel/vertex indices and the variable values in the tree template.

For each greyordinate (i.e., voxel or vertex) in the mask a row will be added for each filename matching the short name

The output dataframe will be stored in the feather format, which is a language agnostic format
for storing tables (https://blog.rstudio.com/2016/03/29/feather/)
Warning: the resulting dataframe can be very large
"""
import os.path as op
from fsl.utils.filetree import FileTree
from typing import Sequence
import pandas as pd
from . import to_dataframe


def run(tree: FileTree, names: Sequence[str],
        vol_mask: str=None, surf_mask: str=None,
        join='inner', ignore_vars=('basename', 'name')) -> pd.DataFrame:
    """
    Extracts the information from the files matching the named templates into a dataframe

    :param tree: set of input files
    :param names: names matching templates in the tree
    :param vol_mask: volumetric NIFTI mask
    :param surf_mask: surface GIFTI mask
    :param join: How to join the dataframes from the different templates (use 'outer' to keep all data)
    :param ignore_vars: which variables to ignore
    :return: pandas dataframe with all the information of the NIFTI/GIFTI/CIFTI files
    """
    df = None
    all_variables = set()
    for name in names:
        dfs = []
        for filename in tree.get_all(name, glob_vars='all'):
            variables = tree.extract_variables(name, filename)
            if vol_mask is not None and not op.exists(vol_mask):
                vol_mask_use = tree.update(**variables).get(vol_mask)
            else:
                vol_mask_use = vol_mask
            if surf_mask is not None and not op.exists(surf_mask):
                surf_mask_use = tree.update(**variables).get(surf_mask)
            else:
                surf_mask_use = surf_mask
            all_variables.update(variables.keys())

            df_new = to_dataframe.convert_filenames([(name, filename)], vol_mask_use, surf_mask_use)
            for var, value in variables.items():
                if var not in ignore_vars:
                    df_new[var] = value
            dfs.append(df_new)
        df_new = pd.concat(dfs)

        if df is None:
            df = df_new
        else:
            shared_names = (
                ('structure', 'cifti_label'), ('structure', 'hemisphere'), ('structure', 'region'),
                ('vertex', ''), ('voxel', 'i'), ('voxel', 'j'), ('voxel', 'k')
            ) + tuple(variables)
            sort_by = [name for name in shared_names
                       if name in df and name in df_new]
            df = df.merge(df_new, on=sort_by, how=join)
    for name in (tuple(('structure', name) for name in ['hemisphere', 'cifti_label', 'region']) +
                 tuple(all_variables)):
        if name in df:
            df[name] = df[name].astype('category')
    return df


def run_from_args(args):
    """
    Runs the script based on a Namespace containing the command line arguments
    """
    tree = FileTree.read(args.tree, args.directory, dict(args.variable))
    df = run(tree, args.name, args.vol_mask, args.surf_mask)
    df.to_feather(args.output, complib='blosc', mode='w')


def add_to_parser(parser):
    """
    Creates the parser of the command line arguments
    """
    parser.add_argument('tree', help='tree name or filename')
    parser.add_argument('output', help='feather file to store the pandas dataframe in')
    parser.add_argument('name', nargs='+', help='one or more short names of files to extract from the tree')
    parser.add_argument('-v', '--vol_mask', help='volumetric mask applied to NIFTI files')
    parser.add_argument('-s', '--surf_mask', help='surface mask applied to GIFTI files')
    parser.add_argument('-d', '--directory', default='.', help='path to the top-level directory')
    parser.add_argument('-var', '--variable', nargs=2, action='append', default=(), help='fixes a variable to a certain value')
