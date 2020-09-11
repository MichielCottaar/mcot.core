"""
Copies files matching short name from jalapeno

Warning: this script will only work if your directory structure matches mine (DO NOT USE!).
"""
import argparse
from fsl.utils.filetree import FileTree
from fsl.utils.filetree.utils import Template
from loguru import logger
from mcot.core.sync import sync
from typing import Sequence
import itertools


def sync_tree(tree: FileTree, directory: str, short_names: Sequence[str], dry_run=False,
              echo=False, jal00=False, reverse=False, max_size=False, from_dir=None):
    """
    Syncs given short names in the FileTree

    :param tree: defines templates for the input/output directories
    :param directory: parent directory (should be set to '' in the tree)
    :param short_names: which templates should be synced
    :param dry_run: do a dry run listing which files will be transferred
    :param echo: prints the rsync command rather than running it
    :param jal00: runs on jalapeno00 rather than jalapeno
    :param reverse: transfers to jalapeno rather than from it
    :param max_size: maximum file size to include
    :param from_dir: copy from given directory rather than from jalapeno
    """
    options = ['-avP', '--prune-empty-dirs']
    if dry_run:
        options[0].append('n')
    if max_size:
        options.append(f'--max_size={max_size}')
    options.append(f'--include=*/')
    server = 'jal00' if jal00 else 'jal'
    if from_dir is not None:
        server = None

    # add filters
    for short_name in short_names:
        text, variables = tree.get_template(short_name)
        template = Template.parse(text)
        filled = template.fill_known(variables)
        remaining = filled.required_variables()
        optional = filled.optional_variables()

        for keep in itertools.product(*[(True, False) for _ in optional]):
            sub_variables = {var: '*' for k, var in zip(keep, optional) if k}
            for required in remaining.difference(optional):
                sub_variables[required] = '*'
            star_filled = filled.resolve(sub_variables)
            options.append('--include=/' + star_filled)
    options.append('--exclude=*')

    if reverse:
        sync(directory, options=options, target=server, echo=echo, from_dir=from_dir)
    else:
        sync(directory, options=options, source=server, echo=echo, from_dir=from_dir)


def run_from_args(args):
    """
    Runs script from command line arguments

    :param args: command line arguments from :func:`add_to_parser`.
    """
    variables = {}
    for variable_value in args.vars:
        key, value = variable_value.split('=')
        variables[key] = value
    tree = FileTree.read(args.tree, directory='', **variables)
    short_names = tuple(tree.templates.keys()) if len(args.short_names) == 0 else args.short_names

    sync_tree(
        tree,
        args.directory,
        short_names,
        echo=args.echo,
        dry_run=args.dry_run,
        jal00=args.jal00,
        reverse=args.reverse,
        max_size=args.max_size,
        from_dir=args.from_dir,
    )


def add_to_parser(parser=None):
    """
    Creates the parser of the command line arguments

    After parsing the script can be run using :func:`run_from_args`.

    :param parser: parser to add arguments to (default: create a new one)
    """
    if parser is None:
        parser = __doc__
    if isinstance(parser, str):
        parser = argparse.ArgumentParser(parser)
    parser.add_argument('tree', help='FileTree to define files')
    parser.add_argument("short_names", nargs='*', help='filenames to print')
    parser.add_argument("-d", "--directory", default='.', help='base directory on jalapeno and locally')
    parser.add_argument("--from_dir", default=None, help='Directory to copy from rather than from jalapeno')
    parser.add_argument("--vars", nargs='*', default=(), help='<key>=<value> pairs')
    parser.add_argument('--max-size', help='Maximum size of files to transfer')
    parser.add_argument('-n', '--dry-run', action='store_true',
                        help='do a dry run listing which files will be transferred')
    parser.add_argument('-e', '--echo', action='store_true',
                        help='prints the command to stdout rather than running it')
    parser.add_argument('--jal00', action='store_true', help='use jalapeno00 rather than jalapeno')
    parser.add_argument('-r', '--reverse', action='store_true', help='send data to jalapeno rather than retrieving it')
