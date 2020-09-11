"""
Selects vertices on surface to run tractography from
"""
from mcot.core.surface import CorticalMesh
import numpy as np


def run_from_args(args):
    """
    Runs script from command line arguments

    :param args: command line arguments from :func:`add_to_parser`.
    """
    surf = CorticalMesh.read(args.input)
    mask = np.zeros(surf.nvertices, dtype=int)
    mask[list(args.index)] = 1
    surf.write(args.output, mask)


def add_to_parser(parser):
    """
    Creates the parser of the command line arguments

    After parsing the script can be run using :func:`run_from_args`.

    :param parser: parser to add arguments to (default: create a new one)
    """
    parser.add_argument('input', help='input surface')
    parser.add_argument('output', help='same as input surface by only with selected vertices unmasked')
    parser.add_argument('index', nargs='+', type=int, help='vertices to select')
