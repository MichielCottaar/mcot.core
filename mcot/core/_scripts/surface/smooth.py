"""
Smooths the surface mesh

Similar to the wb_command implementation, but includes additional options for volume preservation.
This difference is only really relevant for small meshes (i.e., with high curvature).
"""
from mcot.core.surface import CorticalMesh


def run_from_args(args):
    """
    Runs script from command line arguments

    :param args: command line arguments from :func:`add_to_parser`.
    """
    surf = CorticalMesh.read(args.input)
    if args.preserve_bumps:
        res = surf.smooth(args.nsteps, args.step)
    else:
        intermediate = surf.smooth(args.nsteps, args.step, expand_step=0)
        res = intermediate.inflate(volume=surf.volume() - intermediate.volume())
    res.write(args.output, **surf.anatomy.gifti)


def add_to_parser(parser):
    """
    Creates the parser of the command line arguments

    After parsing the script can be run using :func:`run_from_args`.

    :param parser: parser to add arguments to (default: create a new one)
    """
    parser.add_argument("input", help="existing GIFTI (.surf.gii) file with input surface mesh")
    parser.add_argument("output", help="GIFTI (.surf.gii) file with output surface mesh")
    parser.add_argument("-N", "--nsteps", type=int, default=10, help='Number of steps to take (default: 10)')
    parser.add_argument("-s", "--step", type=float, default=0.5,
                        help='Smoothing step size between 0 (no smoothing) and 1 (replace each vertex with average of neighbours). Default is 0.5')
    parser.add_argument("-p", "--preserve-bumps", action='store_true',
                        help="Try to preserve the bumps during the smoothing procedure")
