#!/usr/bin/env python.app
"""GUI for interacting with gyral coordinate scripts"""
try:
    from gooey import Gooey, GooeyParser
except ModuleNotFoundError:
    raise ImportError("You need to install the gooey python library to run the gyral coordinate GUI")


@Gooey
def main():
    parser = GooeyParser(description="Utilities to work with the gyral coordinate system")
    subs = parser.add_subparsers(help='commands', dest='commands')

    generator = subs.add_parser(
            name='generate', description="Computes the radial and tangential hemisphere for every point in space")
    generator.add_argument("algorithm", choices=['line', 'interp', 'closest'], help="""Select which algorithm to use to compute the radial/tangential orientations. There are three options:
1. 'line': strongly recommended with -pd -1: The surface normal (& sulcal depth gradient) are linearly interpolated along N randomly oriented lines and averaged
2. 'interp': best with -pd -3: All surface normals (& sulcal depth depth gradients) are averaged with the weighting depending on the distance from the voxel of interest
3. 'closest': Use the normal (& sulcal depth gradient) from the closest vertex""")
    generator.add_argument("output", help='Output NIFTI file where the gyral coordinate system will be stored',
                           widget='FileSaver')
    generator.add_argument("-m", "--mask", help="computes the gyral coordinates for any non-zero voxels",
                           widget='FileChooser')
    generator.add_argument("-w", "--white", required=True,
                           help="GIFTI file defining the white/gray matter boundary (ending with .surf.gii)",
                           widget='FileChooser')
    generator.add_argument("-p", "--pial", required=True,
                           help="GIFTI file defining the pial surface (ending with .surf.gii)", widget='FileChooser')
    generator.add_argument("-sd", "--sulcal_depth",
                           help="GIFTI file defining the sulcal depth (ending with .shape.gii or .func.gii)",
                           widget='FileChooser')
    generator.add_argument(
            "-N", "--norient", default=300, type=int,
            help="Number of random line orientations to average over (only if algorithm='line'); default: 1000")
    generator.add_argument(
            "-pd", "--power_dist", default=-1., type=float,
            help="weights in averaging = d ** `power_dist`, where weights is the line length (if algorithm='line') or the distance between vertex and point of interest (if algorithm='interp'); default: -1")
    generator.add_argument(
            "-t", "--thickness", widget='FileChooser',
            help="Optional NIFTI output file containing the length of the shortest line connecting the cortical surfaces through the voxel (only valid for the `line` algorithm")

    splitter = subs.add_parser(name='split',
                               description="Extract radial, sulcal, and gyral dyads from a coord NIFTI file")
    splitter.add_argument('coord', help='name of the coord file', widget='FileChooser')
    splitter.add_argument('-b', '--base', help='Basename of the output files', widget='FileSaver')
    splitter.add_argument('-r', '--radial', help='Filename for the radial output (overrides the --base option)',
                          widget='FileSaver')
    splitter.add_argument('-s', '--sulcal', help='Filename for the sulcal output (overrides the --base option)',
                          widget='FileSaver')
    splitter.add_argument('-g', '--gyral', help='Filename for the gyral output (overrides the --base option)',
                          widget='FileSaver')

    mult = subs.add_parser(description="Converts the dyads from the cardinal to the gyral coordinate system",
                           name='multiply')
    mult.add_argument('-d', '--dyads', required=True, help='NIFTI file with the estimated fibre orientations',
                      widget='FileChooser')
    mult.add_argument('-c', '--coord', required=True, nargs='+',
                      help='NIFTI file with the gyral coordinates produced by gcoord_gen', widget='MultiFileChooser')
    mult.add_argument('-o', '--output', required=True,
                      help='Output NIFTI files with the fibre orientations in gyral coordinates', widget='FileSaver')

    transition = subs.add_parser(
            description='Fits a sigmoidal model to the radial index across the white/gray matter boundary',
            name='transition')
    transition.add_argument('coord', help='NIFTI file with radial & tangential orientations from gcood_gen',
                            widget='FileChooser')
    transition.add_argument('dyad', help='NIFTI file with DTI or other hemisphere', widget='FileChooser')
    transition.add_argument('white', help='GIFTI file with white/gray matter boundary', widget='FileChooser')
    transition.add_argument('output', help='CIFTI dscalar file with the output parameters', widget='FileSaver')
    transition.add_argument('-m', '--mask', help='GIFTI file with non-zero values for the vertices to use',
                            widget='FileChooser')
    transition.add_argument('-w', '--weight', type=float, default=1.,
                            help='Weighting of the smoothing term (defaults: 1., which works well for HCP data)')
    transition.add_argument('-md', '--min-dist', type=float, default=-3,
                            help='Minimal voxel distance from the white/gray matter boundary (default: -3)')
    transition.add_argument('--watson', action='store_true',
                            help='Assumes Watson rather than Normal distribution for radial index')
    transition.add_argument(
            '--idx-vertex', widget='FileChooser',
            help='Optional NIFTI file with the index of the vertex that every voxel should be assigned to')
    transition.add_argument('--distance', help='Optional NIFTI file with the distance from the WM/GM boundary',
                        widget='FileChooser')
    transition.add_argument('-ri', '--radial-index', widget='FileSaver',
                            help='Optional NIFTI file output containing the radial index of the best-fit model')

    args = parser.parse_args()

    if args.commands == 'generate':
        args.flip_inpr = False
        args.zval = None
        from mcot.core.surface import orientation
        orientation.run_from_args(args)
    elif args.commands == 'split':
        from mcot.core.surface import utils
        utils.gcoord_split(args)
    elif args.commands == 'multiply':
        from mcot.core.surface import utils
        utils.gcoord_mult(args)
    elif args.commands == 'transition':
        from mcot.core.surface import radial_transition
        radial_transition.run_from_args(args)
    else:
        raise ValueError("Unrecognized sub-command %s" % args.commands)
