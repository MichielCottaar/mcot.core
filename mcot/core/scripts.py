"""
Utilities to create a common interface to all the scripts
"""
import pkgutil
from typing import Union, Sequence, Tuple
import importlib
import ast
import nibabel as nib
from nibabel import cifti2
import argparse
import os.path as op
from nibabel import gifti
import numpy as np
from .surface.cortical_mesh import get_brain_structure, CorticalMesh, BrainStructure
from . import write_gifti
from nibabel.filebasedimages import ImageFileError
import sys
from .log import setup_log
from loguru import logger


def get_docstring(filename):
    """
    Extracts the filename from the docstring

    :param filename: path to python file
    :return: module-level docstring (None if undefined)
    """
    with open(filename) as f:
        base = ast.parse(f.read())
    return ast.get_docstring(base)


def run_script(add_to_parser, run_from_args, argc=None):
    """
    Run script from command line

    :param add_to_parser: function that takes an argument parser and adds information to it
    :param run_from_args: function that runs the script based on the arguments of the parser
    :param argc: command line arguments
    """
    module_name = add_to_parser.__module__
    if module_name != run_from_args.__module__:
        raise ValueError("run_script expected add_to_parser and run_from_args to be defined in the same module")
    module = sys.modules[module_name]
    doc_string = module.__doc__

    logger.enable('mcot.core')
    script_logger = logger.opt(depth=1)
    parser = argparse.ArgumentParser(doc_string)
    add_to_parser(parser)

    args = parser.parse_args(argc)

    setup_log()

    script_logger.info('starting script')
    try:
        run_from_args(args)
    except Exception:
        script_logger.exception('failed script')
        raise
    script_logger.info('finished script')


class _ScriptDirectories(object):
    """
    All script directories that have been registered

    All .py files within this directory are considered scripts (except __init__ and __main__)
    Any sub-directories are considered sub-scripts (as long as they contain a __init__)
    """
    def __init__(self, ):
        self.modules = []

    def add(self, name: str, group):
        """
        Adds a new script directory

        In the __init__ of the script directory add:

        mcot.core.scripts.directories.add(__name__)

        :param name: __name__ of the script directory
        :param group: what group to put the scripts in (set to None for no group)
        """
        self.modules.append((group, importlib.import_module(name)))

    def all_scripts(self, ):
        scripts = {}

        def process(module, script_dict):
            for module_info in pkgutil.iter_modules(module.__path__):
                if module_info.name.startswith('_'):
                    continue
                full_name = f'{module.__name__}.{module_info.name}'
                if module_info.name in script_dict:
                    raise ValueError(f"Dual script definition for {module_info.name}")
                if module_info.ispkg:
                    script_dict[module_info.name] = {}
                    process(
                        importlib.import_module(full_name),
                        script_dict[module_info.name]
                    )
                else:
                    script_dict[module_info.name] = full_name

        for name, module in self.modules:
            if name in scripts:
                raise ValueError(f"Dual script definition for {name}")
            if name is None:
                process(module, scripts)
            else:
                scripts[name] = {}
                process(module, scripts[name])
        return scripts

    def __iter__(self, ):
        """
        Returns strings representing all the scripts
        """
        for script in self._scripts(as_module=True, in_sub_group=True):
            yield script[len(self.in_module) + 1:]

    def to_string(self, tabs=0):
        """
        Creates a string listing all the scripts
        """
        def first_line(fn):
            res = get_docstring(fn)
            if res is None:
                return ''
            return res.splitlines()[0]

        lines = []
        for script in self._scripts():
            line = ' ' * 4 * tabs + '- ' + script.name[:-3]
            line += max(4, 20 - len(line)) * ' ' + first_line(script)
            lines.append(line)

        for group in self._script_groups():
            line = ' ' * 4 * tabs + '- ' + group.name
            line += max(4, 20 - len(line)) * ' ' + first_line(script)
            lines.append(line)
            lines.extend(group.to_string(tabs + 1).splitlines())
        return '\n'.join(lines)

    def __str__(self):
        return self.to_string()

    def get(self, args: Sequence[str]) -> Tuple[Union[str, "ScriptDirectory"], Sequence[str]]:
        """
        Find the requested script

        :param args: sequence of arguments starting with the script name
        :return: script module name or script directory and remaining arguments
        """
        if len(args) == 0:
            return self, args
        name = args[0]
        for script in self._scripts():
            if script.name == name or script.name[:-3] == name:
                return self.in_module + '.' + script.name[:-3], args[1:]
        for group in self._script_groups():
            if group.name == name:
                return group.get(args[1:])
        return self, args

    @staticmethod
    def _scripts2string(scripts, indent=0):
        if isinstance(scripts, dict):
            res = "\n"
            for name in sorted(scripts):
                res = res + " " * indent + f"- {name}:" + _ScriptDirectories._scripts2string(scripts[name], indent + 2)
            return res + ""
        else:
            return "\n"

    def __call__(self, args=None):
        """
        Runs a script identified by the arguments

        :param args: optionally group name and script name together with the script arguments (default: sys.argv[1:])
        """
        if args is None:
            args = sys.argv[1:]
        if '.' in args[0]:
            choose_script, args = args[0].split('.'), args[1:]
        else:
            choose_script, args = args, None

        current_group = []
        scripts = self.all_scripts()
        while len(choose_script) != 0 and isinstance(scripts, dict) and choose_script[0] in scripts.keys():
            current_group.append(choose_script[0])
            scripts = scripts[choose_script[0]]
            choose_script = choose_script[1:]

        if isinstance(scripts, dict):
            print('Usage: mcot [<script_group>...] <script_name> <args>...')
            if len(current_group) == 0:
                print('Available scripts:')
            else:
                print(f'Available scripts in script group {".".join(current_group)}:')
            print(self._scripts2string(scripts))
            print('')
            print("Error: Incomplete or invalid script name provided")
            exit(1)
        if args is None:
            args = choose_script
        elif len(choose_script) != 0:
            raise ValueError(f"Script already fully define before processing .{'.'.join(choose_script)}")

        script = importlib.import_module(scripts)
        if hasattr(script, 'main'):
            script.main(args)
        else:
            run_script(script.add_to_parser, script.run_from_args, args)


directories = _ScriptDirectories()


def load_all_mcot():
    mcot = importlib.import_module("mcot")
    for module_info in pkgutil.iter_modules(mcot.__path__):
        if module_info.ispkg:
            importlib.import_module(f"mcot.{module_info.name}")


def run(argv=None):
    load_all_mcot()
    directories(argv)


def _nifti2cifti(img):
    arr = img.get_data()
    mask = arr != 0
    while mask.ndim > 3:
        mask = mask.any(-1)
    arr = arr[mask]
    bm = cifti2.BrainModelAxis.from_mask(mask, affine=img.affine)
    assert len(bm) == arr.shape[0]
    axes = tuple(cifti2.SeriesAxis(0, 1, sz) for sz in arr.shape[1:]) + (bm, )
    return arr.T, axes


def _gifti2cifti(left, right, table=None):
    if left is not None and left.size != 0:
        mask_left = np.isfinite(left) & (left != 0)
        while mask_left.ndim > 1:
            mask_left = mask_left.any(-1)
        bm_left = cifti2.BrainModelAxis.from_mask(mask_left, 'CortexLeft')

    if right.size != 0:
        mask_right = np.isfinite(right) & (right != 0)
        while mask_right.ndim > 1:
            mask_right = mask_right.any(-1)
        bm_right = cifti2.BrainModelAxis.from_mask(mask_right, 'CortexRight')

        if left.size == 0:
            bm = bm_right
            farr = right[mask_right]
        else:
            bm = bm_left + bm_right
            farr = np.concatenate((left[mask_left], right[mask_right]), 0)
    elif left.size != 0:
        bm = bm_left
        farr = left[mask_left]
    else:
        raise ValueError("Neither left nor right hemisphere provided")

    farr = np.squeeze(farr)
    if farr.ndim == 1:
        farr = farr[:, None]

    new_axes = tuple(range(1, farr.ndim)) + (0, )
    if table is None:
        axes = tuple(cifti2.ScalarAxis(['???'] * sz) for sz in farr.shape[1:]) + (bm, )
        return np.transpose(farr, new_axes), axes
    lab = cifti2.ScalarAxis(['???'] * farr.shape[1]).to_label([table] * farr.shape[1])
    return np.transpose(farr, new_axes), (lab, bm)


def _cifti2nifti(arr, axes):
    bm = axes[-1]
    if not isinstance(bm, cifti2.BrainModelAxis):
        raise ValueError(f"CIFTI file should be dense to extract volumetric array")
    full_vol = np.zeros(bm.volume_shape + arr.shape[:-1])
    full_vol[tuple(bm.voxel[~bm.is_surface].T)] = arr[..., ~bm.is_surface].T
    return nib.Nifti1Image(full_vol, bm.affine)


def _cifti2gifti(arr, axes, get_label=False):
    res = [np.array(()), np.array(())]
    bm = axes[-1]
    if not isinstance(bm, cifti2.BrainModelAxis):
        raise argparse.ArgumentTypeError(f"CIFTI file should be dense to extract surface array")
    for name, slc, sub_bm in bm.iter_structures():
        anatomy = BrainStructure.from_string(name)
        if anatomy == 'CortexLeft' or anatomy == 'CortexRight':
            full_arr = np.zeros((sub_bm.nvertices[name], ) + arr.shape[:-1])
            full_arr[()] = np.nan
            full_arr[sub_bm.vertex] = arr[..., slc].T
            res[anatomy == 'CortexRight'] = full_arr
    if not get_label:
        return tuple(res)
    label = axes[0]
    if not isinstance(label, cifti2.LabelAxis):
        raise ValueError("Input axis must be Label to extract parcellation label table")
    return tuple(res), label.label[0]


def greyordinate_in(value):
    """
    Input greyordinate image

    3 types of input can be expected:
    - CIFTI file
    - GIFTI file
    - NIFTI file

    In the GIFTI and NIFTI files only non-zero values are kept

    :param value: input NIFTI, GIFTI, or CIFTI file
    :return: array, greyordinate axes
    """
    if '@' in value:
        arr = []
        for fn in value.split('@'):
            part_arr, axes = greyordinate_in(fn)
            arr.append(part_arr)
            if len(arr) == 1:
                ref_axes = axes[:-1]
                bm = axes[-1]
            else:
                assert axes[:-1] == ref_axes
                bm = bm + axes[-1]
        return np.concatenate(arr, -1), ref_axes + (bm, )

    if not op.isfile(value):
        raise argparse.ArgumentTypeError(f"Input greyordinate file {value} not found on disk")
    img = nib.load(value)
    try:
        img = nib.Cifti2Image.from_filename(value)
    except ImageFileError:
        pass
    if isinstance(img, nib.Cifti2Image):
        return np.asarray(img.dataobj), [img.get_axis(idx) for idx in range(img.ndim)]
    elif isinstance(img, gifti.GiftiImage):
        arr = np.squeeze(np.stack([darr.data for darr in img.darrays], 0))
        if img.labeltable is not None and len(img.labeltable.labels) > 1:
            table = {label.key: (label.label, label.rgba) for label in img.labeltable.labels}
        else:
            table = None
        if get_brain_structure(img).hemisphere == 'left':
            return _gifti2cifti(arr, np.array(()), table)
        else:
            return _gifti2cifti(np.array(()), arr, table)
    else:
        return _nifti2cifti(img)


def surface_arr_in(value):
    """
    Reads in arrays across the vertex

    For the CIFTI files the output arrays are padded with zeros

    :param value: input GIFTI or CIFTI file
    :return: tuple with vertex values in left and right cortex
    """
    if not op.isfile(value):
        raise argparse.ArgumentTypeError(f"Input surface array file {value} not found on disk")
    img = nib.load(value)
    try:
        img = nib.Cifti2Image.from_filename(value)
    except ImageFileError:
        pass
    if isinstance(img, nib.Cifti2Image):
        arr, axes = np.asarray(img.dataobj), [img.get_axis(idx) for idx in range(img.ndim)]
        res = _cifti2gifti(arr, axes)
        if res[0].size == 0 and res[1].size == 0:
            raise argparse.ArgumentTypeError(f"CIFTI file {value} does not contain cortical surface")
        return res
    elif isinstance(img, gifti.GiftiImage):
        res = [np.zeros(()), np.zeros(())]
        arr = np.squeeze(np.stack([darr.get_data() for darr in img.darrays], -1))
        res[get_brain_structure(img) == 'CortexRight'] = arr
        return tuple(res)
    else:
        raise ValueError(f"Surface arrays should be stored in GIFTI or CIFTI files, not {type(img)}")


def surface_label_in(value):
    """
    Reads in surface parcellation from a label file

    :param value: input GIFTI or CIFTI filename
    :return: tuple with vertex value and table in left and right cortex
    """
    if not op.isfile(value):
        raise argparse.ArgumentTypeError(f"Input surface label file {value} not found on disk")
    img = nib.load(value)
    try:
        img = nib.Cifti2Image.from_filename(value)
    except ImageFileError:
        pass
    if isinstance(img, nib.Cifti2Image):
        arr, axes = np.asarray(img.dataobj), [img.get_axis(idx) for idx in range(img.ndim)]
        res, table = _cifti2gifti(arr, axes, get_label=True)
        if res[0].size == 0 and res[1].size == 0:
            raise argparse.ArgumentTypeError(f"CIFTI file {value} does not contain cortical surface")
        return (res[0], table), (res[1], table)
    elif isinstance(img, gifti.GiftiImage):
        res = [(np.zeros(()), None), (np.zeros(()), None)]
        arr = np.squeeze(np.stack([darr.get_data() for darr in img.darrays], -1))
        table = {label.key: (label.label, label.rgba) for label in img.labeltable.labels}
        res[get_brain_structure(img) == 'CortexRight'] = (arr, table)
        return tuple(res)
    else:
        raise ValueError(f"Surface arrays should be stored in GIFTI or CIFTI files, not {type(img)}")


def volume_in(value):
    """
    Reads in a volume from a NIFTI or CIFTI file

    :param value: input filename (NIFTI or CIFTI)
    :return: nibabel NIFTI1Image (or other volumetric image)
    """
    if not op.isfile(value):
        raise argparse.ArgumentTypeError(f"Input volume file {value} not found on disk")
    img = nib.load(value)
    try:
        img = nib.Cifti2Image.from_filename(value)
    except ImageFileError:
        pass
    if isinstance(value, nib.Cifti2Image):
        arr, axes = np.asarray(img.dataobj), [img.get_axis(idx) for idx in range(img.ndim)]
        return _cifti2nifti(arr, axes)
    else:
        return img


def surface_in(value):
    """
    Reads one or two surfaces from GIFTI files

    To read both surfaces seperate them with an @ sign

    :param value: input filename (or '@'-separated filenames)
    :return: tuple with left and right surface
    """
    if '@' in value:
        fn1, fn2 = value.split('@')
        res1 = surface_in(fn1)
        res2 = surface_in(fn2)
        if res1[0] is not None and res2[0] is not None:
            raise argparse.ArgumentTypeError(f"Both surface files in {value} provide a left hemisphere")
        if res1[1] is not None and res2[1] is not None:
            raise argparse.ArgumentTypeError(f"Both surface files in {value} provide a right hemisphere")
        if res1[0] is None:
            return res2[0], res1[1]
        else:
            return res1[0], res2[1]
    else:
        if not op.isfile(value):
            raise argparse.ArgumentTypeError(f"Input surface file {value} not found on disk")
        surface = CorticalMesh.read(value)
        if surface.anatomy.hemisphere == 'left':
            return surface, None
        elif surface.anatomy.hemisphere == 'right':
            return None, surface
        else:
            raise argparse.ArgumentTypeError(f"Unrecognized hemisphere {surface.anatomy.hemisphere} in {value}")


def output(path, format=None):
    """
    Creates function to write provided output to a path

    Format of the output file is based on extension:
    - '.gii' -> GIFTI (use '@'-separator to store left and right hemisphere separately)
    - '.nii' -> CIFTI
    - '.nii.gz' -> NIFTI

    :param path: output filename
    :return: function writing a volume, surface, or greyordinate array
    """
    if format is None:
        if path[-4:] == '.gii':
            format = 'GIFTI'
        elif path[-4:] == '.nii':
            format = 'CIFTI'
        elif path[-7:] == '.nii.gz':
            format = 'NIFTI'
        else:
            raise argparse.ArgumentTypeError(f"Extension of output filename {path} not recognized")
    if format not in ('GIFTI', 'CIFTI', 'NIFTI'):
        raise argparse.ArgumentTypeError(f"Extension format {format} not recognized")

    def writer(obj):
        """
        Writes given object to the file.

        Can be one of:
        - Nifti1Image (for NIFTI/CIFTI output)
        - array/axes tuple (for any output)
        - tuple with 2 surface arrays (for GIFTI/CIFTI output)

        :param obj: input object
        """
        if not isinstance(obj, tuple):
            # NIFTI Image
            if format == 'GIFTI':
                raise ValueError(f"Can not write volumetric image {obj} to GIFTI path {path}")
            elif format == 'NIFTI':
                obj.to_filename(path)
            else:
                arr, axes = _nifti2cifti(obj)
                cifti2.Cifti2Image(arr, header=axes).to_filename(path)
            return

        part1, part2 = obj
        if isinstance(part2, np.ndarray):
            # surface arrays
            if format == 'NIFTI':
                raise ValueError(f"Can not write surface array to NIFTI path {path}")
            elif format == 'GIFTI':
                if part1.size == 0 and part2.size == 0:
                    raise ValueError("Only empty surface arrays provided")
                if part1.size == 0 or part2.size == 0:
                    if '@' in path:
                        raise ValueError("Single surface array provided to 2 GIFTI files")
                    if part1.size == 0:
                        write_gifti(path, [part2], 'CortexRight')
                    else:
                        write_gifti(path, [part1], 'CortexLeft')
                else:
                    fn1, fn2 = path.split('@')
                    write_gifti(fn1, [part1], 'CortexLeft')
                    write_gifti(fn2, [part2], 'CortexRight')
            else:
                arr, axes = _gifti2cifti(part1, part2)
                cifti2.Cifti2Image(arr, header=axes).to_filename(path)
        elif isinstance(part1, tuple):
            # surface labels
            arr1, table1 = part1
            arr2, table2 = part2
            if format == 'NIFTI':
                raise ValueError(f"Can not write surface array to NIFTI path {path}")
            elif format == 'GIFTI':
                if arr1.size == 0 and arr2.size == 0:
                    raise ValueError("Only empty surface arrays provided")
                if arr1.size == 0 or arr2.size == 0:
                    if '@' in path:
                        raise ValueError("Single surface array provided to 2 GIFTI files")
                    if arr1.size == 0:
                        write_gifti(path, [arr2], 'CortexRight', color_map=table2)
                    else:
                        write_gifti(path, [arr1], 'CortexLeft', color_map=table1)
                else:
                    fn1, fn2 = path.split('@')
                    write_gifti(fn1, [arr1], 'CortexLeft', color_map=table1)
                    write_gifti(fn2, [arr2], 'CortexRight', color_map=table2)
            else:
                if table1 != table2:
                    raise ValueError("Label tables should be the same for both surfaces to " +
                                     "combine into single CIFTI file")
                arr, axes = _gifti2cifti(arr1, arr2, table=table1)
                cifti2.Cifti2Image(arr, header=axes).to_filename(path)
        else:
            if format == 'CIFTI':
                cifti2.Cifti2Image(part1, header=part2).to_filename(path)
            elif format == 'NIFTI':
                _cifti2nifti(part1, part2).to_filename(path)
            else:
                try:
                    (a1, a2), table = _cifti2gifti(part1, part2, get_label=True)
                    as_gifti = ((a1, table), (a2, table))
                except ValueError:
                    as_gifti = _cifti2gifti(part1, part2)
                writer(as_gifti)
    return writer
