#!/usr/bin/env python
"""
Smooths the values in the GIFTI or CIFTI file across the surface
"""
from loguru import logger
from mcot.core import scripts
import numpy as np
from nibabel import cifti2
from mcot.core.surface.cortical_mesh import BrainStructure


def smooth(surface, array, width, vertices=None, axis=-1):
    """
    Smooths the array over the surface

    :param surface: Mesh representing the cortical surface
    :param array: values on the vertices of the array
    :param width: width of the smoothing kernel (in mm)
    :param vertices: indices indicating which vertices have values (default: all vertices)
    :return:
    """
    if width <= 0:
        return array
    if axis < 0:
        axis = axis + array.ndim
    if vertices is not None:
        surface = surface[vertices]
    neighbour_distance = surface.graph_point_point('distance').tocoo()

    min_dist = np.median(neighbour_distance.tocoo().data) / 6
    nsteps = int(round(width ** 2 / min_dist ** 2))
    weight = neighbour_distance.copy()
    if nsteps <= 1:
        nsteps = 1
    start = width ** 2 / nsteps
    weight.data = np.append(
            np.exp(-neighbour_distance.data ** 2 / (2 * start)),
            np.ones(surface.nvertices)
    )
    weight.row = np.append(
            weight.row,
            np.arange(surface.nvertices)
    )
    weight.col = np.append(
            weight.col,
            np.arange(surface.nvertices)
    )
    weight = weight.tocsr()

    trans_axes = [axis] + list(range(axis)) + list(range(axis + 1, array.ndim))
    trans_array = np.transpose(array, trans_axes)
    summed = np.array(weight.sum(0)).flatten()[:, None]
    trans_reshaped = trans_array.reshape((trans_array.shape[0], -1))
    for _ in range(nsteps):
        trans_reshaped = weight.dot(trans_reshaped) / summed
    new_axes = list(range(1, array.ndim))
    new_axes.insert(axis, 0)
    transposed = np.transpose(trans_reshaped.reshape(trans_array.shape), new_axes)
    assert transposed.shape == array.shape
    return transposed


def smooth_cifti(arr, axes, surfaces, width, overwrite=False):
    """
    Smooths a CIFTI array across provided surfaces

    :param arr: input array (overwritten if `overwrite` is True)
    :param axes: CIFTI axes for the array
    :param surfaces: tuple of left & right surface (None if not smoothing on that surface)
    :param width: width of the smoothing kernel (in mm)
    :param overwrite: if True overwrites the input array (saves memory)
    :return: smoothed array
    """
    if overwrite:
        res = arr
    else:
        res = arr.copy()

    smoothed = False
    for dim, bm in enumerate(axes):
        if isinstance(bm, cifti2.BrainModelAxis):
            selector = [slice(None)] * arr.ndim
            for name, slc, bm_part in bm.iter_structures():
                name = BrainStructure.from_string(name)
                selector[dim] = slc
                if name == 'CortexLeft' and surfaces[0] is not None:
                    logger.info(f'Smoothing left hemisphere across dimension {dim}')
                    smoothed = True
                    res[tuple(selector)] = smooth(surfaces[0], arr[tuple(selector)], width,
                                                  bm_part.vertex, axis=dim)
                elif name == 'CortexRight' and surfaces[1] is not None:
                    logger.info(f'Smoothing right hemisphere across dimension {dim}')
                    smoothed = True
                    res[tuple(selector)] = smooth(surfaces[1], arr[tuple(selector)], width,
                                                  bm_part.vertex, axis=dim)
    if not smoothed:
        raise ValueError("Provided surfaces did not match the hemispheres in the input file; no smoothing performed")
    return res


def run_from_args(args):
    """
    Runs the script based on a Namespace containing the command line arguments
    """
    arr, axes = args.input
    res = smooth_cifti(arr, axes, args.surface, args.width, overwrite=True)
    args.output((res, axes))


def add_to_parser(parser):
    """
    Creates the parser of the command line arguments
    """
    parser.add_argument('input', type=scripts.greyordinate_in,
                        help='CIFTI or GIFTI file with the surface array (in a GIFTI file zeros are masked out)')
    parser.add_argument('surface', type=scripts.surface_in,
                        help=".surf.gii file with the surfaces (both can be provided if separated by '@'")
    parser.add_argument('width', type=float, help='smoothing kernel width in mm')
    parser.add_argument('output', type=scripts.output,
                        help='CIFTI or GIFTI file with the smoothed output (ignoring zeroes)')
