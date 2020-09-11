#!/usr/bin/env python
"""Masks out part of a dlabel file based on a dscalar file

if discard is set simply removes all the voxels/vertices that are <= 0.
Otherwise replace those voxels/vertices outside of the mask with a new label (called "outside_mask).
"""
from mcutils.scripts.plot.scatter_dscalar import find_common
import cifti
import numpy as np


def run(label_fn, scalar_fn, out_fn, discard=False, keep_outside_mask=False):
    """
    Masks out part of a dlabel file using a dscalar file

    :param label_fn: CIFTI dlabel file with labels
    :param scalar_fn: CIFTI dscalar file with mask
    :param out_fn: output dlabel filename
    :param discard: if True discard any unmasked voxels/vertices from the output CIFTI files rather than assigning them a new mask
    :param keep_outside_mask: if True included all voxels/vertices in the scalar mask, even if they are non-zero
    """
    # load input files
    labels, (label_ax, label_bm) = cifti.read(label_fn)
    scalar, (scalar_ax, scalar_bm) = cifti.read(scalar_fn)

    # check inputs
    if len(label_ax) != len(scalar_ax) and len(label_ax) != 1 and len(scalar_ax) != 1:
        raise ValueError('number of arrays in dlabel and dscalar do not match')
    if discard and len(scalar_ax) != 1:
        raise ValueError('Discarding voxels/vertices only works if there is only mask value per voxel/vertex')

    if discard:
        keep_label = np.zeros(len(label_bm), dtype='bool')
        scalar_mask = scalar[0] > 0
    else:
        keep_label = np.zeros(labels.shape, dtype='bool')
        scalar_mask = scalar > 0

    if keep_outside_mask:
        scalar_mask[()] = True

    # find any voxels/vertices in common between the dlabel and dscalar files and apply the mask
    for name, label_slice, label_sub_bm in label_bm.iter_structures():
        scalar_slice = scalar_bm.name == name
        idx_labels, idx_scalar = find_common(label_sub_bm, scalar_bm[scalar_slice])
        keep_label[..., label_slice][..., idx_labels] = scalar_mask[..., scalar_slice][..., idx_scalar]

    # apply the mask
    if discard:
        new_labels = labels[:, keep_label]
        new_bm = label_bm[keep_label]
    else:
        # find an unused label id
        label_id = 0
        while any(label_id in label_dict for label_dict in label_ax.label):
            label_id -= 1

        # updates label dictionary with new label
        for label_dict in label_ax.label:
            label_dict[label_id] = ('outside_mask', (0., 0., 0., 0.))

        new_labels = np.array(labels)
        new_labels[~keep_label] = label_id

        new_bm = label_bm

    # store output
    cifti.write(out_fn, new_labels, (label_ax, new_bm))


def add_to_parser(parser):
    """
    Creates the parser of the command line arguments
    """
    parser.add_argument('input', help='dlabel file with input set of labels')
    parser.add_argument('mask', help='dscalar file with input mask; all values > 0 will be considered kept')
    parser.add_argument('output', help='dlabel output file which will contain the output labels')
    parser.add_argument('-d', '--discard', action='store_true',
                        help='if set discard any unmasked voxels/vertices from the output CIFTI file;' +
                             'otherwise these unmasked voxels/vertices are kept but with a new label')
    parser.add_argument('--keep_outside_dscalar', action='store_true',
                        help='if set consider all voxels/vertices in the input, ' +
                             'but not in the mask as masked rather than outside of the mask')


def run_from_args(args):
    run(
            args.input,
            args.mask,
            args.output,
            discard=args.discard,
    )
