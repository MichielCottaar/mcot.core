import datetime
from nibabel import gifti
import nibabel as nib
from loguru import logger
import numpy as np
import colorcet as cc
from mcot.core.surface.cortical_mesh import BrainStructure


def correct_type(arr: np.ndarray):
    """
    Ensures that the data has a type expected in GIFTI

    :param arr: array to be stored in a GIFTI file
    :return: array of the corrected type
    """
    if arr.dtype == np.float32 or arr.dtype == np.int32 or arr.dtype == np.uint8:
        return arr
    if arr.dtype == np.float:
        return arr.astype(np.float32)
    if arr.dtype == np.int:
        return arr.astype(np.int32)
    if arr.dtype == np.uint:
        return arr.astype(np.in32)
    return arr


def write_gifti(filename, arr_list, brain_structure, intent_list=None, color_map=None,
                meta_list=None, **kwargs):
    """
    Writes data to a GIFTI file

    :param filename: output filename
    :param arr_list: list of arrays to be stored
    :param brain_structure: 'CortexLeft' or 'CortexRight'
    :param intent_list: intent of each array (list of same length as arr_list)
    :param color_map: None for non-label giftis, 'default' for default qualitative colour map, dict mapping value to RGBA values otherwise
    :param meta_list: list of dictionaries with the array metadata
    :param kwargs: additional values to be stored to the meta data
    """
    logger.info('writing to %s as GIFTI' % filename)
    if intent_list is None:
        intent_list = ['NIFTI_INTENT_NONE' if color_map is None else 'NIFTI_INTENT_LABEL'] * len(arr_list)
    if meta_list is None:
        meta_list = [{} for _ in arr_list]
    if len(intent_list) != len(arr_list):
        raise ValueError("Number of intents does not match number of arrays")

    if isinstance(brain_structure, str):
        brain_structure = BrainStructure.from_string(brain_structure, issurface=True)
    meta_dict = brain_structure.gifti
    meta_dict.update({'Date': str(datetime.datetime.now()), 'encoding': 'XML'})
    meta_dict.update(kwargs)
    meta = gifti.GiftiMetaData.from_dict(meta_dict)

    if color_map == 'default':
        color_map = {}
    if color_map is not None:
        labels = np.unique(np.concatenate([np.unique(arr) for arr in arr_list]))
        colour_sequence = cc.glasbey

        for label in labels:
            if label not in color_map:
                color_map[label] = (str(label), next(colour_sequence))

        labeltable = gifti.GiftiLabelTable()
        for value, (text, rgba) in color_map.items():
            labeltable.labels.append(gifti.GiftiLabel(value, *rgba))
            labeltable.labels[-1].label = str(text)
    else:
        labeltable = None

    img = gifti.GiftiImage(meta=meta, labeltable=labeltable)
    for arr, intent, arr_meta in zip(arr_list, intent_list, meta_list):
        arr_meta_dict = dict(meta_dict)
        arr_meta_dict.update(arr_meta)
        img.add_gifti_data_array(
                gifti.GiftiDataArray(correct_type(arr), intent, meta=arr_meta_dict)
        )
    for da in img.darrays:
        da.encoding = 2  # Base64Binary
    nib.save(img, filename)
