"""Extends the functionality defined in mesh.py to include anatomical information

For any `CorticalMesh` this information is stored in the `anatomy` property. This property is a `BrainStructure` object.
"""
from .mesh import Mesh2D
import nibabel as nib
from six import string_types
import datetime
from pathlib import Path


class BrainStructure(object):
    """Which brain structure does the parent object describe?
    """
    def __init__(self, primary, secondary=None, hemisphere='both', geometry=None):
        """Creates a new brain structure

        :param primary: Name of the brain structure (e.g. cortex, thalamus)
        :param secondary: Further specification of which part of the brain structure is described (e.g. 'white' or
        'pial' for the cortex)
        :param hemisphere: which hemisphere is the brain structure in ('left', 'right', or 'both')
        :param geometry: does the parent object describe the 'volume' or the 'surface'
        """
        self.primary = primary.lower()
        self.secondary = None if secondary is None else secondary.lower()
        self.hemisphere = hemisphere.lower()
        if geometry not in (None, 'surface', 'volume'):
            raise ValueError("geometry should be set to surface or volume")
        self.geometry = geometry

    def __eq__(self, other):
        """Two brain structures are equal if they could describe the same structure
        """
        if isinstance(other, string_types):
            other = self.from_string(other)
        match_primary = (self.primary == other.primary or self.primary == 'all' or other.primary == 'all' or
                         self.primary == other.geometry or self.geometry == other.primary)
        match_hemisphere = self.hemisphere == other.hemisphere
        match_secondary = (self.secondary is None or other.secondary is None or self.secondary == other.secondary)
        match_geometry = (self.geometry is None or other.geometry is None or self.geometry == other.geometry)
        return match_primary and match_hemisphere and match_secondary and match_geometry

    @property
    def gifti(self, ):
        """Returns the keywords needed to define the surface in the meta information of a GIFTI file
        """
        main = self.primary.capitalize() + ('' if self.hemisphere == 'both' else self.hemisphere.capitalize())
        res = {'AnatomicalStructurePrimary': main}
        if self.secondary is not None:
            res['AnatomicalStructureSecondary'] = self.secondary.capitalize()
        return res

    def __str__(self, ):
        """Returns a short description of the brain structure
        """
        if self.secondary is None:
            return self.primary.capitalize() + self.hemisphere.capitalize()
        else:
            return "%s%s(%s)" % (self.primary.capitalize(), self.hemisphere.capitalize(), self.secondary)

    @property
    def cifti(self, ):
        """Returns a description of the brain structure needed to define the surface in a CIFTI file
        """
        return 'CIFTI_STRUCTURE_' + self.primary.upper() + ('' if self.hemisphere == 'both' else ('_' + self.hemisphere.upper()))

    @classmethod
    def from_string(cls, value, issurface=None):
        """Parses a string to find out which brain structure is being described

        :param value: string to be parsed
        :param issurface: defines whether the object describes the volume or surface of the brain structure (default: surface if the brain structure is the cortex volume otherwise)
        """
        if '_' in value:
            items = [val.lower() for val in value.split('_')]
            if items[-1] in ['left', 'right', 'both']:
                hemisphere = items[-1]
                others = items[:-1]
            elif items[0] in ['left', 'right', 'both']:
                hemisphere = items[0]
                others = items[1:]
            else:
                hemisphere = 'both'
                others = items
            if others[0] in ['nifti', 'cifti', 'gifti']:
                others = others[2:]
            primary = '_'.join(others)
        else:
            low = value.lower()
            if 'left' == low[-4:]:
                hemisphere = 'left'
                primary = low[:-4]
            elif 'right' == low[-5:]:
                hemisphere = 'right'
                primary = low[:-5]
            elif 'both' == low[-4:]:
                hemisphere = 'both'
                primary = low[:-4]
            else:
                hemisphere = 'both'
                primary = low
        if issurface is None:
            issurface = primary == 'cortex'
        if primary == '':
            primary = 'all'
        return cls(primary, None, hemisphere, 'surface' if issurface else 'volume')


class CorticalMesh(Mesh2D):
    """
    Describes a cortical mesh of a brain region

    Besides all the normal mesh operations defined in mesh.Mesh2D this object also contains information about the brain structure in the `anatomy` attribute
    This information is stored/used when reading/writing the file.
    """
    def __init__(self, vertices, faces, flip_normal=False, anatomy=None):
        """
        Creates a new CorticalMesh

        :param vertices: (M, N) array with the vertices of the curve in M-dimensional space.
        :param faces: (2, K) index array with all the line segments.
        :param flip_normal: flips the normal when it is computed (used by `Mesh2D.apply_affine`, should normally not 
        be used)
        :param anatomy: Describes which brain structure this cortical mesh represents
        :type anatomy: BrainStructure
        """
        self.vertices = vertices
        self.faces = faces
        self.flip_normal = flip_normal
        if anatomy is None:
            anatomy = BrainStructure('unknown', None, 'both', 'surface')
        self.anatomy = anatomy
        assert self.ndim == 3

    @classmethod
    def read(cls, gifti_filename):
        """
        Reads a cortical mesh from a surface gifti file (i.e. ending with .surf.gii).

        :param gifti_filename: input filename or Gifti image
        """
        res = Mesh2D.read(gifti_filename)
        gifti_obj = nib.load(str(gifti_filename)) if isinstance(gifti_filename, string_types + (Path, )) else gifti_filename
        res.anatomy = get_brain_structure(gifti_obj)
        return CorticalMesh(res.vertices, res.faces, res.flip_normal,
                            res.anatomy)

    def write(self, gifti_filename, scalar_arr=None, **kwargs):
        """
        Writes a cortical mesh to a gifti file (i.e. ending with .surf.gii)

        :param gifti_filename: output filename
        :param scalar_arr: optionally include a scalar array with same length as number of vertices (as expected by FSL's probtrackX)
        :param kwargs: any keywords are added to the meta information in the GIFTI file
        """
        use_kwargs = self.anatomy.gifti
        use_kwargs.update(kwargs)
        return super(CorticalMesh, self).write(gifti_filename, **use_kwargs)

    def write_metric(self, filename, arr_list, intent_list=None, **kwargs):
        """
        Writes list-like of AxisArrays as a Gifti file.
        """
        if intent_list is None:
            intent_list = ['NIFTI_INTENT_NONE'] * len(arr_list)
        kwargs.update({'Date': str(datetime.datetime.now()),
                       'encoding': 'XML',
                       'AnatomicalStructurePrimary': self.anatomy.gifti['AnatomicalStructurePrimary']})
        meta = gifti.GiftiMetaData.from_dict(kwargs)

        img = gifti.GiftiImage(meta=meta)
        for arr, intent in zip(arr_list, intent_list):
            if arr.shape != (self.nvertices,):
                raise ValueError('Array in array list has the wrong shape')
            img.add_gifti_data_array(gifti.GiftiDataArray.from_array(arr, intent, meta=meta))
        for da in img.darrays:
            da.encoding = 2 # Base64Binary
        gifti.write(img, filename)

    def __getitem__(self, item):
        """
        Gets the surface covering a subsection of all vertices
        """
        res = super(CorticalMesh, self).__getitem__(item)
        res.anatomy = self.anatomy
        return res


def get_brain_structure(gifti_obj):
    """
    Extracts the brain structure from a GIFTI object
    """
    primary_str = 'AnatomicalStructurePrimary'
    secondary_str = 'AnatomicalStructureSecondary'
    primary = "unknown"
    secondary = None
    for meta in [gifti_obj] + gifti_obj.darrays:
        if primary_str in meta.meta.metadata:
            primary = meta.meta.metadata[primary_str]
        if secondary_str in meta.meta.metadata:
            secondary = meta.meta.metadata[secondary_str]
    anatomy = BrainStructure.from_string(primary, issurface=True)
    anatomy.secondary = None if secondary is None else secondary.lower()
    return anatomy
