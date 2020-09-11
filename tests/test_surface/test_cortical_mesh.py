from mcot.core.surface.cortical_mesh import BrainStructure
from mcot.core.surface.test_data import read_fsaverage_surface


def test_brainstructure():
    for primary in ['cortex', 'cerebellum']:
        for secondary in [None, 'white', 'pial']:
            for gtype in [None, 'volume', 'surface']:
                for orientation in ['left', 'right', 'both']:
                    bst = BrainStructure(primary, secondary, orientation, gtype)
                    print(bst.cifti)
                    assert bst.cifti == 'CIFTI_STRUCTURE_%s%s' % (primary.upper(), '' if orientation == 'both' else '_' + orientation.upper())
                    assert bst.gifti['AnatomicalStructurePrimary'][:len(primary)] == primary.capitalize()
                    assert len(bst.gifti) == (1 if secondary is None else 2)
                    if secondary is not None:
                        assert bst.gifti['AnatomicalStructureSecondary'] == secondary.capitalize()
                    assert bst == BrainStructure(primary, secondary, orientation, gtype)
                    assert bst == bst
                    assert bst != BrainStructure('Thalamus', secondary, orientation, gtype)
                    if secondary is None:
                        assert bst == BrainStructure(primary, 'midplane', orientation, gtype)
                    else:
                        assert bst != BrainStructure(primary, 'midplane', orientation, gtype)
                    if (gtype == 'volume' and primary == 'cortex') or (gtype == 'surface' and primary != 'cortex'):
                        assert BrainStructure.from_string(bst.cifti) != bst
                    else:
                        assert BrainStructure.from_string(bst.cifti) == bst
                    assert BrainStructure.from_string(bst.cifti).secondary is None


def test_read():
    surf = read_fsaverage_surface()
    assert surf.ndim == 3
    assert surf.nvertices == 32492
    assert surf.nfaces == 64980
    assert surf.anatomy.primary == 'cortex'
    assert surf.anatomy.hemisphere == 'left'
    assert surf.anatomy.secondary == 'pial'

