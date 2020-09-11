from mcot.core.surface import cortex
from numpy import testing
import numpy as np
import shutil
import pytest
from mcot.core.surface.test_data import read_fsaverage_cortex


def test_base():
    right, left = read_fsaverage_cortex()
    assert right.hemisphere == 'right'
    assert left.hemisphere == 'left'
    for cort in right, left:
        assert cort.primary == 'cortex'
        assert len(cort.surfaces) == 3
        for surf, name in zip(cort.surfaces, ('graywhite', 'midthickness', 'pial')):
            assert surf.anatomy.primary == "cortex"
            assert surf.anatomy.secondary == name


def test_volume():
    lower = cortex.CorticalMesh(
            np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [1., 1., 0.]]).T,
            np.array([[0, 1, 2], [1, 2, 3]]).T,
    )
    assert lower.ndim == 3
    assert lower.nvertices == 4
    assert lower.nfaces == 2
    new_vertices = lower.vertices.copy()
    new_vertices[2] += 1
    upper = cortex.CorticalMesh(
            new_vertices,
            lower.faces
    )
    layer = cortex.CorticalLayer(lower, upper)
    testing.assert_allclose(layer.wedge_volume(), 0.5)


@pytest.mark.skipif(shutil.which('wb_command') is None,
                    reason='Requires wb_command installed')
def test_volume_with_wb():
    right, left = read_fsaverage_cortex()
    with_wb = right.wedge_volume(use_wb=True, atpoint=True)
    no_wb = right.wedge_volume(use_wb=False, atpoint=True)
    norm_mean = (np.mean(with_wb - no_wb) / np.mean(with_wb + no_wb))
    norm_std = (np.std(with_wb - no_wb) / np.mean(with_wb + no_wb))
    assert abs(norm_mean) < 1e-2
    assert abs(norm_std) < 1e-1
