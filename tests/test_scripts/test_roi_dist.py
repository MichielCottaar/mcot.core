from fsl.utils.tempdir import tempdir
from pathlib import Path
from mcot.core.surface import CorticalMesh
from mcot.core import write_gifti
from mcot.core.scripts import directories
import numpy as np
import nibabel as nib
from mcot.core.surface.test_data import fsaverage_directory


def run_roi_dist_gifti(surf_fn, roi, brain_structure, project=None):
    with tempdir():
        write_gifti('roi.shape.gii', [roi], brain_structure)
        if project is not None:
            write_gifti('in_project.shape.gii', [project], brain_structure)
            cmd_project = ['-p', 'in_project.shape.gii', 'out_project.shape.gii']
        else:
            cmd_project = []
        directories(['surface', 'roi_dist_gifti', str(surf_fn), 'roi.shape.gii',
                     '-od', 'dist.shape.gii',
                     '-ocv', 'closest.shape.gii',
                     '-opv', 'pred.shape.gii',
                     ] + cmd_project)
        res = (
            nib.load('dist.shape.gii').darrays[0].data,
            nib.load('closest.shape.gii').darrays[0].data,
            nib.load('pred.shape.gii').darrays[0].data,
        )
        if project is not None:
            res = res + (nib.load('out_project.shape.gii').darrays[0].data, )
    return res


def test_roi_dist_gifti():
    surf_fn = Path(fsaverage_directory) / '100000.L.white.32k_fs_LR.surf.gii'
    surface = CorticalMesh.read(surf_fn)
    roi = np.zeros(surface.nvertices, dtype='i4')
    roi[5] = 1
    tp = np.arange(surface.nvertices) ** 2
    dist, closest, pred, projected = run_roi_dist_gifti(surf_fn, roi, surface.anatomy, project=tp)

    assert dist[5] == 0
    assert (dist[6:] > 0).all()
    assert (dist[:5] > 0).all()
    assert (closest == 5).all()
    assert pred[5] == -1
    assert (projected == 25).all()

    roi[50] = 1
    dist, closest, pred = run_roi_dist_gifti(surf_fn, roi, surface.anatomy)
    assert dist[5] == 0
    assert dist[50] == 0
    assert (dist[:5] > 0).all()
    assert (dist[6:50] > 0).all()
    assert (dist[51:] > 0).all()
    assert ((closest == 5) | (closest == 50)).all()
    assert ((projected == 25) | (projected == 2500)).all()
    assert closest[5] == 5
    assert closest[50] == 50
    assert pred[5] == -1
    assert pred[50] == -1
