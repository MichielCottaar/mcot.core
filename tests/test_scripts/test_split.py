from fsl.utils.platform import platform
from fsl.utils.tempdir import tempdir
from fsl.utils.run import runfsl, run
import os.path as op
import pytest
import nibabel as nib
from numpy.testing import assert_allclose
from mcot.core._scripts.split import submit
import os
from argparse import ArgumentParser


@pytest.mark.skipif(platform.fsldir is None or not op.isdir(platform.fsldir),
                    reason='FSLDIR is not properly set up')
def test_fslmaths():
    std_dir = op.join(platform.fsldir, 'data', 'standard')
    mask = op.join(std_dir, 'MNI152_T1_2mm_brain_mask.nii.gz')
    assert op.isfile(mask)
    img1 = op.join(std_dir, 'MNI152_T1_2mm_b0.nii.gz')
    assert op.isfile(img1)
    img2 = op.join(std_dir, 'MNI152_T1_2mm.nii.gz')
    assert op.isfile(img2)
    with tempdir():
        cmd = ['fslmaths', img1, '-add', img2, '-mul', mask, 'reference.nii.gz']
        runfsl(cmd)
        assert op.isfile('reference.nii.gz')
        cmd[-2] = 'MASK'
        cmd[-1] = 'splitJOBID.nii.gz'

        parser = ArgumentParser()
        submit.add_to_parser(parser)
        args = parser.parse_args(('3', mask, ' '.join(cmd)))
        _environ = dict(os.environ)
        try:
            if 'SGE_ROOT' in os.environ:
                del os.environ['SGE_ROOT']
            submit.run_from_args(args)
        finally:
            os.environ.clear()
            os.environ.update(_environ)
        assert op.isfile('split.nii.gz')
        ref = nib.load('reference.nii.gz')
        split = nib.load('split.nii.gz')
        assert_allclose(ref.get_fdata(), split.get_fdata())
        assert (ref.affine == split.affine).all()
        assert ref.header == split.header

