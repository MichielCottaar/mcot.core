"""
Tests the concatenation of bvals/bvecs/SIDECAR files into a single XPS structure
"""
from pytest import raises
from mcot.core._scripts.sidecar import merge as sidecar_merge
from mcot.core.sidecar import AcquisitionParams, concat
from subprocess import call
import os.path as op
import sys
from numpy.testing import assert_allclose


def test_parser():
    f_parse = sidecar_merge.parse_args
    with raises(SystemExit):
        f_parse(())
    with raises(ValueError):
        f_parse(('output.nii.gz', ))
    assert f_parse(('output.json', )) == ('output.json', [])
    assert f_parse(('output.json', '-X', 'test.mat')) == ('output.json', [('SIDE', ['test.mat'])])
    assert (f_parse(('output.json', '-X', 'test.mat', '--LTE', 'bvals', 'bvecs')) ==
            ('output.json', [('SIDE', ['test.mat']), ('LTE', ['bvals', 'bvecs'])]))
    assert (f_parse(('output.json', '-X', 'test.mat', '--PTE', 'bvals', 'bvecs')) ==
            ('output.json', [('SIDE', ['test.mat']), ('PTE', ['bvals', 'bvecs'])]))
    assert (f_parse(('output.json', '-X', 'test.mat', '--STE', 'bvals')) ==
            ('output.json', [('SIDE', ['test.mat']), ('STE', ['bvals'])]))
    with raises(ValueError):
        f_parse(('output.json', '-X', '-L', 'bvals', 'bvecs'))
    with raises(ValueError):
        f_parse(('output.json', '-X', 'test.mat', 'test2.json', '-L', 'bvals', 'bvecs'))
    with raises(ValueError):
        f_parse(('output.json', '-X', 'test.mat', '-L', 'bvals'))


def test_run_script():
    directory = op.join(op.split(__file__)[0], 'sidecar_data')
    out_fn = op.join(directory, 'test.json')
    call([
        'mcot', 'sidecar.merge', out_fn,
        '-X', op.join(directory, '..', 'test_utils', 'test_xps.mat'),
        '-L', op.join(directory, '..', 'test_utils', 'bvals'), op.join(directory, '..', 'test_utils', 'bvecs')
    ])

    xps_new = AcquisitionParams.read(out_fn)
    xps_part = AcquisitionParams.read(op.join(directory, '..', 'test_utils', 'test_xps.mat'))
    xps_ref = concat(xps_part, xps_part)
    assert xps_new.n == 8
    assert xps_ref.n == 8
    assert_allclose(xps_new['bt'][:4], xps_ref['bt'][:4])
    assert_allclose(xps_new['b'][:4], xps_ref['b'][:4])
    assert_allclose(xps_new['bt'][4:], xps_ref['bt'][4:] / 1e6)
    assert_allclose(xps_new['b'][4:], xps_ref['b'][4:] / 1e6)

    call([
        'rm', out_fn
    ])
