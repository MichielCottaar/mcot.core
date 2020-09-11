from mcot.core._scripts.MDE import spherical_mean
import numpy as np
import nibabel as nib


def test_harmonics():
    bvecs = np.array([[1, 1, 1], [-1, -1, -1], [0, 0, 1], [1, 0, 0], [0, 1, 0]])
    data = np.random.rand(2, 5)
    mod = spherical_mean.SPHModel(data, bvecs, order=0)
    assert mod.ngrad == 5
    assert mod.ncomponents == 1
    assert mod.diff_mat.shape == (5, 1)
    assert abs(mod.diff_mat - 1).max() < 1e-8

    mod = spherical_mean.SPHModel(data, bvecs, order=2)
    assert mod.ngrad == 5
    assert mod.ncomponents == 6
    assert mod.diff_mat.shape == (5, 6)
    assert abs(mod.diff_mat[0, :] - mod.diff_mat[1, :]).max() < 1e-8
    assert abs(mod.diff_mat[:, 0] - 1).max() < 1e-8


def get_data(bvecs, params, order):
    mod = spherical_mean.SPHModel(np.zeros((1, bvecs.shape[0])), bvecs, order=order)
    return nib.Nifti1Image(mod.diff_mat.dot(params)[None, None, None, :], np.eye(4))


def test_run():
    bvecs = np.random.randn(30, 3)
    data = nib.Nifti1Image(np.random.rand(2, 2, 2, 30), np.eye(4))
    assert (spherical_mean.run_single(data).get_fdata() == np.mean(np.asanyarray(data.dataobj), -1)).all()
    assert (spherical_mean.run_single(data, bvecs=bvecs).get_fdata() == np.mean(np.asanyarray(data.dataobj), -1)).all()
    assert abs(np.asanyarray(spherical_mean.run_single(data, 'shm', bvecs=bvecs, order=0).dataobj) - np.mean(np.asanyarray(data.dataobj), -1)).max() < 1e-8

    params = np.zeros(6)
    params[0] = 2.
    data = get_data(bvecs, params, order=2)
    assert abs(data.get_fdata() - 2).max() < 1e-8
    assert abs(spherical_mean.run_single(data, 'shm', bvecs=bvecs, order=2).get_fdata() - 2).max() < 1e-8

    for _ in range(3):
        params = np.random.randn(6)
        data = get_data(bvecs, params, 2)
        assert abs(spherical_mean.run_single(data, 'shm', bvecs=bvecs, order=2).get_fdata() - params[0]).max() < 1e-8

        params = np.random.randn(15)
        data = get_data(bvecs, params, 4)
        assert abs(spherical_mean.run_single(data, 'shm', bvecs=bvecs, order=4).get_fdata() - params[0]).max() < 1e-8
