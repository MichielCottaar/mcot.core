import numpy as np
from mcot.core._scripts.MDE import micro_anisotropy
from numpy.testing import assert_allclose
from dipy.core import sphere
from mcot.core import spherical


def get_bvecs(nvec, nopt=300):
    pos = np.random.randn(3, nvec)
    pos /= np.sqrt(sum(pos ** 2, 0))
    hemi = sphere.HemiSphere(*pos)
    new_hemi, pot = sphere.disperse_charges(hemi, nopt)
    return new_hemi.vertices


def test_ratio():
    anis = np.random.rand(50)
    assert_allclose(
            micro_anisotropy.ratio_LTE_PTE(anis),
            micro_anisotropy.ratio_LTE_STE(anis) / micro_anisotropy.ratio_PTE_STE(anis)
    )

    assert_allclose(
            micro_anisotropy.ratio_PTE_STE(anis),
            micro_anisotropy.mean_signal(anis, -0.5)
    )

    assert_allclose(
            micro_anisotropy.ratio_LTE_STE(anis),
            micro_anisotropy.mean_signal(anis, 1)
    )

    assert_allclose(
            micro_anisotropy.mean_signal(anis, 0),
            1
    )


def test_run():
    np.random.seed(1)
    bvec = get_bvecs(200)
    lte_mat = bvec[:, :, None] * bvec[:, None, :]
    pte_mat = (np.eye(3) - lte_mat) / 2.
    ste_mat = np.eye(3) / 3.

    for _ in range(10):
        rotation = spherical.euler2mat(*(np.random.rand(3) * np.pi))
        dtensor = rotation.T.dot(np.diag([2, 0.5, 0.5]).dot(rotation))
        lte_mean = np.exp(-(dtensor * lte_mat).sum((-1, -2))).mean()
        pte_mean = np.exp(-(dtensor * pte_mat).sum((-1, -2))).mean()
        ste_mean = np.exp(-(dtensor * ste_mat).sum((-1, -2)))
        assert_allclose(
                ste_mean,
                micro_anisotropy.mean_signal(1.5, 0, np.exp(-1)),
        )
        assert_allclose(
                lte_mean,
                micro_anisotropy.mean_signal(1.5, 1, np.exp(-1)),
                rtol=0.1
        )
        assert_allclose(
                pte_mean,
                micro_anisotropy.mean_signal(1.5, -0.5, np.exp(-1)),
                rtol=0.1
        )

        assert_allclose(
                micro_anisotropy.run_single_shell((lte_mean, pte_mean), (1, -0.5))[0],
                1.5, rtol=0.4)
        assert_allclose(
                micro_anisotropy.run_single_shell((lte_mean, ste_mean), (1, 0))[0],
                1.5, rtol=0.4)
        assert_allclose(
                micro_anisotropy.run_single_shell((pte_mean, ste_mean), (-0.5, 0))[0],
                1.5, rtol=0.4)

        assert_allclose(
                micro_anisotropy.run_single_shell((lte_mean, pte_mean), (1, -0.5), bvals=2)[0],
                0.75, rtol=0.4)

        assert_allclose(
                micro_anisotropy.ratio_LTE_PTE(
                        micro_anisotropy.run_single_shell((lte_mean, pte_mean), (1, -0.5))[0],
                ),
                lte_mean / pte_mean, rtol=1e-3)
        assert_allclose(
                micro_anisotropy.ratio_LTE_STE(
                        micro_anisotropy.run_single_shell((lte_mean, ste_mean), (1, 0))[0],
                ),
                lte_mean / ste_mean, rtol=1e-3)
        assert_allclose(
                micro_anisotropy.ratio_PTE_STE(
                        micro_anisotropy.run_single_shell((pte_mean, ste_mean), (-0.5, 0))[0],
                ),
                pte_mean / ste_mean, rtol=1e-3)
