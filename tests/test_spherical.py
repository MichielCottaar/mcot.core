from mcot.core import spherical
import numpy as np
from numpy.testing import assert_allclose


def test_known():
    assert (1, 0, 0) == spherical.cart2spherical(0, 0, 1)
    assert (1, 0, np.pi) == spherical.cart2spherical(0, 0, -1)
    assert (1, 0, np.pi / 2) == spherical.cart2spherical(1, 0, 0)
    assert (1, np.pi / 2, np.pi / 2) == spherical.cart2spherical(0, 1, 0)

    assert (2, 0, 0) == spherical.cart2spherical(0, 0, 2)
    assert (2, 0, np.pi / 2) == spherical.cart2spherical(2, 0, 0)
    assert (2, np.pi / 2, np.pi / 2) == spherical.cart2spherical(0, 2, 0)

    assert abs(np.array((0, 0, 1)) - spherical.spherical2cart(1, 0, 0)).max() < 1e-8
    assert abs(np.array((0, 0, 1)) - spherical.spherical2cart(1, np.pi / 2, 0)).max() < 1e-8
    assert abs(np.array((1, 0, 0)) - spherical.spherical2cart(1, 0, np.pi / 2)).max() < 1e-8
    assert abs(np.array((0, 1, 0)) - spherical.spherical2cart(1, np.pi / 2, np.pi / 2)).max() < 1e-8


def test_random():
    for _ in range(10):
        xyz = np.random.randn(3)
        assert abs(xyz - spherical.spherical2cart(*spherical.cart2spherical(*xyz))).max() < 1e-8
        xyz = np.random.randn(3, 10)
        assert abs(xyz - spherical.spherical2cart(*spherical.cart2spherical(*xyz))).max() < 1e-8
        xyz = np.random.randn(3, 2, 2)
        assert abs(xyz - spherical.spherical2cart(*spherical.cart2spherical(*xyz))).max() < 1e-8


def test_single_rotation():
    for axis in range(3):
        assert_allclose(spherical.single_rotmat(0, axis), np.eye(3), atol=1e-8)
        assert_allclose(spherical.single_rotmat(np.pi * 2, axis), np.eye(3), atol=1e-8)
        res = -np.eye(3)
        res[axis, axis] *= -1
        assert_allclose(spherical.single_rotmat(np.pi, axis), res, atol=1e-8)

        res = np.zeros((3, 3))
        ax1 = (axis + 1) % 3
        ax2 = (axis + 2) % 3
        res[axis, axis] = 1
        res[ax1, ax2] = 1
        res[ax2, ax1] = -1
        assert_allclose(spherical.single_rotmat(np.pi / 2, axis), res, atol=1e-8)


def test_euler2mat():
    assert_allclose(spherical.euler2mat(0, 0, 0), np.eye(3), atol=1e-8)
    for phi, psi in np.random.randn(5, 2):
        assert_allclose(spherical.euler2mat(phi, 0, psi).dot([0, 0, 1]), [0, 0, 1], atol=1e-8)
        assert_allclose(spherical.euler2mat(phi, np.pi / 2, psi).dot([0, 0, 1])[-1], 0, atol=1e-8)
        assert_allclose(spherical.euler2mat(phi, np.pi, psi).dot([0, 0, 1]), [0, 0, -1], atol=1e-8)
    for phi, theta, psi in np.random.randn(5, 3):
        mat = spherical.euler2mat(phi, theta, psi)
        assert_allclose(mat.dot(mat.T), np.eye(3), atol=1e-8)
        assert_allclose(mat.T.dot(mat), np.eye(3), atol=1e-8)
        assert_allclose(mat.dot([0, 0, 1]),
                        (np.cos(phi) * np.sin(theta), np.sin(phi) * np.sin(theta), np.cos(theta)))


def test_mat2euler():
    assert_allclose(
            abs(spherical.euler2mat(*spherical.mat2euler(np.eye(3))).dot(np.eye(3))),
            np.eye(3), atol=1e-8
    )
    assert_allclose(
            abs(spherical.euler2mat(*spherical.mat2euler(np.eye(3)[::-1])).dot(np.eye(3)[::-1])),
            np.eye(3), atol=1e-8
    )

    for _ in range(10):
        rot_mat = spherical.euler2mat(*np.random.randn(3))
        res = spherical.euler2mat(*spherical.mat2euler(rot_mat))
        assert_allclose(
                rot_mat, res, atol=1e-8
        )


def test_clean_euler():
    for phi, theta, psi in np.random.randn(50, 3) * 100:
        phi2, theta2, psi2 = spherical.clean_euler(phi, theta, psi)
        assert_allclose(
                spherical.euler2mat(phi, theta, psi),
                spherical.euler2mat(phi2, theta2, psi2),
        )
        assert phi2 >= -np.pi
        assert phi2 <= np.pi
        assert theta2 >= 0
        assert theta2 <= np.pi
        assert psi2 >= -np.pi
        assert psi2 <= np.pi
    phi, theta, psi = np.random.randn(3, 10, 10) * 100
    phi2, theta2, psi2 = spherical.clean_euler(phi, theta, psi)
    assert_allclose(
            spherical.euler2mat(phi, theta, psi),
            spherical.euler2mat(phi2, theta2, psi2),
    )
    assert (phi2 >= -np.pi).all()
    assert (phi2 <= np.pi).all()
    assert (theta2 >= 0).all()
    assert (theta2 <= np.pi).all()
    assert (psi2 >= -np.pi).all()
    assert (psi2 <= np.pi).all()
