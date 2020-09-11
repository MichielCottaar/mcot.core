"""
Defines functions to work with spherical coordinates
"""
import numpy as np


def cart2spherical(x, y, z):
    """
    Converts to spherical coordinates

    :param x: x-component of the vector
    :param y: y-component of the vector
    :param z: z-component of the vector
    :return: tuple with (r, phi, theta)-coordinates
    """
    vectors = np.array([x, y, z])
    r = np.sqrt(np.sum(vectors ** 2, 0))
    theta = np.arccos(vectors[2] / r)
    phi = np.arctan2(vectors[1], vectors[0])
    if vectors.ndim == 1:
        if r == 0:
            phi = 0
            theta = 0
    else:
        phi[r == 0] = 0
        theta[r == 0] = 0
    return r, phi, theta


def spherical2cart(r, phi, theta):
    """
    Converts from spherical coordinates

    :param r: radius
    :param phi: angle within the x-y plane (longitude)
    :param theta: angle relative to z-axis (latitude)
    :return: tuple with (x, y, z) coordinates
    """
    return (
        r * np.cos(phi) * np.sin(theta),
        r * np.sin(phi) * np.sin(theta),
        r * np.cos(theta),
    )


def mat2euler(rot_mat):
    """
    Converts a rotation matrix to spherical coordinates

    see `spherical.euler2mat` for the relation between the rotation matrix and the euler angles

    :param rot_mat: (..., 3, 3) array
    :return: euler angles (phi, theta, psi))
    """
    phi, theta = [angle.T for angle in cart2spherical(*rot_mat.dot([0, 0, 1]).T)[1:]]

    test_mat = euler2mat(phi, theta, 0)

    psi = np.arctan2(
            (test_mat[..., 0] * rot_mat[..., 1]).sum(-1),
            (test_mat[..., 1] * rot_mat[..., 1]).sum(-1),
    )
    return phi, theta, psi


def single_rotmat(theta, axis):
    """
    Rotate around the `axis` with `theta` radians

    :param theta: Angle of the rotations
    :param axis: index of the axis (0, 1, or 2 for x, y, or z)
    :return: (3, 3) array with the rotation matrix
    """
    theta = np.asarray(theta)
    arr = np.eye(3) * np.cos(theta[..., None, None])
    arr[..., axis, axis] = 1.
    s = np.sin(theta)
    dim1 = (axis + 1) % 3
    dim2 = (axis + 2) % 3
    arr[..., dim1, dim2] = s
    arr[..., dim2, dim1] = -s
    return arr


def euler2mat(phi, theta, psi):
    """
    Computes a rotation matrix based on the Euler angles

    The z-axis is mapped to:
    euler2mat(phi, theta, _).dot([0, 0, 1] = (cos(phi) * sin(theta), sin(phi) * sin(theta), cos(theta))
    psi determines the x- and y-axes orientations in the plane perpendicular to this

    Bingham matrix is assumed to initially point in the z-direction

    :param phi: orientation of the z-axis projection on the x-y plane
    :param theta: polar angle (0 for keeping the z-axis in the z-direction,
        pi/2 for projecting the z-axis onto the x-y plane)
    :param psi: rotation of the major dispersion axis around the main fibre orientation
    :return: (..., 3, 3) array with rotation matrix
    """
    return np.matmul(np.matmul(single_rotmat(-phi, 2), single_rotmat(-theta, 1)), single_rotmat(psi, 2))


def clean_euler(phi, theta, psi):
    """
    Finds a set of angles matching the same orientation with

    - phi between -pi and pi
    - theta between 0 and pi
    - psi between -pi and pi

    :param phi: orientation of the z-axis projection on the x-y plane
    :param theta: polar angle (0 for keeping the z-axis in the z-direction,
        pi/2 for projecting the z-axis onto the x-y plane)
    :param psi: rotation of the major dispersion axis around the main fibre orientation
    :return: tuple with new (phi, theta, psi)
    """
    return mat2euler(euler2mat(phi, theta, psi))

