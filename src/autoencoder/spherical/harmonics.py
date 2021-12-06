import numpy as np
from scipy.special import sph_harm


def convert_cart_to_s2(points_cart: np.ndarray):
    """Covert array of points from Cartesian to S2 coordinate system.

    Arguments:
        points_cart (numpy array N x 3): array of points in Cartesian coordinate system.
    Returns:
        numpy array N x 2: array of points in S2 coordinate system (where r=1)
    """
    x, y, z = (points_cart[:, 0], points_cart[:, 1], points_cart[:, 2])

    # Code from https://github.com/dipy/dipy/blob/0f2dad23ea25aff3f514e4e8d69a1221264110ec/dipy/core/geometry.py#L101
    r = np.sqrt(x * x + y * y + z * z)
    theta = np.arccos(np.divide(z, r, where=r > 0))
    theta = np.where(r > 0, theta, 0.0)
    phi = np.arctan2(y, x)
    _, theta, phi = np.broadcast_arrays(r, theta, phi)

    return np.transpose([theta, phi])


def sh_basis_real(s2_coord, L, even=True):
    """Real spherical harmonic basis of even degree.

    Arguments:
        s2_coord (numpy array): S2 points coordinates
        L (int): spherical harmonic degree
        even (bool): flag indicating whether to compute only spherical harmonics of even degree
    Returns:
        numpy array: spherical harmonic bases (each column is a basis)
    """
    s = 2 if even else 1
    n_sph = np.sum([2 * i + 1 for i in range(0, L + 1, s)])
    Y = np.zeros((s2_coord.shape[0], n_sph), dtype=np.float32)
    n_sph = 0

    for i in range(0, L + 1, s):
        ns, ms = np.zeros(2 * i + 1) + i, np.arange(-i, i + 1)
        Y_n_m = sph_harm(ms, ns, s2_coord[:, 1:2], s2_coord[:, 0:1])

        # Convert complex harmonic to real.
        if i > 0:
            Y_n_m[:, 0:i] = (
                np.sqrt(2) * np.power(-1, np.arange(0, i)) * np.imag(Y_n_m[:, :i:-1])
            )
            Y_n_m[:, (i + 1) :] = (
                np.sqrt(2)
                * np.power(-1, np.arange(1, i + 1))
                * np.real(Y_n_m[:, (i + 1) :])
            )
        Y[:, n_sph : n_sph + 2 * i + 1] = Y_n_m
        n_sph += 2 * i + 1

    return Y


def gram_schmidt_sh_inv(Y, L, n_iters=1000, even=True):
    """Inversion of spherical harmonic basis with Gram-Schmidt orthonormalization process.

    Arguments:
        Y (numpy array): Spherical harmonics, real or complex
        L (int): spherical harmonic degree
        n_iters (float): number of iterations for degree shuffling
    Returns:
        numpy array: inverted spherical harmonic bases
    """
    s = 2 if even else 1
    Y_inv_final = np.zeros_like(Y.T)

    for k in range(n_iters):
        order = []
        count_h = 0

        for i in range(0, L + 1, s):
            order_h = count_h + np.arange(0, 2 * i + 1)
            np.random.shuffle(order_h)
            order.extend(list(order_h))
            count_h += 2 * i + 1

        deorder = np.argsort(order)
        Y_inv = np.zeros_like(Y.T)

        for i in range(Y_inv.shape[0]):
            Y_inv[i, :] = Y[:, order[i]]
            for j in range(0, i):
                if np.sum(Y_inv[j, :] ** 2) > 1.0e-8:
                    Y_inv[i, :] -= (
                        np.sum(Y_inv[i, :] * Y_inv[j, :])
                        / (np.sum(Y_inv[j, :] ** 2) + 1.0e-8)
                        * Y_inv[j, :]
                    )
            Y_inv[i, :] /= np.sqrt(np.sum(Y_inv[i, :] ** 2))
        Y_inv_final += Y_inv[deorder, :]
    Y_inv_final /= n_iters

    n = np.dot(Y[:, 0:1].T, Y[:, 0:1])[0, 0]
    Y_inv_final /= np.sqrt(n)

    return Y_inv_final
