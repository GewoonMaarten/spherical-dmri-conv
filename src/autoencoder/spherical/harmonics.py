import torch
from e3nn import o3


def sh_basis_real(cart_coord: torch.Tensor, L: int, symmetric: bool = True):
    """Real spherical harmonic basis of even degree.

    Arguments:
        cart_coord (torch.Tensor): cartesian point coordinates
        L (int): spherical harmonic degree
        symmetric (bool): flag indicating whether to compute only spherical harmonics of even degree
    Returns:
        torch.Tensor: spherical harmonic bases (each column is a basis)
    """
    symmetric = 2 if symmetric else 1
    n_sph = torch.sum(
        torch.tensor([2 * i + 1 for i in range(0, L + 1, symmetric)], dtype=torch.uint8)
    )
    Y = torch.zeros((cart_coord.shape[0], n_sph), dtype=torch.float32)

    alpha, beta, gamma = o3.rand_angles()
    n_sph = 0
    for i in range(0, L + 1, symmetric):
        Y_n_m = o3.spherical_harmonics(i, cart_coord[:, :3], True).float()

        rot = o3.wigner_D(i, alpha, beta, gamma).float()
        Y_n_m = torch.einsum("mn,bn->bm", rot, Y_n_m)
        Y[:, n_sph : n_sph + 2 * i + 1] = Y_n_m
        n_sph += 2 * i + 1

    return Y


def gram_schmidt_sh_inv(
    Y: torch.Tensor, L: int, n_iters: int = 1000, symmetric: bool = True
):
    """Inversion of spherical harmonic basis with Gram-Schmidt orthonormalization process.

    Arguments:
        Y (torch.Tensor): Spherical harmonics, real or complex
        L (int): spherical harmonic degree
        n_iters (int): number of iterations for degree shuffling
        symmetric (bool): flag indicating whether to compute only spherical harmonics of even degree
    Returns:
        torch.Tensor: inverted spherical harmonic bases
    """

    def project(v, u):
        return (v * u).sum() / ((u * u).sum() + 1.0e-8) * u

    symmetric = 2 if symmetric else 1
    Y_inv_final = torch.zeros_like(Y.T, device=Y.device)

    for _ in range(n_iters):
        order = list()
        count_h = 0

        for i in range(0, L + 1, symmetric):
            order_h = count_h + torch.randperm(2 * i + 1)
            order.append(order_h)
            count_h += 2 * i + 1

        order = torch.cat(order)
        deorder = torch.argsort(order)

        Y_inv = torch.zeros_like(Y.T, device=Y.device)
        for i in range(Y_inv.shape[0]):
            Y_inv[i, :] = Y[:, order[i]]
            for j in range(0, i):
                if torch.sum(Y_inv[j, :] ** 2) > 1.0e-8:
                    Y_inv[i, :] -= project(Y_inv[i, :], Y_inv[j, :])

            Y_inv[i, :] /= torch.sqrt((Y_inv[i, :] * Y_inv[i, :]).sum())
        Y_inv_final += Y_inv[deorder, :]
    Y_inv_final /= n_iters

    n = torch.matmul(Y[:, 0:1].T, Y[:, 0:1])[0, 0]
    Y_inv_final /= torch.sqrt(n)

    return Y_inv_final
