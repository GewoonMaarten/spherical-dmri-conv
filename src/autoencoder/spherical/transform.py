from typing import Callable, Dict, Literal

import numpy as np
import torch
from dipy.reconst.shm import (
    cart2sphere,
    real_sh_descoteaux_from_index,
    sph_harm_ind_list,
)


class Signal_to_S2(torch.nn.Module):
    def __init__(
        self,
        gradients: torch.Tensor,
        sh_degree_max: int,
        inversion_function: Literal["lms", "lms_tikhonov", "lms_laplace_beltrami", "gram_schmidt"],
        **kwargs,
    ):
        """Computes the Spherical harmonic coefficients, according to:

        .. math::

            \hat{s}^m_l = \int_{S^2} s(r) \overline{Y^m_l(r)} dr

        Args:
            gradients: Vectors to fit the Spherical Harmonics to. Has to be of shape ``(a, b, 3)``, where
                ``a`` are the number of b values (shells) and ``b`` is the number of gradient directions. The vector is in
                cartesian coordinates (xyz).
            sh_degree_max: Maximum degree of Spherical Harmonics to fit.
            inversion_function: name of the inversion function to apply (see :py:attr:`Signal_to_S2.inversion_functions`
                for options).

        Raises:
            ValueError: raised when an unknown inversion function is given.

        Example:

        .. code-block:: python
            :linenos:
            :emphasize-lines: 3

            gradients = torch.rand((4, 90, 3))
            data = torch.rand((512, 90, 4)) # dwi data with 90 gradient directions and 4 b-values
            signal_to_s2 = Signal_to_S2(gradients, 4, "gram_schmidt")
            signal_to_s2(data)
        """
        super(Signal_to_S2, self).__init__()

        self.inversion_functions: Dict[str, Callable] = {
            "lms": self.lms_sh_inv,
            "lms_tikhonov": self.lms_tikhonov_sh_inv,
            "lms_laplace_beltrami": self.lms_laplace_beltrami_sh_inv,
            "gram_schmidt": self.gram_schmidt_sh_inv,
        }
        self.sh_degree_max = sh_degree_max

        if inversion_function not in self.inversion_functions.keys():
            raise ValueError(
                f"inversion_function '{inversion_function}' unknown, inversion_function has to be one of {*self.inversion_functions,}"
            )

        m, n = sph_harm_ind_list(self.sh_degree_max)
        n_shells = gradients.shape[0]

        self.Y_inv = np.zeros((n_shells, self.n_sh, gradients.shape[1]))
        for sh_idx in range(n_shells):
            x, y, z = (
                gradients[sh_idx, :, 0],
                gradients[sh_idx, :, 1],
                gradients[sh_idx, :, 2],
            )
            _, theta, phi = cart2sphere(x, y, z)
            Y_gs = real_sh_descoteaux_from_index(m, n, theta[:, None], phi[:, None])
            self.Y_inv[sh_idx, :, :] = self.inversion_functions[inversion_function](Y_gs, self.sh_degree_max, **kwargs)

        self.Y_inv = torch.from_numpy(self.Y_inv).float()
        self.Y_inv = torch.nn.Parameter(self.Y_inv, requires_grad=False)

    @property
    def n_sh(self):
        return np.sum([2 * l + 1 for l in range(0, self.sh_degree_max + 1, 2)])

    def forward(self, x: torch.Tensor):
        return torch.einsum("npc,clp->ncl", x, self.Y_inv)

    def lms_sh_inv(self, sh: np.ndarray, l_max: int, **kwargs) -> np.ndarray:
        """Inversion of spherical harmonic basis with least-mean square

        Args:
            sh: real spherical harmonics bases of even degree (each column is a basis)
            l_max: spherical harmonics degree

        Returns:
            inverted spherical harmonic bases of even degree
        """
        return np.linalg.pinv(sh)

    def lms_tikhonov_sh_inv(self, sh: np.ndarray, l_max: int, **kwargs) -> np.ndarray:
        """Inversion of spherical harmonic basis with least-mean square regularized with Tikhonov regularization term :cite:p:`hess2006q`

        Args:
            sh: real spherical harmonics bases of even degree (each column is a basis)
            l_max: spherical harmonics degree
            **lambda_ (float): regularization weight. Defaults to 0.006

        Returns:
            inverted spherical harmonic bases of even degree
        """
        lambda_ = kwargs.get("lambda_", 0.006)

        tikhonov_reg = np.eye(sh.shape[1], dtype=np.float32)
        return np.dot(np.linalg.inv(np.dot(sh.T, sh) + lambda_ * tikhonov_reg), sh.T)

    def lms_laplace_beltrami_sh_inv(self, sh: np.ndarray, l_max: int, **kwargs) -> np.ndarray:
        """Inversion of spherical harmonic basis with least-mean square regularized with Laplace-Beltrami regularization term :cite:p:`descoteaux2007regularized`

        Args:
            sh: real spherical harmonics bases of even degree (each column is a basis)
            l_max: spherical harmonics degree
            **lambda_ (float): regularization weight. Defaults to 0.006

        Returns:
            inverted spherical harmonic bases of even degree
        """
        lambda_ = kwargs.get("lambda_", 0.006)

        lb_reg = np.zeros((sh.shape[1], sh.shape[1]), dtype=np.float32)
        count = 0
        for l in range(0, l_max + 1, 2):
            for _ in range(-l, l + 1):
                lb_reg[count, count] = l**2 * (l + 1) ** 2
                count += 1
        return np.dot(np.linalg.inv(np.dot(sh.T, sh) + lambda_ * lb_reg), sh.T)

    def gram_schmidt_sh_inv(self, sh: np.ndarray, l_max: int, **kwargs) -> np.ndarray:
        """Inversion of spherical harmonic basis with Gram-Schmidt orthonormalization process

        Args:
            sh: real spherical harmonics bases of even degree (each column is a basis)
            l_max: spherical harmonics degree
            **n_iter (int): number of iterations for degree shuffling. Defaults to 1000

        Returns:
            inverted spherical harmonic bases of even degree
        """
        n_iters = kwargs.get("n_iters", 1000)

        np.random.seed(1234)

        sh_inv_final = np.zeros_like(sh.T)
        for _ in range(n_iters):
            order = []
            count_h = 0
            for i in range(0, l_max + 1, 2):
                order_h = count_h + np.arange(0, 2 * i + 1)
                np.random.shuffle(order_h)
                order.extend(list(order_h))
                count_h += 2 * i + 1

            deorder = np.argsort(order)
            sh_inv = np.zeros_like(sh.T)
            for i in range(sh_inv.shape[0]):
                sh_inv[i, :] = sh[:, order[i]]
                for j in range(0, i):
                    if np.sum(sh_inv[j, :] ** 2) > 1.0e-8:
                        sh_inv[i, :] -= (
                            np.sum(sh_inv[i, :] * sh_inv[j, :]) / (np.sum(sh_inv[j, :] ** 2) + 1.0e-8) * sh_inv[j, :]
                        )
                sh_inv[i, :] /= np.sqrt(np.sum(sh_inv[i, :] ** 2))
            sh_inv_final += sh_inv[deorder, :]
        sh_inv_final /= n_iters

        n = np.dot(sh[:, 0:1].T, sh[:, 0:1])[0, 0]
        sh_inv_final /= np.sqrt(n)

        return sh_inv_final


class S2_to_Signal(torch.nn.Module):
    def __init__(self, gradients: torch.Tensor, sh_degree_max: int):
        """Computes the DWI from the spherical coefficients, according to:

        .. math::

            s(r) = \sum^{L_{max}}_{l=0} \sum^{m=l}_{m=-l} \hat{s}^m_l Y^m_l(r)

        Args:
            gradients: Vectors to fit the Spherical Harmonics to. Has to be of shape ``(a,b,3)``, where
                ``a`` are the number of b-values (shells) and ``b`` is the number of gradient directions. The vector is in
                cartesian coordinates (xyz).
            sh_degree_max: Maximum degree of Spherical Harmonics to fit.

        Example:

        .. code-block:: python
            :linenos:
            :emphasize-lines: 3

            gradients = torch.rand((4, 90, 3))
            data = torch.rand((512, 4, 15)) # Spherical coefficients with L of degree 4 and 4 b-values
            s2_to_signal = S2_to_Signal(gradients, 4)
            s2_to_signal(data)
        """
        super(S2_to_Signal, self).__init__()

        self.sh_degree_max = sh_degree_max

        m, n = sph_harm_ind_list(self.sh_degree_max)
        n_shells = gradients.shape[0]

        self.Y_gs = np.zeros((n_shells, self.n_sh, gradients.shape[1]))
        for sh_idx in range(n_shells):
            x, y, z = (
                gradients[sh_idx, :, 0],
                gradients[sh_idx, :, 1],
                gradients[sh_idx, :, 2],
            )
            _, theta, phi = cart2sphere(x, y, z)
            self.Y_gs[sh_idx, :, :] = real_sh_descoteaux_from_index(m, n, theta[:, None], phi[:, None]).T

        self.Y_gs = torch.from_numpy(self.Y_gs).float()
        self.Y_gs = torch.nn.Parameter(self.Y_gs, requires_grad=False)

    @property
    def n_sh(self):
        return np.sum([2 * l + 1 for l in range(0, self.sh_degree_max + 1, 2)])

    def forward(self, x: torch.Tensor):
        return torch.einsum("ncl,clp->npc", x, self.Y_gs)
