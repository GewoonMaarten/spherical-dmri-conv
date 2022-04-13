from typing import Callable, Dict, Literal

import numpy as np
import torch
from dipy.reconst.shm import (
    cart2sphere,
    real_sh_descoteaux_from_index,
    sph_harm_ind_list,
)


class SignalToS2(torch.nn.Module):
    def __init__(
        self,
        gradients: torch.Tensor,
        sh_degree_max: int,
        inversion_function: Literal["lms", "lms_tikhonov", "lms_laplace_beltrami", "gram_schmidt"],
        **kwargs,
    ):
        """Computes the spherical harmonic coefficients.
        According to:

        .. math::

            \hat{s}^m_l = \int_{S^2} s(r) \overline{Y^m_l(r)} dr

        where :math:`\hat{s}^m_l` are the spherical coefficients, :math:`r \in \mathbb{R}^3`,
        :math:`s : S^2 \mapsto \mathbb{C}` and :math:`\overline{Y^m_l(r)}` denotes the `spherical harmonics`_, the
        overbar denotes the `complex conjugation`_.

        See "Spherical CNNs" by T. Cohen `et al.` for more information :cite:p:`cohen2018spherical`.

        Example usage:

        .. code-block:: python
            :linenos:

            # Create random dwi data with 90 gradient directions and 4 b-values
            gradients = torch.rand((4, 90, 3)) # {b-values, gradients, xyz}
            data = torch.rand((512, 90, 4)) # {batch size, gradients, b-values}

            signal_to_s2 = SignalToS2(gradients, 4, "gram_schmidt")
            signal_to_s2(data)

        Args:
            gradients: Vectors to fit the Spherical Harmonics to. Has to be of shape ``(a, b, 3)``, where
                ``a`` are the number of b-values (shells) and ``b`` is the number of gradient directions. The vector is
                in cartesian coordinates (xyz).
            sh_degree_max: Maximum degree (``l``) of Spherical Harmonics to fit. Denoted by :math:`L_{max}` in the
                equation.
            inversion_function: name of the inversion function to apply. Available functions are:

                - ``"lms"`` = :func:`lms_sh_inv`
                - ``"lms_tikhonov"`` = :func:`lms_tikhonov_sh_inv`
                - ``"lms_laplace_beltrami"`` = :func:`lms_laplace_beltrami_sh_inv`
                - ``"gram_schmidt"`` = :func:`gram_schmidt_sh_inv`

        Raises:
            ValueError: raised when an unknown inversion function is given.

        .. _spherical harmonics: https://en.wikipedia.org/wiki/Spherical_harmonics
        .. _complex conjugation: https://en.wikipedia.org/wiki/Complex_conjugate
        """
        super(SignalToS2, self).__init__()

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
        return sum([2 * l + 1 for l in range(0, self.sh_degree_max + 1, 2)])

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
        """Inversion of spherical harmonic basis with Gram-Schmidt orthonormalization process :cite:p:`yeo2005computing`

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


class S2ToSignal(torch.nn.Module):
    def __init__(self, gradients: torch.Tensor, sh_degree_max: int):
        """Computes the DWI from the spherical harmonic coefficients.
        According to:

        .. math::

            s(r) = \sum^{L_{max}}_{l=0} \sum^{m=l}_{m=-l} \hat{s}^m_l Y^m_l(r)

        where :math:`s(r)` is the DWI signal, :math:`r \in \mathbb{R}^3`,
        :math:`Y^m_l(r)` denotes the `spherical harmonics`_ and :math:`\hat{s}^m_l` are the spherical coefficients.

        See "Spherical CNNs" by T. Cohen `et al.` for more information :cite:p:`cohen2018spherical`.

        Example usage:

        .. code-block:: python
            :linenos:

            # Create random sh coefficients with 90 gradient directions and 4 b-values
            gradients = torch.rand((4, 90, 3)) # {b-values, gradients, xyz}
            data = torch.rand((512, 90, 4)) # {batch size, gradients, SH coefficients}

            s2_to_signal = S2ToSignal(gradients, 4)
            s2_to_signal(data)

        Args:
            gradients: Vectors to fit the Spherical Harmonics to. Has to be of shape ``(a,b,3)``, where
                ``a`` are the number of b-values (shells) and ``b`` is the number of gradient directions. The vector is
                in cartesian coordinates (xyz).
            sh_degree_max: Maximum degree (``l``) of Spherical Harmonics to fit. Denoted by :math:`L_{max}` in the
                equation.

        .. _spherical harmonics: https://en.wikipedia.org/wiki/Spherical_harmonics
        """
        super(S2ToSignal, self).__init__()

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
        return sum([2 * l + 1 for l in range(0, self.sh_degree_max + 1, 2)])

    def forward(self, x: torch.Tensor):
        return torch.einsum("ncl,clp->npc", x, self.Y_gs)


class SO3ToSignal(torch.nn.Module):
    def __init__(self, gradients: torch.Tensor, sh_degree_max: int):
        """Computes the DWI from the Wigner-D matrix coefficients.
        Every coefficient in the `Wigner-D matrix`_ where :math:`m != 0` are thrown away, leaving us with the
        spherical harmonic coefficients. More formally:

        .. math::

            Y_l^n(r) = D_l^{m=0,n}(R)

        After this, the same steps are used as in :class:`.S2ToSignal`.

        Example usage:

        .. code-block:: python
            :linenos:

            gradients = torch.rand((4, 90, 3)) # {b-values, gradients, xyz}
            data = {
                0: torch.rand((512, 90, 1, 1)), # {batch size, gradients, Wigner-D coefficients, Wigner-D coefficients}
                2: torch.rand((512, 90, 5, 5)),
                4: torch.rand((512, 90, 9, 9))
            } # data is expected to be a dict where each key is a sh degree.

            so3_to_signal = SO3ToSignal(gradients, 4)
            so3_to_signal(data)

        Args:
            gradients: Vectors to fit the Spherical Harmonics to. Has to be of shape ``(a,b,3)``, where
                ``a`` are the number of b-values (shells) and ``b`` is the number of gradient directions. The vector is
                in cartesian coordinates (xyz).
            sh_degree_max: Maximum degree of Spherical Harmonics to fit.

        .. _Wigner-D matrix: https://en.wikipedia.org/wiki/Wigner_D-matrix
        """
        super(SO3ToSignal, self).__init__()
        self.s2_to_signal = S2ToSignal(gradients, sh_degree_max)

    def forward(self, x: Dict[int, torch.Tensor]):
        x = [x[l][:, :, :, :, :, (2 * l + 1) // 2] for l in x]
        x = torch.cat(x, 4).squeeze()
        return self.s2_to_signal(x)
