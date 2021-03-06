from typing import Dict, Optional, Tuple

import torch
from e3nn import o3
import math


class QuadraticNonLinearity(torch.nn.Module):
    def __init__(self, l_in: int, l_out: int, symmetric: bool = True) -> None:
        """Special non-linearity function for data in spectral domain.
        Code adapated from: https://gitlab.inria.fr/ssedlar/fourier_s2cnn/-/blob/master/fourier_s2cnn/estimators/fourier_s2cnn_utils.py#L180

        Args:
            l_in: input Spherical Harmonic degree.
            l_out: output Spherical Harmonic degree.
            symmetric: only use Spherical Harmonics of even degree. Defaults to True.
        """
        super(QuadraticNonLinearity, self).__init__()

        self.register_buffer("_l_in", torch.tensor(l_in))
        self.register_buffer("_l_out", torch.tensor(l_out))
        self.register_buffer("_symmetric", torch.tensor(2 if symmetric else 1))

        self.wigner_j: Dict[int, Dict] = dict()
        for l in range(0, int(self._l_out) + 1, int(self._symmetric)):
            self.wigner_j[l] = dict()
            for l1 in range(0, int(self._l_in) + 1, int(self._symmetric)):
                self.wigner_j[l][l1] = dict()
                for l2 in range(0, int(self._l_in) + 1, int(self._symmetric)):
                    if l2 < l1:
                        continue
                    if abs(l2 - l1) > l or l > (l1 + l2):
                        continue

                    self.wigner_j[l][l1][l2] = o3.wigner_3j(l1, l2, l).T

    def forward(self, x: Dict[int, torch.Tensor]) -> Dict[int, torch.Tensor]:
        rh_n: Dict[int, torch.Tensor] = dict()

        for l in range(0, int(self._l_out + 1), int(self._symmetric)):
            for l1 in range(0, int(self._l_in + 1), int(self._symmetric)):
                for l2 in range(0, int(self._l_in + 1), int(self._symmetric)):
                    if l1 not in x or l2 not in x:
                        continue
                    if l2 < l1:
                        continue
                    if abs(l2 - l1) > l or l > (l1 + l2):
                        continue

                    cg_ = self.wigner_j[l][l1][l2].to(device=x[0].device)
                    cg_r = torch.reshape(cg_, (2 * l + 1, 2 * l1 + 1, 2 * l2 + 1))
                    cg_l = torch.transpose(cg_r, 1, 2)

                    if l2 != l1:
                        cg_r = math.sqrt(2) * cg_r
                        cg_l = math.sqrt(2) * cg_l

                    n, a, b, c, _, _ = x[l1].shape

                    x_r = torch.einsum("nabcij,klj->nabckli", x[l2], cg_r)
                    x_r = torch.reshape(x_r, [n, a, b, c, 2 * l + 1, -1])

                    x_l = torch.einsum("nabcji,klj->nabckil", x[l1], cg_l)
                    x_l = torch.reshape(x_l, [n, a, b, c, 2 * l + 1, -1])

                    x_a = torch.einsum("nabcki,nabcji->nabckj", x_l, x_r)

                    if l not in rh_n:
                        rh_n[l] = x_a
                    else:
                        rh_n[l] += x_a

        return rh_n


class S2Convolution(torch.nn.Module):
    def __init__(self, ti_n: int, te_n: int, l_in: int, b_in: int, b_out: int, symmetric: bool = True):
        """Convolution between spherical signals and kernels in spectral domain.

        Args:
            ti_n: number of TI values
            te_n: number of TE values
            l_in: Spherical Harmonics degree
            b_in: number of input b-values
            b_out: number of output b-values
            symmetric: only use Spherical Harmonics of even degree. Defaults to True.
        """

        super(S2Convolution, self).__init__()

        self.register_buffer("_l_in", torch.tensor(l_in))
        self.register_buffer("_symmetric", torch.tensor(2 if symmetric else 1))

        self.weights = dict()
        for l in range(0, int(self._l_in + 1), int(self._symmetric)):
            n_sh_l = 2 * l + 1
            self.weights[l] = torch.nn.Parameter(torch.rand(ti_n, te_n, b_in, b_out, n_sh_l) * 0.1)
            # Manually register parameters
            self.register_parameter(f"weights_{l}", self.weights[l])

        self.bias = torch.nn.Parameter(torch.zeros(1, ti_n, te_n, b_out, 1, 1))

    def forward(self, x: Dict[int, torch.Tensor]) -> Dict[int, torch.Tensor]:
        rh: Dict[int, torch.Tensor] = dict()
        for l in range(0, int(self._l_in) + 1, int(self._symmetric)):
            rh[l] = torch.einsum("nabil, abiok->nabolk", x[l], self.weights[l])
            rh[l] += self.bias if l == 0 else torch.tensor(0)

        return rh


class SO3Convolution(torch.nn.Module):
    def __init__(self, ti_n, te_n, l_in, b_in, b_out, symmetric: bool = True):
        """Convolution between SO(3) signals and kernels in spectral domain.

        Args:
            ti_n: number of TI values
            te_n: number of TE values
            l_in: Spherical Harmonics degree
            b_in: number of input b-values
            b_out: number of output b-values
            symmetric: only use Spherical Harmonics of even degree. Defaults to True.
        """
        super(SO3Convolution, self).__init__()

        self.register_buffer("_l_in", torch.tensor(l_in))
        self.register_buffer("_symmetric", torch.tensor(2 if symmetric else 1))

        self.weights = dict()
        for l in range(0, int(self._l_in + 1), int(self._symmetric)):
            n_sh_l = 2 * l + 1
            self.weights[l] = torch.nn.Parameter(torch.rand(ti_n, te_n, b_in, b_out, n_sh_l, n_sh_l) * 0.1)
            # Manually register parameters
            self.register_parameter(f"weights_{l}", self.weights[l])

        self.bias = torch.nn.Parameter(torch.zeros(1, ti_n, te_n, b_out, 1, 1))

    def forward(self, x: Dict[int, torch.Tensor]) -> Dict[int, torch.Tensor]:
        rh: Dict[int, torch.Tensor] = dict()
        for l in range(0, int(self._l_in) + 1, int(self._symmetric)):
            rh[l] = (2 * l + 1) * torch.einsum("nabilk, abiokj->nabolj", x[l], self.weights[l])
            rh[l] += self.bias if l == 0 else torch.tensor(0)

        return rh
