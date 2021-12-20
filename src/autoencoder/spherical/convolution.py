from typing import Optional

import numpy as np
import torch
from pytorch_lightning.profiler import PassThroughProfiler


class QuadraticNonLinearity(torch.nn.Module):
    def __init__(self, l_in, l_out, cg_r, cg_l, symmetric: bool = True) -> None:
        super(QuadraticNonLinearity, self).__init__()

        self._l_in = l_in
        self._l_out = l_out
        self._cg_r = cg_r
        self._cg_l = cg_l
        self._symmetric = 2 if symmetric else 1

    def forward(
        self, x: tuple[dict[int, torch.Tensor], Optional[torch.Tensor]]
    ) -> tuple[dict[int, torch.Tensor], torch.Tensor]:
        rh_n = dict()
        rh, feats = x

        for l in range(0, self._l_out + 1, self._symmetric):
            for l1 in range(0, self._l_in + 1, self._symmetric):
                for l2 in range(0, self._l_in + 1, self._symmetric):
                    if l1 not in rh or l2 not in rh:
                        continue
                    if l2 < l1:
                        continue
                    if np.abs(l2 - l1) <= l <= (l1 + l2):

                        cg_r = self._cg_r[l][l1, l2]
                        cg_l = self._cg_l[l][l1, l2]

                        n, a, b, c, _, _ = rh[l1].shape

                        x = torch.einsum("nabcij,klj->nabckli", rh[l2], cg_r)
                        x = torch.reshape(x, [n, a, b, c, 2 * l + 1, -1])

                        y = torch.einsum("nabcji,klj->nabckil", rh[l1], cg_l)
                        y = torch.reshape(y, [n, a, b, c, 2 * l + 1, -1])

                        z = torch.einsum("nabcki,nabcji->nabckj", y, x)

                        if l not in rh_n:
                            rh_n[l] = z
                        else:
                            rh_n[l] += z

        return rh_n, self._extract_features(rh_n, feats)

    def _extract_features(
        self, rh_n: dict[int, torch.Tensor], feats: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Extract rotation invariant features.

        Args:
            rh_n (dict[int, torch.Tensor]): result from the quadratic non-linearity.
        """
        for l in range(0, self._l_out + 1, self._symmetric):
            n_l = 8 * (np.pi ** 2) / (2 * l + 1)

            feats_l = torch.flatten(
                torch.sum(torch.pow(rh_n[l], 2), (5, 4)), start_dim=1
            )

            if feats is None:
                feats = n_l * feats_l
            else:
                feats = torch.cat((feats, n_l * feats_l), dim=1)

        return feats


class S2Convolution(torch.nn.Module):
    def __init__(
        self, ti_n, te_n, l_in, b_in, b_out, symmetric: bool = True, profiler=None
    ):
        """Convolution between spherical signals and kernels in spectral domain."""
        super(S2Convolution, self).__init__()

        self._l_in = l_in
        self._symmetric = 2 if symmetric else 1
        self._profiler = profiler or PassThroughProfiler()

        self.weights = dict()
        for l in range(0, self._l_in + 1, self._symmetric):
            n_sh_l = 2 * l + 1
            self.weights[l] = torch.nn.Parameter(
                torch.rand(ti_n, te_n, b_in, b_out, n_sh_l) * 0.1
            )
            # Manually register parameters
            self.register_parameter(f"weights_{l}", self.weights[l])

        self.bias = torch.nn.Parameter(torch.zeros(1, ti_n, te_n, b_out, 1, 1))

    def forward(
        self, x: dict[int, torch.Tensor]
    ) -> tuple[dict[int, torch.Tensor], Optional[torch.Tensor]]:
        with self._profiler.profile("S2Convolution"):
            # convolution
            rh = dict()
            for l in range(0, self._l_in + 1, self._symmetric):
                rh[l] = torch.einsum("nabil, abiok->nabolk", x[l], self.weights[l])
                rh[l] += self.bias if l == 0 else 0

        return rh, None


class SO3Convolution(torch.nn.Module):
    def __init__(
        self, ti_n, te_n, l_in, b_in, b_out, symmetric: bool = True, profiler=None
    ):
        """Convolution between SO(3) signals and kernels in spectral domain."""
        super(SO3Convolution, self).__init__()

        self._l_in = l_in
        self._symmetric = 2 if symmetric else 1
        self._profiler = profiler or PassThroughProfiler()

        self.weights = dict()
        for l in range(0, self._l_in + 1, self._symmetric):
            n_sh_l = 2 * l + 1
            self.weights[l] = torch.nn.Parameter(
                torch.rand(ti_n, te_n, b_in, b_out, n_sh_l, n_sh_l) * 0.1
            )
            # Manually register parameters
            self.register_parameter(f"weights_{l}", self.weights[l])

        self.bias = torch.nn.Parameter(torch.zeros(1, ti_n, te_n, b_out, 1, 1))

    def forward(
        self, x: tuple[dict[int, torch.Tensor], Optional[torch.Tensor]]
    ) -> tuple[dict[int, torch.Tensor], Optional[torch.Tensor]]:
        data, feats = x
        with self._profiler.profile("SO3Convolution"):
            # convolution
            rh = dict()
            for l in range(0, self._l_in + 1, self._symmetric):
                rh[l] = (2 * l + 1) * torch.einsum(
                    "nabilk, abiokj->nabolj", data[l], self.weights[l]
                )
                rh[l] += self.bias if l == 0 else 0

        return rh, feats
