import torch
import numpy as np
from pytorch_lightning.profiler import PassThroughProfiler


def quadratic_non_linearity(rh, L_in, L_out, CG_r, CG_l, symmetric: int = 2):
    rh_n = dict()
    for l in range(0, L_out + 1, symmetric):
        for l1 in range(0, L_in + 1, symmetric):
            for l2 in range(0, L_in + 1, symmetric):
                if l1 not in rh or l2 not in rh:
                    continue
                if l2 < l1:
                    continue
                if np.abs(l2 - l1) <= l <= (l1 + l2):

                    cg_r = CG_r[l][l1, l2]
                    cg_l = CG_l[l][l1, l2]

                    a, b, n, c, _, _ = rh[l1].shape

                    x = torch.einsum("abncij,klj->abnckli", rh[l2], cg_r)
                    x = torch.reshape(x, [a, b, n, c, 2 * l + 1, -1])

                    y = torch.einsum("abncji,klj->abnckil", rh[l1], cg_l)
                    y = torch.reshape(y, [a, b, n, c, 2 * l + 1, -1])

                    z = torch.einsum("abncki,abncji->abnckj", y, x)

                    if l not in rh_n:
                        rh_n[l] = z
                    else:
                        rh_n[l] += z
    return rh_n


class S2Convolution(torch.nn.Module):
    def __init__(
        self,
        ti_n,
        te_n,
        l_in,
        l_out,
        b_in,
        b_out,
        cg_r,
        cg_l,
        *,
        symmetric=2,
        profiler=None,
    ):
        """Convolution between spherical signals and kernels in spectral domain."""
        super(S2Convolution, self).__init__()

        self.l_in = l_in
        self.l_out = l_out
        self.symmetric = symmetric
        self.cg_r = cg_r
        self.cg_l = cg_l

        self.profiler = profiler or PassThroughProfiler()

        self.weights = dict()
        for l in range(0, self.l_in + 1, self.symmetric):
            n_sh_l = 2 * l + 1
            self.weights[l] = torch.nn.Parameter(
                torch.rand(ti_n, te_n, b_in, b_out, n_sh_l) * 0.1
            )
            # Manually register parameters
            self.register_parameter(f"weights_{l}", self.weights[l])

        self.bias = torch.nn.Parameter(torch.zeros(ti_n, te_n, 1, b_out, 1, 1))

    def forward(self, x):
        with self.profiler.profile("S2Convolution"):
            # convolution
            rh = dict()
            for l in range(0, self.l_in + 1, self.symmetric):
                rh[l] = torch.einsum("nabil, abiok->abnolk", x[l], self.weights[l])
                rh[l] += self.bias if l == 0 else 0

            # activation function
            rh_n = quadratic_non_linearity(
                rh,
                self.l_out,
                self.l_in,
                self.cg_r,
                self.cg_l,
                symmetric=self.symmetric,
            )

            # feature extraction
            for l in range(0, self.l_out + 1, self.symmetric):
                n_l = 8 * (np.pi ** 2) / (2 * l + 1)

                rh_n_l_t = torch.transpose(rh_n[l], 2, 0)
                rh_n_l_p = torch.pow(rh_n_l_t, 2)
                rh_n_l_s = torch.sum(rh_n_l_p, (5, 4))
                feats_l = torch.flatten(rh_n_l_s, start_dim=1)

                if l == 0:
                    feats = n_l * feats_l
                else:
                    feats = torch.cat((feats, n_l * feats_l), dim=1)

        return rh_n, feats


class SO3Convolution(torch.nn.Module):
    def __init__(
        self,
        ti_n,
        te_n,
        l_in,
        l_out,
        b_in,
        b_out,
        cg_r,
        cg_l,
        *,
        symmetric=2,
        profiler=None,
    ):
        """Convolution between SO(3) signals and kernels in spectral domain."""
        super(SO3Convolution, self).__init__()

        self.l_in = l_in
        self.l_out = l_out
        self.symmetric = symmetric
        self.cg_r = cg_r
        self.cg_l = cg_l

        self.profiler = profiler or PassThroughProfiler()

        self.weights = dict()
        for l in range(0, self.l_in + 1, self.symmetric):
            n_sh_l = 2 * l + 1
            self.weights[l] = torch.nn.Parameter(
                torch.rand(ti_n, te_n, b_in, b_out, n_sh_l, n_sh_l) * 0.1
            )
            # Manually register parameters
            self.register_parameter(f"weights_{l}", self.weights[l])

        self.bias = torch.nn.Parameter(torch.zeros(ti_n, te_n, 1, b_out, 1, 1))

    def forward(self, x):
        with self.profiler.profile("SO3Convolution"):
            # convolution
            rh = dict()
            for l in range(0, self.l_in + 1, self.symmetric):
                rh[l] = (2 * l + 1) * torch.einsum(
                    "abnilk, abiokj->abnolj", x[l], self.weights[l]
                )
                rh[l] += self.bias if l == 0 else 0

            # activation function
            rh_n = quadratic_non_linearity(
                rh,
                self.l_out,
                self.l_in,
                self.cg_r,
                self.cg_l,
                symmetric=self.symmetric,
            )

            # feature extraction
            for l in range(0, self.l_out + 1, self.symmetric):
                n_l = 8 * (np.pi ** 2) / (2 * l + 1)

                rh_n_l_t = torch.transpose(rh_n[l], 2, 0)
                rh_n_l_p = torch.pow(rh_n_l_t, 2)
                rh_n_l_s = torch.sum(rh_n_l_p, (5, 4))
                feats_l = torch.flatten(rh_n_l_s, start_dim=1)

                if l == 0:
                    feats = n_l * feats_l
                else:
                    feats = torch.cat((feats, n_l * feats_l), dim=1)

        return rh_n, feats
