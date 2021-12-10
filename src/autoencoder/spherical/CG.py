import numpy as np
import torch
from sympy.physics.quantum.cg import CG


def convert_real_to_complex(L):
    """Convert real to complex spherical harmonics.

    Arguments:
        L (int): maximum SH degree
    Returns:
        numpy array: mapping between real and complex SH basis
    """
    n_sph = 2 * L + 1

    M = np.zeros((n_sph, n_sph), dtype=np.complex64)

    M[L, L] = 1
    for m in range(-L, 0):
        M[m + L, m + L] = -1 / np.sqrt(2) * 1j
        M[m + L, -m + L] = 1 / np.sqrt(2)
        M[-m + L, m + L] = np.power(-1, np.abs(m)) / np.sqrt(2) * 1j
        M[-m + L, -m + L] = np.power(-1, np.abs(m)) / np.sqrt(2)

    return M


def clebsch_gordan_matrix_R(L1, L2, L):
    """Create Clebsch-Gordan matrix associated to real WignerD matrices.

    Arguments:
        L1 (int): RH degree of the left RH coefficients
        L2 (int): RH degree of the right RH coefficients
        L (int): RH degree of the output RH coefficients
    """
    CG_M = np.zeros(((2 * L1 + 1) * (2 * L2 + 1), 2 * L + 1))
    for m in range(-L1, L1 + 1):
        for m_p in range(-L2, L2 + 1):
            M_p = m + m_p
            cg = CG(L1, m, L2, m_p, L, M_p).doit()
            if cg:
                CG_M[(m + L1) * (2 * L2 + 1) + m_p + L2, m + m_p + L] = cg

    M1 = convert_real_to_complex(L1)
    M2 = convert_real_to_complex(L2)
    M = convert_real_to_complex(L)

    if (L1 + L2 + L) % 2 == 0:
        return np.real(np.dot(M.conj().T, np.dot(CG_M.T, np.kron(M1, M2))))
    else:
        return np.imag(np.dot(M.conj().T, np.dot(CG_M.T, np.kron(M1, M2))))


def real_clebsch_gordan_all(L, L_max, *, device="cpu"):
    """Create a dictionary with Clebsch-Gordan matrices.

    Arguments:
        L (int) : maximal RH degree of input
        L_max (int): maximal RH degree of output
    Returns:
        dictionary of Clebsch-Gordan matrices
    """
    if L_max > 2 * L:
        print("L_max should not be higher than 2*L")
        raise

    CG_all_r = dict()
    CG_all_l = dict()
    for l in range(0, L_max + 1):
        CG_all_r[l] = dict()
        CG_all_l[l] = dict()
        for l1 in range(0, L + 1):
            for l2 in range(0, L + 1):
                if l2 < l1:
                    continue
                if np.abs(l2 - l1) <= l <= (l1 + l2):
                    cg_ = torch.from_numpy(clebsch_gordan_matrix_R(l1, l2, l)).float()
                    cg_ = cg_.to(device)

                    cg_r = torch.reshape(cg_, (2 * l + 1, 2 * l1 + 1, 2 * l2 + 1))
                    cg_l = torch.transpose(cg_r, 1, 2)

                    if l2 != l1:
                        sqrt_2 = np.sqrt(2)
                        CG_all_r[l][l1, l2] = sqrt_2 * cg_r
                        CG_all_l[l][l1, l2] = sqrt_2 * cg_l
                    else:
                        CG_all_r[l][l1, l2] = cg_r
                        CG_all_l[l][l1, l2] = cg_l
    return CG_all_r, CG_all_l
