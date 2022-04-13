import numpy as np
import pytest
import torch
from autoencoder.logger import get_logger
from autoencoder.spherical.transform import SO3ToSignal, S2ToSignal, SignalToS2
from dipy.core.sphere import hemi_icosahedron

logger = get_logger()


@pytest.fixture
def mock_data():
    b_vals = 4
    sphere = hemi_icosahedron.subdivide(2)
    sphere_xyz = sphere.vertices
    gradients = np.empty((b_vals, sphere_xyz.shape[0], sphere_xyz.shape[1]))
    for i in range(b_vals):
        gradients[i] = sphere_xyz
    data = torch.rand((10, sphere_xyz.shape[0], b_vals))

    return gradients, data


def test_unknow_sh_inv(mock_data):
    gradients, _ = mock_data
    with pytest.raises(ValueError):
        SignalToS2(gradients, 0, "unknown")


def test_lms_sh_inv(mock_data):
    gradients, data = mock_data

    for sh_degree in [0, 2, 4, 8, 10]:
        signal_to_s2 = SignalToS2(gradients, sh_degree, "lms")
        s2_to_signal = S2ToSignal(gradients, sh_degree)

        loss = torch.nn.functional.mse_loss(s2_to_signal(signal_to_s2(data)), data)
        logger.info(f"sh_degree: {sh_degree:2}, MSE: {loss.item()}")


def test_lms_tikhonov_sh_inv(mock_data):
    gradients, data = mock_data

    for sh_degree in [0, 2, 4, 8, 10]:
        signal_to_s2 = SignalToS2(gradients, sh_degree, "lms_tikhonov")
        s2_to_signal = S2ToSignal(gradients, sh_degree)

        loss = torch.nn.functional.mse_loss(s2_to_signal(signal_to_s2(data)), data)
        logger.info(f"sh_degree: {sh_degree:2}, MSE: {loss.item()}")


def test_lms_laplace_beltrami_sh_inv(mock_data):
    gradients, data = mock_data

    for sh_degree in [0, 2, 4, 8, 10]:
        signal_to_s2 = SignalToS2(gradients, sh_degree, "lms_laplace_beltrami")
        s2_to_signal = S2ToSignal(gradients, sh_degree)

        loss = torch.nn.functional.mse_loss(s2_to_signal(signal_to_s2(data)), data)
        logger.info(f"sh_degree: {sh_degree:2}, MSE: {loss.item()}")


def test_gram_schmidt(mock_data):
    gradients, data = mock_data

    for sh_degree in [0, 2, 4, 8, 10]:
        signal_to_s2 = SignalToS2(gradients, sh_degree, "gram_schmidt")
        s2_to_signal = S2ToSignal(gradients, sh_degree)

        loss = torch.nn.functional.mse_loss(s2_to_signal(signal_to_s2(data)), data)
        logger.info(f"sh_degree: {sh_degree:2}, MSE: {loss.item()}")
