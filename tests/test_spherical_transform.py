from typing import Dict

import numpy as np
import pytest
import torch
from autoencoder.logger import get_logger
from autoencoder.spherical.transform import S2ToSignal, SignalToS2
from dipy.core.sphere import hemi_icosahedron

logger = get_logger()


@pytest.fixture
def mock_data():
    b_vals = 4
    sphere = hemi_icosahedron.subdivide(2)
    sphere_xyz = sphere.vertices
    gradients = np.empty((1, 1, b_vals, sphere_xyz.shape[0], sphere_xyz.shape[1]))
    for i in range(b_vals):
        gradients[0, 0, i] = sphere_xyz

    data: Dict[int, torch.Tensor] = dict()
    for sh_degree in [0, 2, 4, 6, 8, 10]:
        num_sh_coefficients = sum([2 * l + 1 for l in range(0, sh_degree + 1, 2)])
        data[sh_degree] = torch.rand((10, 1, 1, b_vals, num_sh_coefficients))

    return gradients, data


def test_unknow_sh_inv(mock_data):
    gradients, _ = mock_data
    with pytest.raises(ValueError):
        SignalToS2(gradients, 0, "unknown")


def test_lms_sh_inv(mock_data):
    gradients, data = mock_data

    for sh_degree in [0, 2, 4, 6, 8, 10]:
        model = torch.nn.Sequential(
            S2ToSignal(gradients, sh_degree),
            SignalToS2(gradients, sh_degree, "lms"),
        )

        loss = torch.nn.functional.mse_loss(model(data[sh_degree]), data[sh_degree])
        logger.info(f"sh_degree: {sh_degree:2}, MSE: {loss.item()}")


def test_lms_tikhonov_sh_inv(mock_data):
    gradients, data = mock_data

    for sh_degree in [0, 2, 4, 6, 8, 10]:
        model = torch.nn.Sequential(
            S2ToSignal(gradients, sh_degree),
            SignalToS2(gradients, sh_degree, "lms_tikhonov"),
        )

        loss = torch.nn.functional.mse_loss(model(data[sh_degree]), data[sh_degree])
        logger.info(f"sh_degree: {sh_degree:2}, MSE: {loss.item()}")


def test_lms_laplace_beltrami_sh_inv(mock_data):
    gradients, data = mock_data

    for sh_degree in [0, 2, 4, 6, 8, 10]:
        model = torch.nn.Sequential(
            S2ToSignal(gradients, sh_degree),
            SignalToS2(gradients, sh_degree, "lms_laplace_beltrami"),
        )

        loss = torch.nn.functional.mse_loss(model(data[sh_degree]), data[sh_degree])
        logger.info(f"sh_degree: {sh_degree:2}, MSE: {loss.item()}")


def test_gram_schmidt(mock_data):
    gradients, data = mock_data

    for sh_degree in [0, 2, 4, 6, 8, 10]:
        model = torch.nn.Sequential(
            S2ToSignal(gradients, sh_degree),
            SignalToS2(gradients, sh_degree, "gram_schmidt"),
        )

        loss = torch.nn.functional.mse_loss(model(data[sh_degree]), data[sh_degree])
        logger.info(f"sh_degree: {sh_degree:2}, MSE: {loss.item()}")
