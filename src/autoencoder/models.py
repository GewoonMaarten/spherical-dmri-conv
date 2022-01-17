from collections import OrderedDict

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.profiler import PassThroughProfiler
from pytorch_lightning.utilities.cli import MODEL_REGISTRY
from torch import nn

from autoencoder.logger import logger
from autoencoder.spherical.convolution import (
    QuadraticNonLinearity,
    S2Convolution,
    SO3Convolution,
)


class Encoder(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        max_temp: float = 10.0,
        min_temp: float = 0.1,
        reg_threshold: float = 3.0,
        reg_eps: float = 1e-10,
        profiler=None,
    ) -> None:
        """Feature selection encoder. Implemented according to [_Concrete Autoencoders for Differentiable Feature Selection and Reconstruction_](https://arxiv.org/abs/1901.09346).

        Args:
            input_size (int): size of the input layer. Should be the same as the `output_size` of the decoder.
            output_size (int): size of the latent layer. Should be the same as the `input_size` of the decoder.
            max_temp (float, optional): maximum temperature for Gumble Softmax. Defaults to 10.0.
            min_temp (float, optional): minimum temperature for Gumble Softmax. Defaults to 0.1.
            reg_threshold (float, optional): regularization threshold. The encoder will be penalized when the sum of
            probabilities for a selection neuron exceed this threshold. Defaults to 0.3.
            reg_eps (float, optional): regularization epsilon. Minimum value for the clamped softmax function in
            regularization term. Defaults to 1e-10.
        """
        super(Encoder, self).__init__()

        self.profiler = profiler or PassThroughProfiler()

        self.register_buffer("temp", torch.tensor(max_temp))
        self.register_buffer("max_temp", torch.tensor(max_temp))
        self.register_buffer("min_temp", torch.tensor(min_temp))
        self.register_buffer("reg_threshold", torch.tensor(reg_threshold))
        self.register_buffer("reg_eps", torch.tensor(reg_eps))

        logits = nn.init.xavier_normal_(torch.empty(output_size, input_size))
        self.logits = nn.Parameter(logits, requires_grad=True)

    @property
    def latent_features(self):
        return torch.argmax(self.logits, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Uses the trained encoder to make inferences.

        Args:
            x (torch.Tensor): input data. Should be the same size as the encoder input.

        Returns:
            torch.Tensor: encoder output of size `output_size`.
        """
        with self.profiler.profile("encoder"):
            logits_size = self.logits.size()

            selections: torch.Tensor = None
            if self.training:
                uniform = torch.rand(logits_size, device=x.device)
                gumbel = -torch.log(-torch.log(uniform))
                noisy_logits = (self.logits + gumbel) / self.temp
                samples = F.softmax(noisy_logits, dim=1)

                selections = samples
            else:
                dim_argmax = len(logits_size) - 1
                logits_argmax = torch.argmax(self.logits, dim_argmax)
                discrete_logits = F.one_hot(logits_argmax, num_classes=logits_size[1])

                selections = discrete_logits

            encoded = torch.matmul(x, torch.transpose(selections.float(), 0, 1))
        return encoded

    def update_temp(self, current_epoch, max_epochs) -> torch.Tensor:
        self.temp = self.max_temp * torch.pow(
            (self.min_temp / self.max_temp), (current_epoch / max_epochs)
        )
        return self.temp

    def calc_mean_max(self) -> torch.Tensor:
        logits_softmax = F.softmax(self.logits, dim=1)
        logits_max = torch.max(logits_softmax, 1).values
        mean_max = torch.mean(logits_max)

        return mean_max

    def regularization(self) -> float:
        """Regularization term according to https://homes.esat.kuleuven.be/~abertran/reports/TS_JNE_2021.pdf. The sum of
        probabilities for a selection neuron is penalized if its larger than the threshold value. The returned value is
        summed with the loss function."""
        selection = torch.clamp(F.softmax(self.logits, dim=1), self.reg_eps, 1)
        return torch.sum(F.relu(torch.norm(selection, 1, dim=0) - self.reg_threshold))


class Decoder(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        n_hidden_layers: int,
        negative_slope: float = 0.2,
    ) -> None:
        """Standard decoder. It generates a network from `input_size` to `output_size`. The layers are generates as
        follows:
        ```python
        import numpy as np
        step_size = abs(output_size - input_size) // n_hidden_layers
        layer_sizes = np.arange(input_size, output_size, step_size)
        ```

        Args:
            input_size (int): size of the latent layer. Should be the same as the `output_size` of the encoder.
            output_size (int): size of the output layer. Should be the same as `input_size` of the encoder.
            n_hidden_layers (int): number of hidden layers. If 0 then the input will be directly connected to the
            output.
            negative_slope (float, optional): negative slope for the Leaky ReLu activation layer. Defaults to 0.2.
        """
        super(Decoder, self).__init__()

        indices = np.arange(2 + n_hidden_layers)
        data_indices = np.array([indices[0], indices[-1]])
        data = np.array([input_size, output_size])

        layer_sizes = np.interp(indices, data_indices, data).astype(int)
        n_layers = len(layer_sizes)

        # Construct the network
        layers = OrderedDict()
        for i in range(1, n_layers):
            n = i - 1
            if i == n_layers - 1:  # Last layer
                layers[f"linear_{n}"] = nn.Linear(layer_sizes[i - 1], layer_sizes[i])
            else:
                layers[f"linear_{n}"] = nn.Linear(layer_sizes[i - 1], layer_sizes[i])
                layers[f"relu_{n}"] = nn.LeakyReLU(negative_slope)

        logger.debug("decoder layers: %s", layers)

        self.decoder = nn.Sequential(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Uses the trained decoder to make inferences.

        Args:
            x (torch.Tensor): input data. Should be the same size as the decoder input.

        Returns:
            torch.Tensor: decoder output of size `output_size`.
        """
        decoded = self.decoder(x)
        return decoded


@MODEL_REGISTRY
class ConcreteAutoencoder(pl.LightningModule):
    def __init__(
        self,
        input_output_size: int = 1344,
        latent_size: int = 500,
        decoder_hidden_layers: int = 2,
        learning_rate: float = 1e-3,
        max_temp: float = 10.0,
        min_temp: float = 0.1,
        reg_lambda: float = 0.0,
        reg_threshold: float = 1.0,
    ) -> None:
        """Trains a concrete autoencoder. Implemented according to [_Concrete Autoencoders for Differentiable Feature Selection and Reconstruction_](https://arxiv.org/abs/1901.09346).

        Args:
            input_output_size (int): size of the input and output layer.
            latent_size (int): size of the latent layer.
            decoder_hidden_layers (int, optional): number of hidden layers for the decoder. Defaults to 2.
            learning_rate (float, optional): learning rate for the optimizer. Defaults to 1e-3.
            max_temp (float, optional): maximum temperature for Gumble Softmax. Defaults to 10.0.
            min_temp (float, optional): minimum temperature for Gumble Softmax. Defaults to 0.1.
            reg_lambda(float, optional): how much weight to apply to the regularization term. If the value is 0.0 then no regularization will be applied. Defaults to 0.0.
            reg_threshold (float, optional): regularization threshold. The encoder will be penalized when the sum of probabilities for a selection neuron exceed this threshold. Defaults to 1.0.
        """
        super(ConcreteAutoencoder, self).__init__()
        self.save_hyperparameters()

        self.encoder = Encoder(
            input_output_size,
            latent_size,
            max_temp,
            min_temp,
            reg_threshold=reg_threshold,
        )
        self.decoder = Decoder(latent_size, input_output_size, decoder_hidden_layers)

        self.learning_rate = learning_rate
        self.reg_lambda = reg_lambda

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Uses the trained autoencoder to make inferences.

        Args:
            x (torch.Tensor): input data. Should be the same size as encoder input.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: (encoder output, decoder output)
        """
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

    def configure_optimizers(self) -> torch.optim.Adam:
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        loss = self._shared_eval(batch, batch_idx, "train")

        if self.reg_lambda > 0:
            reg_term = self.encoder.regularization()
            loss = loss + (self.reg_lambda * reg_term)

            self.log("regularization_term", reg_term, on_step=False)
            self.log("regularized_train_loss", loss, on_step=False)

        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        return self._shared_eval(batch, batch_idx, "val")

    def on_train_epoch_start(self) -> None:
        temp = self.encoder.update_temp(self.current_epoch, self.trainer.max_epochs)
        self.log("temp", temp, on_step=False, prog_bar=True)

    def on_epoch_end(self) -> None:
        mean_max = self.encoder.calc_mean_max()
        self.log("mean_max", mean_max, on_step=False, prog_bar=True)

    def _shared_eval(
        self, batch: torch.Tensor, batch_idx: int, prefix: str
    ) -> torch.Tensor:
        """Calculate the loss for a batch.

        Args:
            batch (torch.Tensor): batch data.
            batch_idx (int): batch id.
            prefix (str): prefix for logging.

        Returns:
            torch.Tensor: calculated loss.
        """
        data = batch["target"]
        _, decoded = self.forward(data)
        loss = F.mse_loss(decoded, data)

        self.log(f"{prefix}_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        return loss


@MODEL_REGISTRY
class FCNDecoder(pl.LightningModule):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_layers: int = 2,
        learning_rate: float = 1e-3,
    ):
        """Fully Connected Network decoder

        Args:
            input_size (int): input size of the network
            output_size (int): output size of the network
            hidden_layers (int, optional): number of hidden layers. Defaults to 2.
            learning_rate (float, optional): learning rate. Defaults to 1e-3.
        """
        super(FCNDecoder, self).__init__()
        self.learning_rate = learning_rate
        self.decoder = Decoder(input_size, output_size, hidden_layers)

    def forward(self, x):
        return self.decoder(x)

    def configure_optimizers(self) -> torch.optim.Adam:
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def training_step(
        self,
        batch: dict[str, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        return self._shared_eval(batch, batch_idx, "train")

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        return self._shared_eval(batch, batch_idx, "val")

    def _shared_eval(
        self, batch: torch.Tensor, batch_idx: int, prefix: str
    ) -> torch.Tensor:
        """Calculate the loss for a batch.

        Args:
            batch (torch.Tensor): batch data.
            batch_idx (int): batch id.
            prefix (str): prefix for logging.

        Returns:
            torch.Tensor: calculated loss.
        """
        data, target = batch["sample"], batch["target"]

        decoded = self.forward(data)
        loss = F.mse_loss(decoded, target)

        self.log(f"{prefix}_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        return loss


@MODEL_REGISTRY
class SphericalDecoder(pl.LightningModule):
    def __init__(
        self,
        n_ti: int,
        n_te: int,
        linear_layer_input_size: int,
        n_shells: list[int] = [5, 8, 16],
        L: list[int] = [2, 2, 0],
        learning_rate: float = 1e-3,
    ) -> None:
        """Spherical decoder based on: "A Spherical Convolutional Neural Network for White Matter Structure Imaging via dMRI" by Sedlar et al.

        Args:
            n_ti (int): number of unique TI values
            n_te (int): number of unique TE values
            linear_layer_input_size (int): size of the input of the first linear layer.
            n_shells (list[int], optional): number of b-value shells. List should be of size 3. Defaults to [5, 8, 16].
            L (list[int], optional): degree of spherical harmonic. List should be of size 3. Defaults to [2, 2, 0].
            learning_rate (float, optional): learning rate. Defaults to 1e-3.
        """
        super(SphericalDecoder, self).__init__()

        self.learning_rate = learning_rate
        self.spherical = torch.nn.Sequential(
            S2Convolution(n_ti, n_te, L[0], n_shells[0], n_shells[1]),
            QuadraticNonLinearity(L[0], L[1]),
            SO3Convolution(n_ti, n_te, L[1], n_shells[1], n_shells[2]),
            QuadraticNonLinearity(L[1], L[2]),
        )
        self.linear = torch.nn.Linear(linear_layer_input_size, 1344)

    def forward(self, x: dict[int, torch.Tensor]) -> torch.Tensor:
        _, features = self.spherical(x)
        return self.linear(features)

    def configure_optimizers(self) -> torch.optim.Adam:
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def training_step(
        self,
        batch: dict[str, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        return self._shared_eval(batch, batch_idx, "train")

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        return self._shared_eval(batch, batch_idx, "val")

    def _shared_eval(
        self, batch: torch.Tensor, batch_idx: int, prefix: str
    ) -> torch.Tensor:
        """Calculate the loss for a batch.

        Args:
            batch (torch.Tensor): batch data.
            batch_idx (int): batch id.
            prefix (str): prefix for logging.

        Returns:
            torch.Tensor: calculated loss.
        """
        data, target = batch["sample"], batch["target"]

        decoded = self.forward(data)
        loss = F.mse_loss(decoded, target)

        self.log(f"{prefix}_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        return loss
