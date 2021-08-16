import math
from collections import OrderedDict

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn
from utils.logger import logger


class Encoder(pl.LightningModule):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        max_temp: float = 10.0,
        min_temp: float = 0.1,
        alpha: float = 0.99999,
    ) -> None:
        """Feature selection encoder. Implemented according to [_Concrete Autoencoders for Differentiable Feature Selection and Reconstruction_](https://arxiv.org/abs/1901.09346).

        Args:
            input_size (int): size of the input layer. Should be the same as the `output_size` of the decoder.
            output_size (int): size of the latent layer. Should be the same as the `input_size` of the decoder.
            max_temp (float, optional): maximum temperature for Gumble Softmax. Defaults to 10.0.
            min_temp (float, optional): minimum temperature for Gumble Softmax. Defaults to 0.1.
            alpha (float, optional): amount to multiply with current `temp` to decrease it. Should be `< 1.0`. Defaults
            to 0.99999.
        """
        super(Encoder, self).__init__()

        self.temp = torch.tensor(max_temp, device=self.device)
        self.min_temp = torch.tensor(min_temp, device=self.device)
        self.alpha = torch.tensor(alpha, device=self.device)

        logits = nn.init.xavier_normal_(torch.empty(output_size, input_size))
        self.logits = nn.parameter.Parameter(logits, requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Uses the trained encoder to make inferences.

        Args:
            x (torch.Tensor): input data. Should be the same size as the encoder input.

        Returns:
            torch.Tensor: encoder output of size `output_size`.
        """
        logits_size = self.logits.size()

        self.temp = torch.maximum(self.min_temp, self.temp * self.alpha)

        selections: torch.Tensor = None
        if self.training:
            uniform = torch.rand(logits_size, device=self.device)
            gumbel = -torch.log(-torch.log(uniform))
            noisy_logits = (self.logits + gumbel) / self.temp
            samples = F.softmax(noisy_logits, dim=1)
            selections = samples
        else:
            dim_argmax = len(self.logits.size()) - 1
            discrete_logits = F.one_hot(
                torch.argmax(self.logits, dim_argmax), num_classes=logits_size[1]
            )
            selections = discrete_logits

        encoded = torch.matmul(x, torch.transpose(selections.float(), 0, 1))
        return encoded


class Decoder(pl.LightningModule):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        n_hidden_layers: int,
        negative_slope: float = 0.2,
    ) -> None:
        """Standard decoder. It generates a network from `input_size` to `output_size`. The layers are generates as follows:
        ```python
        import numpy as np
        step_size = abs(output_size - input_size) // n_hidden_layers
        layer_sizes = np.arange(input_size, output_size, step_size)
        ```

        Args:
            input_size (int): size of the latent layer. Should be the same as the `output_size` of the encoder.
            output_size (int): size of the output layer. Should be the same as `input_size` of the encoder.
            n_hidden_layers (int): number of hidden layers. If 0 then the input will be directly connected to the output.
            negative_slope (float, optional): negative slope for the Leaky ReLu activation layer. Defaults to 0.2.
        """
        super(Decoder, self).__init__()

        step_size = abs(output_size - input_size) // (n_hidden_layers + 1)
        layer_sizes = np.arange(input_size, output_size, step_size)
        n_layers = len(layer_sizes)

        # Construct the network
        layers = OrderedDict()
        for i in range(n_layers):
            if i + 1 == n_layers:  # Last layer
                layers[f"linear_{i}"] = nn.Linear(layer_sizes[i], output_size)
                layers[f"sigmoid_{i}"] = nn.Sigmoid()
            else:
                layers[f"linear_{i}"] = nn.Linear(layer_sizes[i], layer_sizes[i + 1])
                layers[f"relu_{i}"] = nn.LeakyReLU(negative_slope)

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


class ConcreteAutoencoder(pl.LightningModule):
    def __init__(
        self,
        input_output_size: int,
        latent_size: int,
        decoder_hidden_layers: int = 2,
        learning_rate: float = 1e-2,
        max_temp: float = 10.0,
        min_temp: float = 0.1,
    ) -> None:
        """Trains a concrete autoencoder. Implemented according to [_Concrete Autoencoders for Differentiable Feature Selection and Reconstruction_](https://arxiv.org/abs/1901.09346).

        Args:
            input_output_size (int): size of the input and output layer.
            latent_size (int): size of the latent layer.
            decoder_hidden_layers (int, optional): number of hidden layers for the decoder. Defaults to 2.
            learning_rate (float, optional): learning rate for the optimizer. Defaults to 1e-2.
            max_temp (float, optional): maximum temperature for Gumble Softmax. Defaults to 10.0.
            min_temp (float, optional): minimum temperature for Gumble Softmax. Defaults to 0.1.
        """
        super(ConcreteAutoencoder, self).__init__()
        self.save_hyperparameters()

        self.encoder = Encoder(input_output_size, latent_size, max_temp, min_temp)
        self.decoder = Decoder(latent_size, input_output_size, decoder_hidden_layers)

        self.learning_rate = learning_rate

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
        """Initializes the Adam optimizer.

        Returns:
            torch.optim.Adam: the created optimizer.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """Trains one batch of data.

        Args:
            batch (torch.Tensor): batch data
            batch_idx (int): batch index

        Returns:
            torch.Tensor: calculated loss
        """
        return self._shared_eval(batch, batch_idx, "train")

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        """Validateds one batch of data.

        Args:
            batch (torch.Tensor): batch data.
            batch_idx (int): batch id.
        """
        return self._shared_eval(batch, batch_idx, "val")

    def on_epoch_start(self) -> None:
        """Each epoch the mean max of probabilities of the feature selector of the encoder is calculated. This is
        tracked for early stopping under the name `mean_max`."""
        mean_max = torch.mean(
            torch.max(F.softmax(self.encoder.logits, dim=1), 1).values
        )
        self.log("mean_max", mean_max, prog_bar=True)

    def on_train_start(self) -> None:
        """At the beginning of training the `alpha` for the encoder is calculated."""
        dataset = self.train_dataloader()
        batch_size = dataset.batch_size

        num_epochs = self.trainer.max_epochs
        min_temp = self.encoder.min_temp
        temp = self.encoder.temp

        alpha = math.exp(math.log(min_temp / temp) / (num_epochs * batch_size))
        self.encoder.alpha = torch.tensor(alpha, device=self.device)

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
        encoded = self.encoder(batch)
        decoded = self.decoder(encoded)
        loss = F.mse_loss(decoded, batch)

        self.log(f"{prefix}_loss", loss)

        return loss
