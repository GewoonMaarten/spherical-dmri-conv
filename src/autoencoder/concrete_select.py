from collections import OrderedDict

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn


class Encoder(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        max_temp: float = 10.0,
        min_temp: float = 0.1,
        alpha: float = 0.99999,
    ) -> None:
        """Feature selection encoder. Implemented according to [_Concrete Autoencoders for Differentiable Feature Selection and Reconstruction_](https://arxiv.org/abs/1901.09346)

        Args:
            input_size (int): size of the input layer. Should be the same as the `output_size` of the decoder.
            output_size (int): size of the latent layer. Should be the same as the `input_size` of the decoder.
            max_temp (float, optional): maximum temperature for Gumble Softmax. Defaults to 10.0.
            min_temp (float, optional): minimum temperature for Gumble Softmax. Defaults to 0.1.
            alpha (float, optional): amount to multiply with current `temp` to decrease it. Should be `< 1.0`. Defaults
            to 0.99999.
        """
        super(Encoder, self).__init__()

        self.temp = torch.tensor(max_temp)
        self.min_temp = torch.tensor(min_temp)
        self.alpha = torch.tensor(alpha)

        logits = nn.init.xavier_normal_(torch.empty(output_size, input_size))
        self.logits = nn.parameter.Parameter(logits, requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits_size = self.logits.size()

        self.temp = torch.maximum(self.min_temp, self.temp * self.alpha)

        selections: torch.Tensor = None
        if self.training:
            uniform = torch.rand(logits_size)
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


class Decoder(nn.Module):
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
            n_hidden_layers (int): number of hidden layers.
            negative_slope (float, optional): negative slope for the Leaky ReLu activation layer. Defaults to 0.2.
        """
        super(Decoder, self).__init__()

        step_size = abs(output_size - input_size) // n_hidden_layers
        layer_sizes = np.arange(input_size, output_size, step_size)

        # Construct the network
        layers = OrderedDict()
        for i in range(len(layer_sizes)):
            if i + 1 == len(layer_sizes):  # Last layer
                layers[f"linear_{i}"] = nn.Linear(layer_sizes[i], output_size)
                layers[f"sigmoid_{i}"] = nn.Sigmoid()
            else:
                layers[f"linear_{i}"] = nn.Linear(layer_sizes[i], layer_sizes[i + 1])
                layers[f"relu_{i}"] = nn.LeakyReLU(negative_slope)

        self.decoder = nn.Sequential(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        decoded = self.decoder(x)
        return decoded


class ConcreteAutoencoder(nn.Module):
    def __init__(
        self,
        output_dim,
        input_shape,
        decoder,
        device,
        n_features=500,
        start_temp=10.0,
        min_temp=0.1,
        alpha=0.99999,
        **kwargs,
    ):
        super(ConcreteAutoencoder, self).__init__(**kwargs)
        # encoder
        self.output_dim = output_dim
        # the input layer has output (None,N_params_in). In this case, probably equal to input_dim
        self.input_shape = input_shape
        self.decoder = decoder(n_features, device, input_shape)
        self.device = device
        self.start_temp = start_temp
        # self.min_temp = K.constant(min_temp)
        self.min_temp = nn.init.constant_(
            torch.tensor(np.zeros(1), device=self.device), min_temp
        )
        # self.alpha = K.constant(alpha)
        self.alpha = nn.init.constant_(
            torch.tensor(np.zeros(1), device=self.device), alpha
        )
        # self.name = name

        # equivalent to build in Keras
        self.temp = torch.tensor([self.start_temp], device=self.device)
        tensor_logits = nn.init.xavier_normal_(
            torch.empty(self.output_dim, self.input_shape, device=self.device)
        )
        self.logits = nn.Parameter(tensor_logits, requires_grad=True)

    # equivalent to call in Keras -> encoder, the concrete layer itself

    def encoder(self, X):

        uniform = torch.rand(self.logits.size(), device=self.device)
        gumbel = -torch.log(-torch.log(uniform))
        self.temp = torch.maximum(self.min_temp, self.temp * self.alpha)
        # print('temperature {}'.format(self.temp))
        # noisy_logits = (self.logits + gumbel.to(device)) / self.temp
        noisy_logits = (self.logits + gumbel) / self.temp
        samples = F.softmax(noisy_logits, dim=1)

        # numClasses = self.logits.size()[1]
        dim_argmax = len(self.logits.size()) - 1
        discrete_logits = F.one_hot(
            torch.argmax(self.logits, dim_argmax), num_classes=self.logits.size()[1]
        )

        if self.training:
            self.selections = samples
        else:
            self.selections = discrete_logits

        # Y = torch.dot(X,torch.transpose(self.selections, 0, 1))
        # dot is not exactly equal to a dot product, it could be a matrix product in keras
        Y = torch.matmul(X, torch.transpose(self.selections.float(), 0, 1))
        return Y

    def forward(self, X):
        y = self.encoder(X)  # selected features
        x = self.decoder(y)  # reconstructed signals

        return x, y
