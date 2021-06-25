import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def decoder_1l(n_features, device):
    """Simple decoder from the latent layer to the output layer"""
    dense1344 = nn.Linear(n_features, 1344, device=device)

    def decoder(x):
        return dense1344(x)

    return decoder


def decoder_2l(n_features, device):
    """Decoder from the latent layer to the output layer with an extra layer of size 1000 in between"""
    dense1000 = nn.Linear(n_features, 1000, device=device)
    dense1344 = nn.Linear(1000, 1344, device=device)
    act = nn.LeakyReLU(0.2)

    def decoder(x):
        x = act(dense1000(x))
        return dense1344(x)

    return decoder


def decoder_3l(n_features, device):
    """Complex decoder with two intermediate layers between the latent and output layer."""
    dense800 = nn.Linear(n_features, 800, device=device)
    dense1000 = nn.Linear(800, 1000, device=device)
    dense1344 = nn.Linear(1000, 1344, device=device)
    act = nn.LeakyReLU(0.2)

    def decoder(x):
        x = act(dense800(x))
        x = act(dense1000(x))
        return dense1344(x)

    return decoder


class ConcreteSelect(nn.Module):
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
        **kwargs
    ):
        super(ConcreteSelect, self).__init__(**kwargs)
        # encoder
        self.output_dim = output_dim
        # the input layer has output (None,N_params_in). In this case, probably equal to input_dim
        self.input_shape = input_shape
        self.decoder = decoder(n_features, device)
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
