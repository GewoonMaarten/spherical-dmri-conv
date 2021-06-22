import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class ConcreteSelect(nn.Module):

    def __init__(self, output_dim, input_shape, device, n_features=500, start_temp=10.0, min_temp=0.1, alpha=0.99999, **kwargs):
        super(ConcreteSelect, self).__init__(**kwargs)
        # encoder
        self.output_dim = output_dim
        # the input layer has output (None,N_params_in). In this case, probably equal to input_dim
        self.input_shape = input_shape
        self.device = device
        self.start_temp = start_temp
        #self.min_temp = K.constant(min_temp)
        self.min_temp = nn.init.constant_(
            torch.tensor(np.zeros(1), device=self.device), min_temp)
        #self.alpha = K.constant(alpha)
        self.alpha = nn.init.constant_(
            torch.tensor(np.zeros(1), device=self.device), alpha)
        #self.name = name

        # equivalent to build in Keras
        self.temp = torch.tensor([self.start_temp], device=self.device)
        tensor_logits = nn.init.xavier_normal_(torch.empty(
            self.output_dim, self.input_shape, device=self.device))
        self.logits = nn.Parameter(tensor_logits, requires_grad=True)

        # for the decoder, we define three different Linear/dense layers and the activation function
        self.dense800 = nn.Linear(n_features, 800)
        # self.dense800 = nn.Linear(500,800) # the example for the standard 500 features value
        self.dense1000 = nn.Linear(800, 1000)
        self.dense1344 = nn.Linear(1000, 1344)
        self.act = nn.LeakyReLU(0.2)

    # equivalent to call in Keras -> encoder, the concrete layer itself
    def encoder(self, X, training=None):

        uniform = torch.rand(self.logits.size(), device=self.device)
        gumbel = -torch.log(-torch.log(uniform))
        self.temp = torch.maximum(self.min_temp, self.temp * self.alpha)
        #print('temperature {}'.format(self.temp))
        #noisy_logits = (self.logits + gumbel.to(device)) / self.temp
        noisy_logits = ((self.logits + gumbel) / self.temp)
        samples = F.softmax(noisy_logits, dim=1)

        #numClasses = self.logits.size()[1]
        dim_argmax = len(self.logits.size()) - 1
        discrete_logits = F.one_hot(torch.argmax(
            self.logits, dim_argmax), num_classes=self.logits.size()[1])

        # probably unnecessary
        if training is None:
            training = self.training

        if self.training:
            self.selections = samples
        else:
            self.selections = discrete_logits

        #Y = torch.dot(X,torch.transpose(self.selections, 0, 1))
        # dot is not exactly equal to a dot product, it could be a matrix product in keras
        Y = torch.matmul(X, torch.transpose(self.selections.float(), 0, 1))
        return Y

    # decoder: we suppose the two-layers scheme. In keras this is defined outside
    def decoder(self, x):
        # x.to("cpu")
        x = self.act(self.dense800(x))
        x = self.act(self.dense1000(x))
        x = self.dense1344(x)

        return x

    def forward(self, X, training=None):
        y = self.encoder(X)  # selected features
        x = self.decoder(y)  # reconstructed signals

        return x, y
