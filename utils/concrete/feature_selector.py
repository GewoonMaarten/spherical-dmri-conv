import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from ..logger import logger
from .concrete_select import ConcreteSelect
from .stopper import StopperCallback


class ConcreteAutoencoderFeatureSelector():

    # def __init__(self, K, output_function, num_epochs = 100, learning_rate = 0.001, start_temp = 10.0, min_temp = 0.1, tryout_limit = 5, input_dim = 1344, callback=None, writer=None): #batch_size = None,
    # , losstrain=None, lossval=None): #batch_size = None,
    def __init__(self, K, decoder, device, num_features=500, num_epochs=100, learning_rate=0.001, start_temp=10.0, min_temp=0.1, tryout_limit=5, input_dim=1344, checkpt=True, callback=None, writer=None, path=''):
        self.K = K  # equivalent to output_dim
        self.decoder = decoder
        self.device = device
        # self.output_function = output_function # this function is now included in the ConcreteSelect class
        # but now we have to define the number of features to be extracted from the encoder
        self.num_features = num_features
        self.num_epochs = num_epochs
#         self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.start_temp = start_temp
        self.min_temp = min_temp
        self.tryout_limit = tryout_limit
        self.input_dim = input_dim
        self.checkpt = checkpt
        self.callback = callback
        self.writer = writer
        self.path = path  # str(Path(ROOT_PATH, 'runs', 'models'))
        #self.losstrain = losstrain
        #self.lossval = lossval

    def fit(self, X, val_X=None):
        #         if self.batch_size is None:
        #             self.batch_size = max(len(X) // 256, 16)

        num_epochs = self.num_epochs
        steps_per_epoch = X.__len__()  # (len(X) + self.batch_size - 1) // self.batch_size
        logger.info("steps per epoch: %s", steps_per_epoch)
        writer = self.writer
        # losses,losses_val=[],[]

        for i in range(self.tryout_limit):

            alpha = math.exp(
                math.log(self.min_temp / self.start_temp) / (num_epochs * steps_per_epoch))

            # we apply the model
            self.model = ConcreteSelect(
                self.K, self.input_dim, self.decoder, self.device, self.num_features, self.start_temp, self.min_temp, alpha).to(self.device)

            # we define the loss and the optimizer functions
            criterion = nn.MSELoss().to(self.device)
            optimizer = torch.optim.Adam(
                self.model.parameters(), lr=self.learning_rate)

            if self.checkpt == True:
                checkpoint = torch.load(
                    Path(self.path, 'runs', 'models', 'check15', 'model.pt'))
                self.model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                epoch_check = checkpoint['epoch']
                loss = checkpoint['loss']
            self.model.train()

            stopper_callback = StopperCallback()  # writer=self.writer)

            for epoch in range(num_epochs):

                if self.checkpt == True:
                    if epoch < epoch_check:
                        continue

                value_stop = torch.mean(
                    torch.max(F.softmax(self.model.logits, dim=1), 1).values)

                if value_stop >= stopper_callback.mean_max_target:
                    break

                self.model.train()

                total_loss, total_val_loss = 0, 0
                for j, signals in enumerate(X):
                    signals = signals.to(self.device)

                    # steps in pytorch:
                    # 1. Initialise gradients at the start of each batch
                    # 2. Run the forward and then the backwards pass
                    # 3. Compute the loss and update the weights

                    # Initialise gradients
                    optimizer.zero_grad()

                    outputs, _ = self.model(signals)
                    # like criterion(yhat,target) -> the target in the autoencoder is the input
                    loss = criterion(outputs, signals)

                    writer.add_scalar(
                        str(Path(self.path, 'runs', 'scalars')), loss, epoch)

                    # print('Epoch {}: Loss = {}'.format(epoch+1, loss.item())) # just to check how it's going

                    # Backward pass
                    loss.backward()
                    # Compute the loss and update the weights
                    optimizer.step()

                    total_loss += loss.item()

                if val_X is not None:
                    # Evaluate the model
                    self.model.eval()

                    #steps_per_epoch_val = val_X.__len__()
                    for j, signals in enumerate(val_X):
                        signals = signals.to(self.device)
                        outputs_pred, _ = self.model(signals)

                        loss = criterion(outputs_pred, signals)
                        total_val_loss += loss.item()

                total_loss = total_loss / len(X)
                total_val_loss = total_val_loss / len(val_X)

                logger.info('epoch: %s/%s, loss: %4.4f, val loss: %4.4f',
                            epoch + 1, num_epochs, total_loss, total_val_loss)
                logger.info('mean max of probabilities: %.8f, temperature: %.8f',
                            value_stop.item(), self.model.temp.item())

                # save for checkpoint
                torch.save({'epoch': epoch,
                            'model_state_dict': self.model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'loss': loss.item(),
                            }, Path(self.path, 'runs', 'models', f'checkpoint_epoch={num_epochs}_net.pth'))

            num_epochs *= 2

        self.probabilities = F.softmax(self.model.logits, dim=1)
        self.indices = torch.argmax(self.model.logits, 1)

        return self

    def get_indices(self):
        val = torch.argmax(self.model.logits, 1)
        return val
        # return K.get_value(K.argmax(self.model.get_layer('concrete_select').logits))

    def get_mask(self):
        # nn.functional.one_hot(torch.argmax(self.logits),list(self.logits.size())[1], dim = )
        dim_argmax = len(self.model.logits.size())-1
        val = torch.sum(nn.functional.one_hot(torch.argmax(
            self.model.logits, dim_argmax), self.model.logits.size()[1]))
        return val
        # return K.get_value(K.sum(K.one_hot(K.argmax(self.model.get_layer('concrete_select').logits), self.model.get_layer('concrete_select').logits.shape[1]), axis = 0))

    def get_support(self, indices=False):
        return self.get_indices() if indices else self.get_mask()

    def get_params(self):
        return self.model
        # return self.output_function(self.concrete_select)
