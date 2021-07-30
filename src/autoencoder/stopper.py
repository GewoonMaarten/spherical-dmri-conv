from pytorch_lightning.callbacks import EarlyStopping


class StopperCallback(EarlyStopping):

    def __init__(self, mean_max_target=0.998):  # , writer=None):
        self.mean_max_target = mean_max_target
        #self.writer = writer
        # super(StopperCallback, self).__init__(monitor = '', patience = float('inf'), verbose = 1, mode = 'max')#, baseline = self.mean_max_target)
        super(StopperCallback, self).__init__(monitor='',
                                              patience=float('inf'), verbose=True, mode='max')
