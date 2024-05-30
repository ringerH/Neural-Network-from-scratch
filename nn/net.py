from nn.losses import *
import numpy as np

class Model:
    def __init__(self,
                 loss_fxn=None,
                 logger=None,
                 lr=1e-3,
                 type='regression',
                 epochs=1000,
                 verbose=False):

        self.loss_fxn = loss_fxn
        self.layers = []
        self.lr = lr
        self.dW, self.dB = [], []
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_acc': [], 'val_acc': []
        }
        self.logger = logger
        self.epochs = epochs
        self.verbose = verbose

    def __str__(self):
        out = ""
        for layer in self.layers:
            out += layer.__str__() + "\n"
        return out

    def add(self, layer):
        self.layers.append(layer)

    def __call__(self, x):
        '''
            x: (bs, dim_in)
        '''
        for layer in self.layers:
            x = layer(x)
        return x

    def backward(self):
        dLy = self.loss_fxn.grad()
        common = dLy

        for i in range(len(self.layers) - 1, -1, -1):
            dzw, dzx, daz = self.layers[i].get_grads()
            if i != len(self.layers) - 1:
                common = common @ self.layers[i + 1].W
            common = common * daz
            dw = common[:, :, None] * dzw
            db = common[:, :] * 1

            self.dW.append(np.mean(dw, axis=0))
            self.dB.append(np.mean(db, axis=0))

    def update_gradients(self):
        for i, (dw, db) in enumerate(zip(reversed(self.dW), reversed(self.dB))):
            self.layers[i].W += - self.lr * dw
            self.layers[i].b += - self.lr * db

        self.dW.clear()
        self.dB.clear()
