import numpy as np


class Softmax:
    def __call__(self, x):
        return np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)

    def __str__(self):
        return "Softmax"

    @staticmethod
    def grad(x):
        return np.ones_like(x)


class MSELoss:
    def __call__(self, y_true, y_pred):
        """
        :param y_true: Ground Truth
        :param y_pred: Prediction
        :return: Mean Squared Error
        """
        self.y_true, self.y_pred = y_true, y_pred
        return np.mean((self.y_true - self.y_pred) ** 2)

    def grad(self):
        return -2 * (self.y_true - self.y_pred)


class BCELoss:
    def __call__(self, y, y_pred):
        self.y, self.y_pred = y, y_pred

    def grad(self):
        pass


class CrossEntropyLoss:
    def __call__(self, y, y_pred):
        self.y, self.y_pred = y, y_pred
        return np.mean(np.sum(-y * np.log(y_pred), axis=-1))

    def grad(self):
        return self.y_pred - self.y
