import numpy as np


class Relu:
    def __call__(self, x):
        return np.maximum(np.zeros_like(x), x)

    def __str__(self):
        return "ReLU"

    @staticmethod
    def grad(x):
        return (x >= 0).astype(np.float32)


class Sigmoid:
    def __call__(self, x):
        return 1 / (1 + np.exp(-x))

    def __str__(self):
        return "Sigmoid"

    def grad(self, x):
        return self.__call__(x) * (1 - self.__call__(x))


class Linear:
    def __call__(self, x):
        return x

    def __str__(self):
        return "Linear"

    @staticmethod
    def grad(x):
        return np.ones_like(x)


class Tanh:
    def __call__(self, x):
        return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

    def __str__(self):
        return "Tanh"

    def grad(self, x):
        return 1 - (self.__call__(x) ** 2)


class LeakyReLU:
    def __init__(self, alpha=0.01):
        self.alpha = alpha

    def __call__(self, x):
        return np.maximum(self.alpha * x, x)

    def __str__(self):
        return "LeakyReLU"

    def grad(self, x):
        g = (x >= 0).astype(np.float32)
        return np.where(g == 1, 1, -self.alpha)


class SiLU:
    def __init__(self):
        self.sigmoid = Sigmoid()

    def __call__(self, x):
        return x * self.sigmoid(x)

    def __str__(self):
        return "SiLU"

    def grad(self, x):
        return 1 * self.sigmoid(x) + x * self.sigmoid.grad(x)


class Softmax:
    def __call__(self, x):
        return np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)

    def __str__(self):
        return "Softmax"

    @staticmethod
    def grad(x):
        return np.ones_like(x)


def main():
    leaky_relu = LeakyReLU(alpha=0)
    x = np.array([[-10, 2, 5, 3, -5, -9]])
    print(leaky_relu(x))
    print(leaky_relu.grad(x))


if __name__ == "__main__":
    main()
