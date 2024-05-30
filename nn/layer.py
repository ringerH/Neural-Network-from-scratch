import numpy as np
from nn.activations import *
class Neuron:
    def __init__(self, dim_in, activation):
        self.dzw, self.dzx, self.daz = 0, 0, 0
        self.dim_in = dim_in
        self.activation = activation

    def get_grads(self):
        return [self.dzw, self.dzx, self.daz]

    def calculate_grad(self, x, z, w, index):
        self.dzw = x
        self.dzx = w[index]
        self.daz = self.activation.grad(z[:, index])
        return [self.dzw, self.dzx, self.daz]


activations = {
    'relu': Relu,
    'tanh': Tanh,
    'sigmoid': Sigmoid,
    'silu': SiLU,
    'leaky_relu': LeakyReLU,
    'linear': Linear,
    'softmax': Softmax
}

class Layer:
    def __init__(self, dim_in, dim_out, activation):
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.activation = activations[activation]()

        self.W = np.random.randn(self.dim_out, self.dim_in)
        self.b = np.random.randn(self.dim_out)

        self.neurons = [Neuron(self.dim_in, self.activation) for _ in range(self.dim_out)]

        self.dzw, self.dzx, self.daz = [], [], []

    def get_grads(self):
        grads = [np.stack(self.dzw, axis=1),
                 np.stack(self.dzx, axis=-1),
                 np.stack(self.daz, axis=-1)]

        self.dzw.clear()
        self.dzx.clear()
        self.daz.clear()
        return grads

    def __str__(self):
        return f"Layer: [in:{self.dim_in}] [out:{self.dim_out}] [activation:{self.activation}]"

    def __call__(self, x):
        """
            x: (bs, dim_in)
        """

        if x.shape[1] != self.dim_in:
            raise TypeError(f'Input should have dimension {self.dim_in} but found {x.shape[1]}')

        z = x @ self.W.T + self.b
        self.a = self.activation(z)

        self.daz.clear()
        self.dzx.clear()
        self.dzw.clear()

        for i, neuron in enumerate(self.neurons):
            dzw, dzx, daz = neuron.calculate_grad(x, z, self.W, i)
            self.dzw.append(dzw)
            self.dzx.append(dzx)
            self.daz.append(daz)

        return self.a
