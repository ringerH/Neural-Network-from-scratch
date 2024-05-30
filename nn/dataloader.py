import numpy as np
import math
class DataLoader:
    def __init__(self,
                 X,
                 y,
                 batch_size,
                 drop_last: bool = False,
                 shuffle: bool = True):

        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.i = 0

    def __iter__(self):
        return self

    def __len__(self):
        if self.drop_last:
            return math.floor(len(self.X) / self.batch_size)
        else:
            return math.ceil(len(self.X) / self.batch_size)

    def __next__(self):
        start, end = self.batch_size * self.i, self.batch_size * (self.i + 1)
        self.i = (self.i + 1)

        if start >= len(self.X):
            self.i = 0
            raise StopIteration

        if end > len(self.X):
            if self.drop_last:
                self.i = 0
                raise StopIteration
            else:
                return self.X[start:-1, ], self.y[start:-1, ]

        return self.X[start: end, ], self.y[start: end, ]


def main():
    X = np.random.randn(100, 5)
    y = np.random.randn(100, 1)

    train_loader = DataLoader(X, y, batch_size=32, drop_last=False)
    for (x, y) in train_loader:
        print(x.shape, y.shape)


if __name__ == "__main__":
    main()
