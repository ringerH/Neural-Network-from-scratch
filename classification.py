import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

import sklearn
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from nn.net import Model
from nn.losses import CrossEntropyLoss
from nn.layer import Layer
from nn.dataloader import DataLoader

class Trainer:
    def __init__(self,
                 model):
        self.model = model
        self.accuracy = accuracy_score

    def forward(self, x):
        return self.model(x)

    def training_step(self, x, y):
        y_pred = self.forward(x)
        loss = self.model.loss_fxn(y, y_pred)
        acc = self.accuracy(np.argmax(y, axis=-1), np.argmax(y_pred, axis=-1))

        self.model.backward()
        self.model.update_gradients()

        return loss, acc

    def validation_step(self, x, y):
        y_pred = self.forward(x)
        loss = self.model.loss_fxn(y, y_pred)
        acc = self.accuracy(np.argmax(y, axis=-1), np.argmax(y_pred, axis=-1))

        return loss, acc

    @staticmethod
    def go_one_epoch(loader, step_fxn):
        loss, acc = 0, 0
        for x, y in loader:
            loss_batch, acc_batch = step_fxn(x, y)
            loss += loss_batch
            acc += acc_batch
        return loss/len(loader), acc/len(loader)

    def train(self,
              train_loader,
              val_loader,
              epochs):

        for epoch in (range(epochs)):
            train_loss, train_acc = self.go_one_epoch(train_loader, self.training_step)
            val_loss, val_acc = self.go_one_epoch(val_loader, self.validation_step)

            if epoch % 10 == 0:
                print(f"Epoch:[{epoch}]")
                print(f"Train:[loss: {train_loss:.4f} acc:{train_acc:.4f}]")
                print(f"Val: [loss: {val_loss:.4f} acc:{val_acc:.4f}]")
                print()


def main():

    X, y = make_classification(n_samples=3000, n_features=10, n_classes=3, n_informative=3)
    onehot_encoder = sklearn.preprocessing.OneHotEncoder()
    y = onehot_encoder.fit_transform(y.reshape(-1, 1)).toarray()
    standard_scaler = StandardScaler()
    X = standard_scaler.fit_transform(X)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)


    train_loader = DataLoader(X_train, y_train, batch_size=64, drop_last=False)
    val_loader = DataLoader(X_val, y_val, batch_size=64, drop_last=False)

    model = Model()
    model.add(Layer(10, 16, 'sigmoid'))
    model.add(Layer(16, 32, 'sigmoid'))
    model.add(Layer(32, 16, 'sigmoid'))
    model.add(Layer(16, 3, 'softmax'))

    model.loss_fxn = CrossEntropyLoss()
    model.lr = 3e-2

    print(model)

    trainer = Trainer(model=model)

    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=500
    )


if __name__ == "__main__":
    main()

