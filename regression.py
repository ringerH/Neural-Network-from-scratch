import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import sys
from nn.net import Model
from nn.losses import MSELoss
from nn.layer import Layer
from nn.dataloader import DataLoader

class Trainer:
    def __init__(self,
                 model):
        self.model = model
        self.accuracy = r2_score
        self.train_losses = []

    def forward(self, x):
        return self.model(x)

    def training_step(self, x, y):
        y_pred = self.forward(x)
        loss = self.model.loss_fxn(y, y_pred)
        acc = self.accuracy(y_pred, y)

        self.model.backward()
        self.model.update_gradients()
        return loss, acc

    def validation_step(self, x, y):
        y_pred = self.forward(x)
        loss = self.model.loss_fxn(y, y_pred)
        acc = self.accuracy(y, y_pred)
        return loss, acc

    def go_one_batch(self, loader, step_fxn):
        loss, acc = 0, 0

        for x, y in loader:
            loss_batch, acc_batch = step_fxn(x, y)
            loss += loss_batch
            acc += acc_batch

        return loss / len(loader), acc / len(loader)

    def train(self,
              train_loader,
              val_loader,
              epochs):

        for epoch in tqdm(range(epochs)):
            train_loss, train_acc = self.go_one_batch(train_loader, self.training_step)
            val_loss, val_acc = self.go_one_batch(val_loader, self.validation_step)

            self.train_losses.append(train_loss)
            if epoch % 10 == 0:
                print(f"Epoch:[{epoch}]")
                print(f"Train:[loss: {train_loss:.4f} acc:{train_acc:.4f}]")
                print(f"Val: [loss: {val_loss:.4f} acc:{val_acc:.4f}]")
                print()


def main():
    # Create synthetic regression data
    X, y = make_regression(n_samples=1000, n_features=10, noise=30)
    X_train, X_val, y_train, y_val = train_test_split(X, y.reshape(-1, 1), test_size=0.2)

    # create train and val dataloaders
    train_loader = DataLoader(X_train, y_train, batch_size=32, drop_last=False)
    val_loader = DataLoader(X_val, y_val, batch_size=32, drop_last=False, shuffle=False)

    model = Model()

    model.add(Layer(10, 16, 'sigmoid'))
    model.add(Layer(16, 32, 'sigmoid'))
    model.add(Layer(32, 1, 'linear'))

    model.loss_fxn = MSELoss()
    model.lr = 1e-3

    # create the trainer object
    trainer = Trainer(model = model)

    # train the model
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=100)


if __name__ == "__main__":
    main()

