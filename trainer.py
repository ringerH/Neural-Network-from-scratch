import numpy as np
from sklearn.metrics import r2_score

class Trainer:
    def __init__(self, model):
        self.model = model
        self.accuracy = r2_score
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []

    def forward(self, x):
        return self.model(x)

    def training_step(self, x, y):
        y_pred = self.forward(x)
        loss = self.model.loss_fxn(y, y_pred)
        acc = self.accuracy(y, y_pred)

        self.model.backward()
        self.model.update_gradients()

        return loss, acc

    def validation_step(self, x, y):
        y_pred = self.forward(x)
        loss = self.model.loss_fxn(y, y_pred)
        acc = self.accuracy(y, y_pred)

        return loss, acc

    def go_one_epoch(self, loader, step_fxn):
        loss, acc = 0, 0
        for x, y in loader:
            loss_batch, acc_batch = step_fxn(x, y)
            loss += loss_batch
            acc += acc_batch

        return loss / len(loader), acc / len(loader)

    def train(self, train_loader, val_loader, epochs):
        for epoch in range(epochs):
            train_loss, train_acc = self.go_one_epoch(train_loader, self.training_step)
            val_loss, val_acc = self.go_one_epoch(val_loader, self.validation_step)

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)

            if epoch % 10 == 0:
                print(f"Epoch:[{epoch}]")
                print(f"Train: [loss: {train_loss:.4f} acc: {train_acc:.4f}]")
                print(f"Val: [loss: {val_loss:.4f} acc: {val_acc:.4f}]")
                print()
