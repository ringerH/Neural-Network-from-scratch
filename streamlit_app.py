import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from nn.net import Model
from nn.losses import MSELoss
from nn.layer import Layer
from nn.dataloader import DataLoader
from trainer import Trainer 

def main():
    st.title("Neural Network Training and Visualization")

    st.sidebar.title("Configuration")
    epochs = st.sidebar.slider("Epochs", 1, 500, 100)
    batch_size = st.sidebar.selectbox("Batch Size", [16, 32, 64, 128], index=1)
    learning_rate = st.sidebar.slider("Learning Rate", 1e-5, 1e-1, 1e-3)

    st.write("### Generate Synthetic Regression Data")
    X, y = make_regression(n_samples=1000, n_features=10, noise=30)
    X_train, X_val, y_train, y_val = train_test_split(X, y.reshape(-1, 1), test_size=0.2)

    standard_scaler = StandardScaler()
    X_train = standard_scaler.fit_transform(X_train)
    X_val = standard_scaler.transform(X_val)

    train_loader = DataLoader(X_train, y_train, batch_size=batch_size, drop_last=False)
    val_loader = DataLoader(X_val, y_val, batch_size=batch_size, drop_last=False, shuffle=False)

    model = Model()
    model.add(Layer(10, 16, 'sigmoid'))
    model.add(Layer(16, 32, 'sigmoid'))
    model.add(Layer(32, 1, 'linear'))

    model.loss_fxn = MSELoss()
    model.lr = learning_rate

    st.write("### Model Structure")
    st.text(model)

    if st.button("Train Model"):
        trainer = Trainer(model=model)
        with st.spinner("Training..."):
            trainer.train(train_loader, val_loader, epochs)

        st.write("### Training Complete")

        st.write("### Loss Plot")
        plt.figure(figsize=(14, 6))
        plt.subplot(1, 2, 1)
        plt.plot(trainer.train_losses, label='Train Loss')
        plt.plot(trainer.val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss vs Epoch')
        plt.legend()

        st.pyplot(plt)

        st.write("### R2 Score Plot")
        plt.subplot(1, 2, 2)
        plt.plot(trainer.train_accuracies, label='Train Accuracy')
        plt.plot(trainer.val_accuracies, label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('R2 Score')
        plt.title('R2 Score vs Epoch')
        plt.legend()

        st.pyplot(plt)

if __name__ == "__main__":
    main()
