# Neural-Network-from-scratch
A self-coded Neural Network implementation from scratch for regression and classification tasks.  
Loss Functions: `MSE, CCE, BCE`
Activations: ```relu, tanh, sigmoid, softmax, linear, leaky_relu, silu```  

1. **Imports**
```
from net import Model
from losses import CrossEntropyLoss
from layer import Layer
from dataloader import DataLoader  
 ``` 
2. **Initialize model**  
 ```
model = Model()
model.add(Layer(dim_in=10, dim_out=16, activation='sigmoid'))
model.add(Layer(16, 32, 'sigmoid'))
model.add(Layer(32, 16, 'sigmoid'))
model.add(Layer(16, 3, 'softmax'))

model.loss_fxn = CrossEntropyLoss()
model.lr = 1e-3  
```  
3. **Dataloader**
```
train_loader = DataLoader(X_train, y_train, batch_size=64, drop_last=False)
val_loader = DataLoader(X_val, y_val, batch_size=64, drop_last=False)  
```
4. **Train-loop** 
```
for epoch in range(epochs):
    loss = 0
    for x, y in train_loader:
        y_pred = model(x)           # forward pass
        loss += model.loss_fxn(y_pred, y)
        model.backward()            # calculate gradients
        model.update_gradients()    # update weights

    loss = loss / len(train_loader)  # take the average loss  
```
5. **Validation Loop**  
 ```
loss = 0
for x, y in val_loader:
    y_pred = model(x)
    loss += model.loss_fxn(y_pred, y)
    # only forward pass and no calculating/updating gradients

loss = loss / len(val_loader)  
```
