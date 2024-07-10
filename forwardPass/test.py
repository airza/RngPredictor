#!/usr/bin/env python3
import torch.nn as nn
import torch
from torch.utils.data import DataLoader, TensorDataset
from forwardPass.DifferentiableShiftNetwork import DifferentiableShiftNetwork
from forwardPass.LogicGateNode import make_correct_node
device = torch.device("mps")
torch.set_default_device(device)
bits = 4
torch.set_printoptions(precision=4,sci_mode=False)
mse = nn.MSELoss()

count = 2048
test_count = 100
shift = -3
X = torch.rand(count+test_count,bits)
X_test, X_train = X[:test_count], X[test_count:]
Y = torch.roll(X, shift, 1)
Y[:,shift:] = 0.0
Y_test, Y_train = Y[:test_count], Y[test_count:]
epochs = 100000
train_loader = DataLoader(TensorDataset(X_train,Y_train),batch_size=4)
model = DifferentiableShiftNetwork(bits, 1)
optimizer = torch.optim.SGD(model.parameters(),lr=1,momentum=.5,nesterov=True)
for epoch in range(epochs):
    total_loss = 0.0
    model.train()
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        batch_X = batch_X
        batch_y = batch_y
        outputs = model(batch_X)
        loss = mse(outputs, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f'Epoch {epoch}, Loss: {total_loss / len(train_loader)}')
    if total_loss / len(train_loader) < 0.1:
        break
model.l1 = make_correct_node(bits, 'and', 0, 1)
model.l2 = make_correct_node(bits, 'xor', 2, 3)
model.l3 = make_correct_node(2, 'and', 0, 1)
model.requires_grad_(False)
x = torch.rand(1, bits, requires_grad=True)
print(x.shape)
x = torch.tensor([.49 + i / 100 if i != 3 else 1 for i in range(4)]).float().unsqueeze(0).requires_grad_(True)
print(x.shape)
y = torch.tensor([1.0])
print(x.round())
for i in range(1000):
    outputs = model(x)
    loss = mse(outputs, y)
    loss.backward()
    print(loss.item())
    if loss.item() < .1:
        break
    with torch.no_grad():
        print(x.grad)
        clamp_grad = torch.clamp(x.grad, -.1, .1)
        x -= clamp_grad
        torch.clamp(x, 0, 1, out=x)
print(x.round())
print(x)