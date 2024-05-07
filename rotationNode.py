#!/usr/bin/env python3
import numpy as np
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

torch.set_printoptions(precision=4,sci_mode=False)
device = torch.device("mps")
torch.set_default_device(device)
mse = nn.MSELoss()
def visualize_loss_and_gradient(model, p1_struct,p2_struct, p1_range, p2_range,X,title,resolution=20):
    p1_param,p1_index = p1_struct
    p2_param,p2_index = p2_struct
    starting_p1 = p1_param[p1_index].item()
    starting_p2 = p2_param[p2_index].item()
    P1 = np.linspace (p1_range[0], p1_range[1], resolution)
    P2 = np.linspace (p2_range[0], p2_range[1], resolution)
    X_axis, Y_axis = np.meshgrid(P1, P2)
    losses = np.zeros_like(X_axis)
    grads_x = np.zeros_like(X_axis)
    grads_y = np.zeros_like(X_axis)
    for i in range(resolution):
        for j in range(resolution):
            # Temporarily set the model weights to these points
            with torch.no_grad():
                p1_param[p1_index] = P1[i]
                p2_param[p2_index] = P2[j]
            Y_pred = model(X)
            loss = mse(Y_pred, Y)
            loss.backward(retain_graph=True)
            # Store the loss and gradients
            losses[i, j] = loss.item()
            grads_x[i, j] = p1_param.grad[p1_index]
            grads_y[i, j] = p2_param.grad[p2_index]
            model.zero_grad()

    # Plotting the loss surface
    plt.figure(figsize=(10, 8))
    plt.axis('equal')
    contour = plt.contourf(P1, P2, losses, levels=50, cmap='viridis')
    plt.colorbar(contour)  # Add a color bar to the contour plot
    plt.title(title)
    starting_coordinate = (starting_p1, starting_p2)  # Define your starting P1, P2 coordinates
    plt.plot(starting_coordinate[0], starting_coordinate[1], 'ro')  # 'ro' for red circle
    # Overlay the gradient field as a quiver plot
    # Use a contrasting color for the arrows, such as white
    plt.show()
    #restore previous values for model:
    with torch.no_grad():
        p1_param[p1_index] = starting_p1
        p2_param[p2_index] = starting_p2
class rotationNode(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.rand(2)*10000)
        self.bias = nn.Parameter(torch.rand(1) - 1)
        self.w = nn.Parameter(torch.rand(2)*torch.pi)
        self.b = nn.Parameter(torch.rand(1) - 0.5)
    def forward(self,x):
        scaled = torch.sigmoid(self.scale*(x+self.bias))
        sums = torch.sum(self.w*scaled,dim=-1) + self.b
        return torch.sin(sums)
correctXorNode = rotationNode()
correctXorNode.scale.data = torch.tensor([100000.00,100000.00])
correctXorNode.bias.data = torch.tensor([-.5])
correctXorNode.w.data = torch.tensor([torch.pi/2,torch.pi/2])
correctXorNode.b.data= torch.tensor([0.0])

X = torch.rand(2000,2)
Y = correctXorNode(X).detach()
epochs = 200
train_loader = DataLoader(TensorDataset(X,Y),batch_size=1024)
model = rotationNode()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01,momentum=0.9)
with torch.no_grad():
    model.scale.data = torch.tensor([1555.7992, 9947.4170])
    model.bias.data = torch.tensor([-0.6989])
    model.w.data = torch.tensor([0.3230, 0.4581])
    model.b.data = torch.tensor([0.2809])
for name, param in model.named_parameters():
    print (name, param)

for epoch in range(epochs):
    total_loss = 0.0
    model.train()
    if epoch % 10 == 0:
        visualize_loss_and_gradient(model, (model.b, 0), (model.bias, 0), [-torch.pi * 2, torch.pi * 2],
                                [-torch.pi * 2, torch.pi * 2], X, "before_{0}".format(epoch))
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        batch_X = batch_X
        batch_y = batch_y
        outputs = model(batch_X)
        loss = mse(outputs, batch_y)
        loss.backward()
        optimizer.step()
        print(model.b.grad)
        total_loss += loss.item()
with torch.no_grad():
    outputs = model(X[:100])
    loss = mse(outputs, Y[:100])
    print('Test Loss:', loss.item())
exit(0);