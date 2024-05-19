#!/usr/bin/env python3
import numpy as np
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
device = torch.device("mps")
torch.set_default_device(device)
bits = 64
mse = nn.MSELoss()
def a_to_b(thetas):
    theta = thetas[0]
    phi = thetas[1]
    x = torch.sin(theta) * torch.cos(phi)
    y = torch.sin(theta) * torch.sin(phi)
    z = torch.cos(theta)
    return torch.tensor([x,y,z],requires_grad=True)

# Example usage:
def visualize_loss_and_gradient(model, p1_struct,p2_struct, p1_range, p2_range,X,title,resolution=30,target_coordinate=None):
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
    if target_coordinate is not None:
        plt.plot(target_coordinate[0], target_coordinate[1], 'go')
    # Overlay the gradient field as a quiver plot
    # Use a contrasting color for the arrows, such as white
    plt.show()
    #restore previous values for model:
    with torch.no_grad():
        p1_param[p1_index] = starting_p1
        p2_param[p2_index] = starting_p2

def softmax1(x,dim=1):
    return torch.exp(x) / (torch.sum(torch.exp(x),dim=dim).unsqueeze(dim))

class rotationNode(nn.Module):
    def __init__(self):
        super().__init__()
        self.intro = nn.Parameter(torch.rand(bits,2))
        self.w = nn.Parameter(torch.rand(4),requires_grad=True)
        self.certainty = nn.Parameter(torch.ones(1),requires_grad=True)
        # 1024,2 * 4x2  * 4 x 1
    def forward(self,x):
        intro = torch.softmax(self.intro*self.certainty,dim=0)
        xx = x -0.5
        xx = torch.matmul(xx,intro)
        sludge = torch.tanh(xx*self.certainty)
        target= torch.tensor([[-1,-1],[-1,1],[1,-1],[1,1]]).float().T
        prod = torch.matmul(sludge,target)
        best_name = torch.softmax(prod*self.certainty,dim=1)
        #return blended product of these three
        boys = torch.matmul(best_name,self.w)
        return torch.sigmoid(boys)
count = 2048
test_count = 100
X = torch.rand(count+test_count,bits)
X_test, X_train = X[:test_count], X[test_count:]
rounded = torch.round(X)
Y = torch.logical_xor(rounded[:,0],rounded[:,6]).float()
Y_test, Y_train = Y[:test_count], Y[test_count:]
epochs = 1000
train_loader = DataLoader(TensorDataset(X_train,Y_train),batch_size=8)
model = rotationNode()
optimizer = torch.optim.SGD(model.parameters(),lr=1)


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
    if total_loss / len(train_loader) < 0.02:
        break
    print(f'Epoch {epoch}, Loss: {total_loss / len(train_loader)}')
    print(model.certainty)
with torch.no_grad():
    outputs = model(X_test)
    loss = mse(outputs, Y_test)
    print('Test Loss:', loss.item())
    print(outputs,Y_test)
    print(model.w)