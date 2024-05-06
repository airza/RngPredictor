import numpy as np
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
torch.set_printoptions(precision=4,sci_mode=False)
mse = nn.MSELoss()
class rotationNode(nn.Module):
    def __init__(self,goodWeights=True):
        super().__init__()
        self.scale = nn.Parameter(torch.rand(2), requires_grad=True)
        self.bias = nn.Parameter(torch.rand(1) - 0.5, requires_grad=True)
        self.w = nn.Parameter(torch.rand(2), requires_grad=True)
        self.b = nn.Parameter(torch.rand(1) - 0.5, requires_grad=True)
    def forward(self,x):
        scaled = torch.sigmoid(self.scale*(x+self.bias))
        sums = torch.sum(self.w*scaled,dim=-1) + self.b
        return torch.sin(sums)
possibilities = torch.tensor(np.array([[1,1],[0,1],[1,0],[0,0]]))
correctXorNode = rotationNode()
correctXorNode.scale.data = torch.tensor([100000.00,100000.00])
correctXorNode.bias.data = torch.tensor([-.5])
correctXorNode.w.data = torch.tensor([torch.pi/2,torch.pi/2])
correctXorNode.b.data= torch.tensor([0.0])

X = torch.rand(1000000,2)
Y = correctXorNode(X)
error = mse(Y,Y)
learnNode = rotationNode(False)
def visualize_loss_and_gradient(model, p1_struct,p2_struct, p1_range, p2_range,X,resolution=10):
    p1_param,p1_index = p1_struct
    p2_param,p2_index = p2_struct
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
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    plt.contourf(P1, P2, losses, levels=50, cmap='viridis')
    plt.colorbar()
    plt.title('Loss surface')

    # Plotting the gradients as a quiver plot
    plt.subplot(1, 2, 2)
    plt.quiver(X_axis, Y_axis, grads_x, grads_y)
    plt.title('Gradient field')
    plt.show()
visualize_loss_and_gradient(learnNode, [learnNode.w,0],[learnNode.w,1], [0, 1], [0,1],X)
exit(0);