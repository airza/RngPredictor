import numpy as np
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
torch.set_printoptions(precision=4,sci_mode=False)
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
with torch.no_grad():
    correctXorNode.scale.data = torch.tensor([100000.00,100000.00])
    correctXorNode.bias.data = torch.tensor([-.5])
    correctXorNode.w.data = torch.tensor([torch.pi/2,torch.pi/2])
    correctXorNode.b.data= torch.tensor([0.0])

all_X = torch.rand(1000000,2)
print(all_X[:10])
print(correctXorNode(all_X)[:10])
exit(0)
def lossFunction(X,model):
    y = model(X)
    correctY = correctXorNode(X)
    return torch.mean((y-correctY)**2)
learnNode = rotationNode(False)
print(lossFunction(all_X,learnNode))


def visualize_loss_and_gradient(model, param1, param2, x_range, y_range, resolution=50,epsilon=1e-6):
    x = np.linspace(x_range[0], x_range[1], resolution)
    y = np.linspace(y_range[0], y_range[1], resolution)
    X, Y = np.meshgrid(x, y)
    losses = np.zeros_like(X)
    grads_x = np.zeros_like(X)
    grads_y = np.zeros_like(X)

    for i in range(resolution):
        for j in range(resolution):
            # Temporarily set the model weights to these points
            param1.data.fill_(X[i, j])
            param2.data.fill_(Y[i, j])

            # Calculate loss
            loss = lossFunction(all_X, model)
            loss.backward()

            # Store the loss and gradients
            losses[i, j] = loss.item()
            grads_x[i, j] = param1.grad.item()
            grads_y[i, j] = param2.grad.item()

            # Reset gradients for next iteration
            model.zero_grad()

    # Plotting the loss surface
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    plt.contourf(X, Y, losses, levels=50, cmap='viridis')
    plt.colorbar()
    plt.title('Loss surface')

    # Plotting the gradients as a quiver plot
    plt.subplot(1, 2, 2)
    plt.quiver(X, Y, grads_x, grads_y)
    plt.title('Gradient field')
    plt.show()
visualize_loss_and_gradient(learnNode, learnNode.w[0], learnNode.w[1], [0, 1], [0,1])