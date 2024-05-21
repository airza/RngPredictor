#!/usr/bin/env python3
import torch.nn as nn
import torch
import numpy as np
device = torch.device("mps")
torch.set_default_device(device)
bits = 4
mse = nn.MSELoss()

# Example usage


def smooth_ceiling(x,k=2):
    return smooth_floor(x,k) + 1
x = torch.linspace(0, 10, 400)

def diff_mod(x, x_low, x_high):
    """Differentiable modulo function between x_low
    and x_high."""
    x = np.pi / (x_high - x_low) * (x - x_low)
    y = torch.arctan(-1.0 / (torch.tan(x))) + 0.5 * np.pi
    y = (x_high - x_low) / np.pi * y + x_low
    return y
import matplotlib.pyplot as plt
plt.plot(x.cpu(), smooth_floor(x).cpu(), label='Smooth Floor')
plt.plot(x.cpu(), torch.floor(x).cpu(), label='Floor', linestyle='dashed')
plt.plot(x.cpu(), smooth_ceiling(x).cpu(), label='Smooth Ceiling')
plt.plot(x.cpu(), torch.ceil(x).cpu(), label='Ceiling', linestyle='dashed')
plt.plot(x.cpu(), diff_mod(smooth_floor(x),0,7).cpu(), label='Diff Mod')
plt.legend()
plt.show()
exit(0)
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
    return torch.exp(x) / (1+torch.sum(torch.exp(x),dim=dim)).unsqueeze(1)
class rotationNode(nn.Module):
    def __init__(self,inputBits):
        super().__init__()
        self.intro = nn.Parameter(torch.rand(inputBits,2))
        self.w = nn.Parameter(torch.rand(4),requires_grad=True)
        self.certainty = nn.Parameter(torch.ones(1),requires_grad=True)
        self.beppis = nn.Parameter(torch.rand(1),requires_grad=True)
        # 1024,2 * 4x2  * 4 x 1
    def forward(self,x):
        x = torch.roll(x,self.beppis.int().item(),1)
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

def make_correct_node(bits,type,position1,position2,certainty=5):
    activations = {
        'xor':[0,1,1,0],
        'and':[0,0,0,1],
        'or':[0,1,1,1]
    }[type]
    activations = [(x-.5)*2 for x in activations]
    node = rotationNode(bits)
    r1 = [0 if i != position1 else 2 for i in range(bits)]
    r2 = [0 if i != position2 else 2 for i in range(bits)]
    node.intro.data = torch.tensor([r1,r2]).T.float()
    node.w.data = torch.tensor(activations).float()
    node.certainty.data = torch.tensor([certainty]).float()
    return node


class multiNode(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = rotationNode(bits)
        self.l2 = rotationNode(bits)
        self.l3 = rotationNode(2)
    def forward(self,_in):
        x = self.l1(_in)
        y = self.l2(_in)
        return self.l3(torch.stack([x,y],dim=1))
        # 1024,2 * 4x2  * 4 x 1
count = 2048
test_count = 100
X = torch.rand(count+test_count,bits)
X_test, X_train = X[:test_count], X[test_count:]
rounded = torch.round(X)
Y_1 = torch.logical_xor(rounded[:,0],rounded[:,1])
Y_2 = torch.logical_or(rounded[:,2],rounded[:,3])
Y = torch.logical_and(Y_1,Y_2).float()
Y_test, Y_train = Y[:test_count], Y[test_count:]
epochs = 100000
train_loader = DataLoader(TensorDataset(X_train,Y_train),batch_size=4)
model = multiNode()
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
    if total_loss / len(train_loader) < 0.02:
        break
model.l1 = make_correct_node(bits,'and',0,1)
model.l2 = make_correct_node(bits,'xor',2,3)
model.l3 = make_correct_node(2,'and',0,1)
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