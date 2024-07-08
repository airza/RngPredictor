import torch
from torch import nn as nn
from AddWithCarryNode import AddWithCarryNode
import itertools
torch.set_printoptions(precision=4,sci_mode=False)


class AddWithCarryNetwork(nn.Module):
    def __init__(self, inputWidth):
        super().__init__()
        self.node = AddWithCarryNode(inputWidth)
    def forward(self, x,y):
        carry = 0.0
        output = torch.zeros_like(x)
        for i in reversed(range(0,x.shape[1])):
            out = self.node(torch.stack([x[:,i],y[:,i],torch.tensor([carry])],dim=1))
            output[:,i] = out[:,0]
            carry = out[:,1]
        return output
adder = AddWithCarryNode(3)
activations = [
    [0, 0, 0],
    [0, 0, 1],
    [0, 1, 0],
    [0, 1, 1],
    [1, 0, 0],
    [1, 0, 1],
    [1, 1, 0],
    [1, 1, 1]
]
outputs = [
    [0,0],
    [1,0],
    [1,0],
    [0,1],
    [1,0],
    [0,1],
    [0,1],
    [1,1]
]
adder.truthTableInputs = torch.tensor(activations).float().T *2 -1
adder.truthTableOutputs = torch.tensor(outputs).float()
adder.certainty.data = torch.tensor([5]).float()
adderNetwork = AddWithCarryNetwork(3)
adderNetwork.node = adder
x = torch.tensor([0,1,1,1,0,1,1,1,1]).float().unsqueeze(0)
y = torch.tensor([1,0,1,0,0,0,0,0,1]).float().unsqueeze(0)
print(adderNetwork(x,y))