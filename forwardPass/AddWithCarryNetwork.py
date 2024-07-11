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