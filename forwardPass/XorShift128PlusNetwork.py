import torch
from torch import nn as nn
from DifferentiableShiftNetwork import DifferentiableShiftNetwork
from LogicGateNetwork import LogicGateNetwork
from forwardPass.AddWithCarryNetwork import AddWithCarryNetwork


class XorShift128NN(nn.Module):
    def __init__(self):
        super().__init__()
        self.shift1 = DifferentiableShiftNetwork(64,3,23)
        self.shift2 = DifferentiableShiftNetwork(64,3,-17)
        self.shift3 = DifferentiableShiftNetwork(64,3,-26)
        self.xor = LogicGateNetwork('xor')
        self.adder = AddWithCarryNetwork(3)
    def forward(self,x):
        f = 10
        x *=f
        x-=(f/2)
        x = torch.sigmoid(x)
        x, y = x.chunk(2,dim=1)
        n1 = self.shift1(x)
        n2 = self.xor(torch.stack([x,n1],dim=1))
        n3 = self.shift2(n2)
        n4 = self.xor(torch.stack([n2,n3],dim=1))
        n5 = self.xor(torch.stack([n4,y,],dim=1))
        n6 = self.shift3(y)
        n7 = self.xor(torch.stack([n5,n6],dim=1))
        xout = y
        yout = n7
        out = self.adder(xout,yout)
        return out, xout, yout
