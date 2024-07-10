import torch
from torch import nn as nn
from DifferentiableShiftNetwork import DifferentiableShiftNetwork
from LogicGateNetwork import LogicGateNetwork
class XorShift128NN(nn.Module):
    def __init__(self):
        super().__init__()
        self.shift1 = DifferentiableShiftNetwork(64,5,23)
        self.shift2 = DifferentiableShiftNetwork(64,5,-17)
        self.xor = LogicGateNetwork('xor')
    def forward(self,x):
        n1 = self.shift1(x)
        n2 = self.xor(torch.stack([x,n1],dim=1))
        n3 = self.shift2(n2)
        n4 = self.xor(torch.stack([n2,n3],dim=1))
        return n4
