import torch
from torch import nn as nn
from forwardPass.LogicGateNode import LogicGateNode
torch.set_printoptions(precision=4,sci_mode=False)
class LogicGateNetwork(nn.Module):
    def __init__(self,type=None):
        super().__init__()
        if type:
            self.node = LogicGateNode(type)
        else:
            self.node = LogicGateNode()
    def forward(self,x):
        #apply same node to each column
        xx = torch.transpose(x,1,2)
        xxx = self.node(xx)
        return xxx