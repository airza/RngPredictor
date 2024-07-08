import torch
from torch import nn as nn


class LogicGateNode(nn.Module):
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
    node = LogicGateNode(bits)
    r1 = [0 if i != position1 else 2 for i in range(bits)]
    r2 = [0 if i != position2 else 2 for i in range(bits)]
    node.intro.data = torch.tensor([r1,r2]).T.float()
    node.w.data = torch.tensor(activations).float()
    node.certainty.data = torch.tensor([certainty]).float()
    return node


class Multinode(nn.Module):
    def __init__(self,bits):
        super().__init__()
        self.l1 = LogicGateNode(bits)
        self.l2 = LogicGateNode(bits)
        self.l3 = LogicGateNode(2)
    def forward(self,_in):
        x = self.l1(_in)
        y = self.l2(_in)
        return self.l3(torch.stack([x,y],dim=1))
        # 1024,2 * 4x2  * 4 x 1
