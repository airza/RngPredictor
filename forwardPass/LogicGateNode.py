import torch
from torch import nn as nn
class LogicGateNode(nn.Module):
    def __init__(self,type=None):
        super().__init__()
        self.w = nn.Parameter(torch.rand(4),requires_grad=True)
        self.certainty = nn.Parameter(torch.ones(1),requires_grad=True)
        if type:
            truth_table = {
                'xor':[0,1,1,0],
                'and':[0,0,0,1],
                'or':[0,1,1,1],
                'nand':[1,1,1,0],
                'nor':[1,0,0,0],
                'xnor':[1,0,0,1],
                'always_true':[1,1,1,1],
                'always_false':[0,0,0,0]
            }[type]
            self.w.data = torch.tensor(truth_table).float()
            self.certainty.data= torch.tensor([2]).float()
    def forward(self,x):
        xx = (x -0.5) * 2
        sludge = torch.tanh(xx*self.certainty)
        target= torch.tensor([[-1,-1],[-1,1],[1,-1],[1,1]]).float().T
        prod = torch.matmul(sludge,target)
        best_name = torch.softmax(prod*self.certainty,dim=2)
        #return blended product of these three
        boys = torch.matmul(best_name,self.w)
        return boys