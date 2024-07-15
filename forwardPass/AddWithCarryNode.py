import torch
from torch import nn as nn
import itertools
torch.set_printoptions(precision=4,sci_mode=False)
def generate_combinations(n):
    combinations = list(itertools.product([-1, 1], repeat=n))
    tensor = torch.tensor(combinations).float().T
    return tensor

class AddWithCarryNode(nn.Module):
    def __init__(self, inputWidth,useDefault=True):
        super().__init__()
        self.truthTableInputs = generate_combinations(inputWidth)
        if useDefault:
            self.truthTableOutputs = torch.tensor([[0,0],
                                                    [1,0],
                                                    [1,0],
                                                    [0,1],
                                                    [1,0],
                                                    [0,1],
                                                    [0,1],
                                                    [1,1]]).float()
        else:
            self.truthTableOutputs= torch.randn([2 ** inputWidth, 2 ** inputWidth - 1]).float().T
        self.certainty = nn.Parameter(torch.ones(1)*2, requires_grad=True)
    def forward(self, x):
        x = 2*(x - 0.5)
        x = torch.matmul(x, self.truthTableInputs)
        x = torch.softmax(x* self.certainty, dim=1)
        x = torch.matmul(x, self.truthTableOutputs)
        return x