import torch.nn as nn
import torch
class Model(nn.Module):
    def __init__(self):
        innerDim = 16
        super(Model, self).__init__()
        self.inn = nn.Linear(128, innerDim)
        self.sigin = nn.Sigmoid()
        self.transformer = nn.Transformer(d_model=innerDim, nhead=4, num_encoder_layers=1, num_decoder_layers=1, dim_feedforward=8,activation=nn.ELU(),dropout=0.0)
        self.out = nn.Linear(innerDim, 1)
        self.sig = nn.Tanh()

    def forward(self, x):
        x = self.inn(x)
        x = self.sigin(x)
        x = self.transformer(x,x)
        x = self.out(x)
        x = self.sig(x)
        return x