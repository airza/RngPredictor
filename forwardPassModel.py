import torch.nn as nn
import torch
class Model(nn.Module):
    def __init__(self):
        innerDim = 2
        super(Model, self).__init__()
        self.inn = nn.Linear(128, innerDim)
        self.sigin = nn.ReLU()
        self.transformer = nn.Transformer(d_model=innerDim, nhead=2, num_encoder_layers=2, num_decoder_layers=1, dim_feedforward=3,dropout=0.1)
        self.out = nn.Linear(innerDim, 1)
        self.sig = nn.Tanh()

    def forward(self, x):
        x = x*torch.rand(x.shape[0],128).to(x.device)
        x = self.inn(x)
        x = self.sigin(x)
        x = self.transformer(x,x)
        x = self.out(x)
        x = self.sig(x)
        return x