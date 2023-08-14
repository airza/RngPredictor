import torch.nn as nn
class Model(nn.Module):
    def __init__(self):
        innerDim = 8
        super(Model, self).__init__()
        self.inn = nn.Linear(128, innerDim)
        self.sigin = nn.Sigmoid()
        self.transformer = nn.Transformer(d_model=innerDim, nhead=8, num_encoder_layers=4, num_decoder_layers=2, dim_feedforward=64)
        self.out = nn.Linear(innerDim, 1)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = self.inn(x)
        x = self.sigin(x)
        x = self.transformer(x,x)
        x = self.out(x)
        x = self.sig(x)
        return x