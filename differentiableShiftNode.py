import torch
import numpy as np
import matplotlib.pyplot as plt
torch.set_printoptions(precision=4,sci_mode=False)
bits = 4
pos = 2
def gaussian(x, n, k=2, sigma=2):
    return k * torch.exp(-((x - n) ** 2) / (2 * sigma ** 2))
def smooth_brain(x, n, k=2, sigma=2):
    return torch.softmax(gaussian(x, n, k, sigma), 0)
def bit_to_minus(n,bits=bits):
    return n-(bits-1)
def brange(bits):
    return range(1-bits,bits)
def bit_shift_matricies(bits):
    eyes = torch.stack([torch.roll(torch.eye(bits), i, 0) for i in brange(bits)])
    for i in range(2 * bits - 1):
        b = bit_to_minus(i)
        if b<0:
            eyes[i][:][b:] = 0.0
        elif b>=0:
            eyes[i][:][:b] = 0.0
    return eyes
def differentiable_shift(x,n,eyes,certainty,bits=bits):
    shift = smooth_brain(torch.arange(1-bits,bits,1),n,certainty,1)
    shift /= shift.sum()
    mults = torch.sum(shift[:,None,None]*eyes,0)
    return torch.matmul(x,mults)
matricies = bit_shift_matricies(bits)
y = torch.tensor([1,2,3,4]).float()
print(differentiable_shift(y,1,matricies,10))
exit(0)
g = smooth_brain(torch.linspace(0, bits-1, bits), pos, 10, 1)
prod = torch.matmul(eyes,g)
print(eyes[pos])
print(prod)
print(torch.matmul(prod,y))
x = torch.linspace(0, 10, 100)
exit(0)
