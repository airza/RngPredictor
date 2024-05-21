import torch
import numpy as np
import matplotlib.pyplot as plt
torch.set_printoptions(precision=4,sci_mode=False)
bits = 4
pos = 2
def smooth_floor(x,k=2):
    return x-(torch.sin(2*torch.pi*x)/(2*torch.pi*k))
def smooth_ceiling(x,k=2):
    return smooth_floor(x,k) + 1

def smooth_brain(x, n, k=2, sigma=2):
    return torch.softmax(k * torch.exp(-((x - n) ** 2) / (2 * sigma ** 2)), 0)

eyes = torch.stack([torch.roll(torch.eye(bits), i, 0) for i in range(bits)])
g = smooth_brain(torch.linspace(0, bits-1, bits), pos, 10, 1)
print(eyes[pos])
print(torch.matmul(eyes,g))
exit(0)
x = torch.linspace(0, 10, 100)
y = gaussian(x,6,2,2)
plt.plot(x.cpu(), y.cpu(), label='Smooth Floor')
plt.legend()
plt.show()