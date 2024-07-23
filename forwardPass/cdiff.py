import torch
from forwardPass.utils import int_to_bits_tensor, float_bits_to_int, print_comparable_bits
from XorShift128PlusNetwork import XorShift128NN
import random
torch.set_printoptions(precision=1,sci_mode=True)

from setupForwardPass import xorshift128plus

oksure = lambda num: format(num, '064b')
MAXSIZE = 0xFFFFFFFFFFFFFFFF

A,B= (7436933115282745233,824923114851450833)
generated = int_to_bits_tensor(15687642868793718113)
x = int_to_bits_tensor(824923114851450833)
y = int_to_bits_tensor(14862719753942267280)
A_bit = int_to_bits_tensor(A)
B_bit = int_to_bits_tensor(B)
model = XorShift128NN()
for param in model.parameters():
    param.requires_grad = False
make_input = lambda: (torch.zeros(64)+.5).requires_grad_(True)
def make_modified_input(x,pos):
    x2 = x.clone().requires_grad_(True)
    with torch.no_grad():
        x2[pos] = torch.rand(len(pos))
    return x2
test_A, test_B =  make_modified_input(A_bit,range(len(A_bit))), make_modified_input(B_bit,[])
for i in range(30000):
    test_generated, test_x, test_y = model(torch.cat([test_A, test_B], dim=0).unsqueeze(0))
    loss = torch.nn.functional.mse_loss(torch.cat([generated,x,y],dim=0),torch.cat([test_generated, test_x, test_y], dim=0).reshape(-1))
    loss.backward()
    if i%200==0:
        print(loss.item())
        tol = .1
        beppis = torch.abs(A_bit-test_A)
        indices = torch.nonzero(beppis > tol).flatten()
        print(test_A.grad[indices[:]])
        print(torch.sign((test_A- A_bit)[indices[:]]/test_A.grad[indices[:]]))
        if loss.item()<.005:
            break
    with torch.no_grad():
        clamp = lambda x: x
        test_A -= clamp(test_A.grad)
        test_B -= clamp(test_B.grad)
        torch.clamp(test_A,0,1,out=test_A)
        torch.clamp(test_B,0,1,out=test_B)
print("DONE")
print(test_A,test_B)
new_A, new_B= float_bits_to_int(test_A),float_bits_to_int(test_B)
print(new_A, new_B)
print("TEST")
print_comparable_bits(new_A,A)
print_comparable_bits(new_B,B)
exit(0)

