import torch
from forwardPass.utils import int_to_bits_tensor, float_bits_to_int
from XorShift128PlusNetwork import XorShift128NN
import random

from setupForwardPass import xorshift128plus

oksure = lambda num: format(num, '064b')
MAXSIZE = 0xFFFFFFFFFFFFFFFF

A,B= (7436933115282745233,824923114851450833)
generated = int_to_bits_tensor(15687642868793718113)
x = int_to_bits_tensor(824923114851450833)
y = int_to_bits_tensor(14862719753942267280)
model = XorShift128NN()
for param in model.parameters():
    param.requires_grad = False
make_input = lambda: (torch.zeros(64)/1000 + .5).requires_grad_(True)
test_A, test_B = make_input(), make_input()
for i in range(3000):
    test_generated, test_x, test_y = model(torch.cat([test_A, test_B], dim=0).unsqueeze(0))
    loss = torch.nn.functional.mse_loss(torch.cat([generated,x,y],dim=0),torch.cat([test_generated, test_x, test_y], dim=0).reshape(-1))
    loss.backward()
    if i%50==0:
        print(loss.item())
        if loss.item()<.001:
            break
    with torch.no_grad():
        clamp = lambda x: x*.1
        test_A -= clamp(test_A.grad)
        test_B -= clamp(test_B.grad)
        torch.clamp(test_A,0,1,out=test_A)
        torch.clamp(test_B,0,1,out=test_B)
print("DONE")
new_x, new_y= float_bits_to_int(test_A),float_bits_to_int(test_B)
print(new_x, new_y)
print("TEST")
print(xorshift128plus(14955768359507090826,824923114851450833))
exit(0)

