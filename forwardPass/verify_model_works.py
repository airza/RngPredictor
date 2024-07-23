import torch
from forwardPass.utils import int_to_bits_tensor, float_bits_to_int
from XorShift128PlusNetwork import XorShift128NN
import random
MAXSIZE = 0xFFFFFFFFFFFFFFFF
def xorshift128plus(x, y):
    s0, s1 = y, x
    s1 ^= (s1 << 23) & MAXSIZE
    s1 ^= (s1 >> 17)
    s1 ^= s0
    s1 ^= (s0 >> 26)
    x = y
    y = s1
    generated = (x + y) & MAXSIZE
    return generated, x, y,
def transform_vector(vector):
    zero_randoms = torch.rand_like(vector) * 0.5
    one_randoms = 0.5 + (torch.rand_like(vector) * 0.5)
    return torch.where(vector == 0, zero_randoms, one_randoms)
for i in range(1000):
    A = random.randint(0,MAXSIZE)
    B = random.randint(0,MAXSIZE)
    x = transform_vector(int_to_bits_tensor(A))
    y = transform_vector(int_to_bits_tensor(B))
    model = XorShift128NN()
    a,b,c = xorshift128plus(A,B)
    d,e,f = model(torch.concat([x,y],dim=0).unsqueeze(0))
    if a!=float_bits_to_int(d) or b!=float_bits_to_int(e) or c!=float_bits_to_int(f):
        print(f"fail on:{i}")
        print(A,B)
        print(a,b,c)
        print(*[float_bits_to_int(i) for i in [d,e,f]])
        break
print("done")